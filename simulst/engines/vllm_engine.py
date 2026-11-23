"""vLLM backends: a synchronous engine for offline replay and an asyncio engine for the live server.

Both build the same token-level prompt
    <|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|>{instruction}
    [ [audio complete]]<|im_end|>\\n<|im_start|>assistant\\n{committed prefix}
and decode greedily or with beam search (simulst.engines.beam, shared with the llama.cpp engine).
"""
import os
import sys
import uuid
from pathlib import Path

import numpy as np

from simulst.common import END_MARK, PROMPTS, SR
from simulst.engines.beam import IM_END, beam_search, split_end, top_k as _top

os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + str(Path(sys.executable).parent)   # ninja for vLLM warm-up
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')

AUDIO = 151675   # <|audio_pad|>


def bound_padding():
    """Pad audio features only to the signal length (instead of 300 s): features of the valid frames are identical,
    and the per-request CPU time in the vLLM front end drops a lot."""
    from transformers import WhisperFeatureExtractor
    orig = WhisperFeatureExtractor.__call__
    if getattr(orig, '_bounded', False):
        return

    def call(self, raw_speech, *args, **kwargs):
        waves = [raw_speech] if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1 else raw_speech
        maximum = max(len(w) for w in waves)
        padded = ((maximum + self.n_fft + self.hop_length - 1) // self.hop_length) * self.hop_length
        if kwargs.get('padding') == 'max_length':
            kwargs['max_length'] = min(self.n_samples, padded)
        return orig(self, raw_speech, *args, **kwargs)
    call._bounded = True
    WhisperFeatureExtractor.__call__ = call


class PromptBuilder:
    def __init__(self, tok):
        assert tok.convert_tokens_to_ids('<|audio_pad|>') == AUDIO and tok.convert_tokens_to_ids('<|im_end|>') == IM_END
        self.tok = tok
        self.special = set(tok.all_special_ids)
        self.n_regular = min(self.special)
        self.head, self.tail = {}, {}
        for d, (system, user) in PROMPTS.items():
            self.head[d] = tok.encode(f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|audio_start|>',
                                      add_special_tokens=False)
            for m in (False, True):
                self.tail[d, m] = tok.encode(f'<|audio_end|>{user}{END_MARK if m else ""}<|im_end|>\n<|im_start|>assistant\n',
                                             add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def __call__(self, q, extra=()):
        return dict(prompt_token_ids=self.head[q.direction] + [AUDIO] + self.tail[q.direction, q.mark] + list(q.anchor) + list(extra),
                    multi_modal_data={'audio': [(q.audio, SR)]}, multi_modal_uuids={'audio': [q.key]})


def engine_kwargs(model, dtype, gpu_mem, max_seqs, eager, tp, max_model_len, seed=0):
    return dict(model=model, dtype=dtype, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
                # no vision encoder in the weights: image / video limit 0 makes vLLM skip it
                enable_prefix_caching=True, limit_mm_per_prompt={'audio': 1, 'image': 0, 'video': 0}, seed=seed,
                max_num_seqs=max_seqs, enforce_eager=eager, tensor_parallel_size=tp)


class VLLMEngine:
    """Synchronous engine (offline replay): greedy(reqs) / beam(reqs, nb) -> [(suffix ids, ended)]."""

    def __init__(self, model, dtype='float16', gpu_mem=0.9, max_seqs=256, eager=False, tp=1, max_new=96, max_model_len=4096):
        os.environ['OMP_NUM_THREADS'] = '1'   # OpenMP threads in the parent process can deadlock the forked engine
        bound_padding()
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams
        self.max_new = max_new
        self.llm = LLM(**engine_kwargs(model, dtype, gpu_mem, max_seqs, eager, tp, max_model_len))
        self.prompts = PromptBuilder(self.llm.get_tokenizer())
        self.special, self.decode, self.n_regular = self.prompts.special, self.prompts.decode, self.prompts.n_regular

    def greedy(self, reqs):
        sps = [self.SP(temperature=0.0, max_tokens=self.max_new, stop_token_ids=[IM_END], skip_special_tokens=False,
                       detokenize=False) for _ in reqs]
        res = self.llm.generate([self.prompts(q) for q in reqs], sps, use_tqdm=False)
        return [split_end(r.outputs[0].token_ids) for r in res]

    def beam(self, reqs, nb, length_penalty=1.0):
        sp = self.SP(temperature=0.0, max_tokens=1, logprobs=2 * nb, skip_special_tokens=False, detokenize=False)
        g = beam_search(len(reqs), nb, self.max_new, length_penalty)
        jobs = next(g)
        while True:
            res = self.llm.generate([self.prompts(reqs[i], prefix) for i, prefix in jobs], [sp] * len(jobs), use_tqdm=False)
            try:
                jobs = g.send([_top(r.outputs[0].logprobs[0], 2 * nb) for r in res])
            except StopIteration as e:
                return e.value

    def run(self, reqs, beams, length_penalty=1.0):
        return self.greedy(reqs) if beams <= 1 else self.beam(reqs, beams, length_penalty)


class AsyncVLLMEngine:
    """asyncio engine (live server). Concurrent calls from different sessions are batched by vLLM."""

    def __init__(self, model, dtype='float16', gpu_mem=0.9, max_seqs=64, eager=False, tp=1, max_new=96, max_model_len=4096):
        bound_padding()
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        try:
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:   # older releases
            from vllm import AsyncLLMEngine as AsyncLLM
        self.SP = SamplingParams
        self.max_new = max_new
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs(model, dtype, gpu_mem, max_seqs, eager, tp, max_model_len)))
        self.prompts = PromptBuilder(AutoTokenizer.from_pretrained(model))
        self.special, self.decode, self.n_regular = self.prompts.special, self.prompts.decode, self.prompts.n_regular

    async def _one(self, prompt, sp):
        last = None
        async for out in self.engine.generate(prompt, sp, uuid.uuid4().hex):
            last = out
        return last

    async def greedy(self, q):
        sp = self.SP(temperature=0.0, max_tokens=self.max_new, stop_token_ids=[IM_END], skip_special_tokens=False, detokenize=False)
        r = await self._one(self.prompts(q), sp)
        return split_end(r.outputs[0].token_ids)

    async def beam(self, q, nb, length_penalty=1.0):
        import asyncio
        sp = self.SP(temperature=0.0, max_tokens=1, logprobs=2 * nb, skip_special_tokens=False, detokenize=False)
        g = beam_search(1, nb, self.max_new, length_penalty)
        jobs = next(g)
        while True:
            res = await asyncio.gather(*[self._one(self.prompts(q, prefix), sp) for _, prefix in jobs])
            try:
                jobs = g.send([_top(r.outputs[0].logprobs[0], 2 * nb) for r in res])
            except StopIteration as e:
                return e.value[0]

    async def run(self, q, beams, length_penalty=1.0):
        return await (self.greedy(q) if beams <= 1 else self.beam(q, beams, length_penalty))

    def shutdown(self):
        try:
            self.engine.shutdown()
        except Exception:
            pass
