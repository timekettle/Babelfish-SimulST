"""llama.cpp backend: run the streaming policy on a GGUF model served by llama-server.

Start the server with the language model and the audio projector (best with the audio patch in patches/llama.cpp),
for example

  llama-server -m Babelfish-30B-A3B-Q8_0.gguf --mmproj mmproj-Babelfish-30B-A3B-F16.gguf -ngl 99 \
      --parallel 8 -c 32768 --port 8080

LlamaCppEngine has the same interface as the vLLM engine (run(requests, beams) -> [(suffix token ids, ended)]), so
the streaming policy is unchanged (python -m simulst.infer.stream --backend llamacpp ...); AsyncLlamaCppEngine is the
asyncio version used by the server (python -m simulst.server --backend llamacpp ...).

Decoding: greedy (one /completion per step), or beam search with the shared simulst.engines.beam.beam_search(): every
round asks for one next token with its top-2nb log-probabilities (n_predict 1, n_probs 2nb) for each live beam. Each
beam call reserves nb server slots and pins every request to one of them (id_slot) with cache_prompt on, so the
audio and the text before the beam prefix are reused from the slot's KV cache and a round only evaluates the new
tokens. Run llama-server with --parallel >= concurrent streams x beam width.

Differences from the vLLM engine:
  * llama-server takes a multimodal prompt as text, so the committed prefix (and a beam's prefix) is sent as decoded
    text and re-tokenized by the server (it almost always gives back the same ids; check_tokens=True counts mismatches);
  * a prefix that ends inside a character (incomplete UTF-8) cannot be sent as text: the partial tokens are left out
    of the prompt and must come out first (greedy: they are stripped from the output; beam search: the round asks
    for len(partial) + 1 tokens and reads the distribution after them; if the model does not reproduce them, that
    beam gets no continuation);
  * audio is sent as a 32-bit float WAV at 16 kHz (no re-quantization); llama.cpp computes its own log-mel
    features and wraps the audio embeddings in <|audio_start|> ... <|audio_end|>.
"""
import atexit
import base64
import http.client
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import numpy as np

from simulst.common import END_MARK, PROMPTS, SR
from simulst.engines.beam import beam_search

IM_END = 151645


def wav_f32(audio):
    """Mono 16 kHz IEEE-float WAV bytes."""
    data = np.ascontiguousarray(audio, dtype='<f4').tobytes()
    fmt = struct.pack('<HHIIHHH', 3, 1, SR, SR * 4, 4, 32, 0)          # WAVE_FORMAT_IEEE_FLOAT, cbSize = 0
    fact = b'fact' + struct.pack('<II', 4, len(data) // 4)
    body = b'WAVE' + b'fmt ' + struct.pack('<I', len(fmt)) + fmt + fact + b'data' + struct.pack('<I', len(data)) + data
    return b'RIFF' + struct.pack('<I', len(body)) + body


# ---------------------------------------------------------------- tokenizer from the GGUF vocabulary

def _bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def read_gguf_vocab(path):
    """(tokens, token_types) from the metadata of a GGUF file (no dependency on the gguf package)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    want = {'tokenizer.ggml.tokens', 'tokenizer.ggml.token_type'}
    out = {}
    with open(path, 'rb') as f:
        def u32():
            return struct.unpack('<I', f.read(4))[0]

        def u64():
            return struct.unpack('<Q', f.read(8))[0]

        def string():
            return f.read(u64()).decode('utf8', errors='surrogateescape')

        def value(t, keep):
            if t == 8:
                return string()
            if t == 9:
                et, n = u32(), u64()
                if et == 8:
                    items = [string() for _ in range(n)]
                    return items if keep else None
                raw = f.read(sizes[et] * n)
                if not keep:
                    return None
                dt = {0: '<u1', 1: '<i1', 2: '<u2', 3: '<i2', 4: '<u4', 5: '<i4', 6: '<f4', 7: '<u1', 10: '<u8', 11: '<i8', 12: '<f8'}[et]
                return np.frombuffer(raw, dtype=dt).tolist()
            f.read(sizes[t])
            return None

        assert f.read(4) == b'GGUF', path
        u32()          # version
        u64()          # tensor count
        for _ in range(u64()):
            key = string()
            v = value(u32(), key in want)
            if key in want:
                out[key] = v
                if len(out) == len(want):
                    break
    return out['tokenizer.ggml.tokens'], out['tokenizer.ggml.token_type']


# special tokens of the Qwen3-Omni tokenizer (Hugging Face `all_special_ids`); a generated suffix that contains one
# of them is rejected by the streaming loop, exactly as with the vLLM engine
QWEN_SPECIAL = ('<|endoftext|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<|video_pad|>',
                '<|audio_start|>', '<|audio_end|>', '<|audio_pad|>')
# added tokens that llama.cpp marks as control but the Hugging Face tokenizer decodes as text
QWEN_TEXT_ADDED = ('<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|fim_pad|>', '<|repo_name|>', '<|file_sep|>')


class GGUFTokenizer:
    """Byte-level BPE decoding with the GGUF vocabulary (decode-only; matches the Hugging Face tokenizer for Qwen:
    decode() skips all control tokens, all_special_ids is the tokenizer's special-token list)."""
    NORMAL, CONTROL, USER_DEFINED, UNUSED = 1, 3, 4, 5

    def __init__(self, tokens, types, special_ids=None):
        dec = {c: b for b, c in _bytes_to_unicode().items()}
        self.tokens, self.types = tokens, types
        self.pieces = []
        for t, ty in zip(tokens, types):
            if ty == self.NORMAL:
                self.pieces.append(bytes(dec[c] for c in t) if all(c in dec for c in t) else t.encode('utf8'))
            else:
                self.pieces.append(t.encode('utf8'))
        self._ids = {t: i for i, t in enumerate(tokens)}
        self._special = {i for i, (t, ty) in enumerate(zip(tokens, types))
                         if (ty == self.CONTROL and t not in QWEN_TEXT_ADDED) or ty == self.UNUSED}
        if special_ids is None:
            special_ids = [self._ids[t] for t in QWEN_SPECIAL] if all(t in self._ids for t in QWEN_SPECIAL) else self._special
        self.all_special_ids = sorted(special_ids)

    def convert_tokens_to_ids(self, token):
        return self._ids[token]

    def decode_bytes(self, ids, skip_special_tokens=True):
        return b''.join(self.pieces[i] for i in ids if not (skip_special_tokens and i in self._special))

    def decode(self, ids, skip_special_tokens=True):
        return self.decode_bytes(ids, skip_special_tokens).decode('utf8', errors='replace')


# ---------------------------------------------------------------- engine

class SlotPool:
    """Server slots reserved by the client, so that requests of one beam call can be pinned to their own slots."""

    def __init__(self, n):
        self.free = list(range(n))
        self.cv = threading.Condition()

    def acquire(self, k):
        with self.cv:
            while len(self.free) < k:
                self.cv.wait()
            got, self.free = self.free[:k], self.free[k:]
            return got

    def release(self, ids):
        with self.cv:
            self.free.extend(ids)
            self.cv.notify_all()


def _lcp(a, b):
    if a is None:
        return -1
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class LlamaCppEngine:
    def __init__(self, url='http://127.0.0.1:8080', tokenizer_dir=None, max_new=96, workers=None, timeout=600,
                 cache_prompt=False, check_tokens=None, retry_s=300, report_stats=None):
        # check_tokens / report_stats default to the environment (SIMULST_LLAMA_CHECK_TOKENS=1 / SIMULST_ENGINE_STATS=1)
        if check_tokens is None:
            check_tokens = os.environ.get('SIMULST_LLAMA_CHECK_TOKENS') == '1'
        if report_stats is None:
            report_stats = os.environ.get('SIMULST_ENGINE_STATS') == '1'
        u = urlparse(url)
        self.retry_s = retry_s
        self.lock = threading.Lock()
        self.stats = {}
        self.host, self.port = u.hostname, u.port or 80
        self.timeout, self.max_new, self.cache_prompt, self.check_tokens = timeout, max_new, cache_prompt, check_tokens
        self._local = threading.local()
        props = self._call('GET', '/props')
        if not (props.get('modalities') or {}).get('audio'):
            raise RuntimeError('llama-server has no audio input; start it with --mmproj <mmproj gguf>')
        self.marker = props.get('media_marker') or '<__media__>'
        self.slots = int(props.get('total_slots') or 1)
        self.model_path = props.get('model_path')
        if tokenizer_dir:          # Hugging Face tokenizer files (same vocabulary)
            from transformers import AutoTokenizer
            hf = AutoTokenizer.from_pretrained(tokenizer_dir)
            control = {i for i, t in hf.added_tokens_decoder.items() if t.special} | set(hf.all_special_ids)
            tokens = hf.convert_ids_to_tokens(list(range(len(hf))))
            self.tok = GGUFTokenizer(tokens, [GGUFTokenizer.CONTROL if i in control else GGUFTokenizer.NORMAL
                                              for i in range(len(tokens))], special_ids=hf.all_special_ids)
        else:                      # vocabulary stored in the GGUF file the server loaded
            self.tok = GGUFTokenizer(*read_gguf_vocab(self.model_path))
        self.special = set(self.tok.all_special_ids)
        self.n_regular = min(self.special)
        assert self.tok.convert_tokens_to_ids('<|im_end|>') == IM_END
        self.slot_pool = SlotPool(self.slots)
        self.pool = ThreadPoolExecutor(max_workers=workers or self.slots)       # greedy requests
        self.inner = ThreadPoolExecutor(max_workers=self.slots)                 # requests of beam rounds
        self.beam_pools = {}                                                    # beam width -> pool of beam calls
        self.head = {d: f'<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n' for d, (s, _) in PROMPTS.items()}
        self.tail = {(d, m): f'{u}{END_MARK if m else ""}<|im_end|>\n<|im_start|>assistant\n'
                     for d, (_, u) in PROMPTS.items() for m in (False, True)}
        self.stats.update(requests=0, prompt_ms=0.0, gen_ms=0.0, wall_ms=0.0, retokenized_mismatch=0, checked=0,
                          partial_tail=0, partial_tail_kept=0, partial_head=0, lossy=0,
                          beam_calls=0, beam_rounds=0, beam_requests=0, beam_prompt_n=0, beam_cache_n=0,
                          beam_partial=0, beam_partial_dead=0)
        self.walls, self.step_walls = [], []
        if report_stats:
            atexit.register(lambda: print(json.dumps(dict(event='backend_stats', **self.summary())), flush=True))

    # -- HTTP
    def _conn(self):
        c = getattr(self._local, 'conn', None)
        if c is None:
            c = self._local.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return c

    def _call(self, method, path, body=None):
        """One HTTP call; connection errors are retried for up to `retry_s` seconds (e.g. while a crashed server
        is being restarted by a supervisor)."""
        data = None if body is None else json.dumps(body).encode()
        t0, attempt = time.monotonic(), 0
        while True:
            c = self._conn()
            try:
                c.request(method, path, body=data, headers={'Content-Type': 'application/json'} if data else {})
                r = c.getresponse()
                raw = r.read()
                if r.status == 503:          # model still loading (e.g. after a restart)
                    raise ConnectionError(f'llama-server {path} -> HTTP 503')
                if r.status != 200:
                    raise RuntimeError(f'llama-server {path} -> HTTP {r.status}: {raw[:500]!r}')
                return json.loads(raw)
            except (ConnectionError, http.client.HTTPException, OSError):
                c.close()
                self._local.conn = None
                attempt += 1
                if time.monotonic() - t0 > self.retry_s:
                    raise
                with self.lock:
                    self.stats['retries'] = self.stats.get('retries', 0) + 1
                if attempt > 1:              # the first retry is immediate (stale keep-alive connection)
                    time.sleep(min(10, attempt - 1))

    def tokenize(self, text):
        return self._call('POST', '/tokenize', dict(content=text, add_special=False, parse_special=True))['tokens']

    # -- prompt
    def _anchor_text(self, anchor):
        """Prefix as text. Incomplete UTF-8 cannot be sent as text: leading partial tokens (old context) are dropped;
        trailing ones are returned as `pending` and must come out first."""
        ids = list(anchor)
        if not ids:
            return '', [], 0
        head = 0
        while head < min(4, len(ids)) and '\ufffd' in self.tok.decode(ids[head:head + 1], False):
            head += 1
        body, pending = ids[head:], []
        while body and len(pending) < 4:
            try:
                return self.tok.decode_bytes(body, False).decode('utf8'), pending, head
            except UnicodeDecodeError:
                pending.insert(0, body.pop())
        body += pending                      # broken bytes inside the prefix: send them lossy
        with self.lock:
            self.stats['lossy'] += 1
        return self.tok.decode(body, False), [], head

    def _check_tokens(self, q, ids, text, dropped, pending):
        if not (self.check_tokens and text):
            return None
        exp = self.tokenize(self.tail[q.direction, q.mark]) + list(ids)[dropped:len(ids) - len(pending)]
        return self.tokenize(self.tail[q.direction, q.mark] + text) != exp

    def _count(self, r, wall, mism, **extra):
        tm = r.get('timings') or {}
        with self.lock:
            s = self.stats
            s['prompt_ms'] += float(tm.get('prompt_ms') or 0)
            s['gen_ms'] += float(tm.get('predicted_ms') or 0)
            s['wall_ms'] += wall
            if mism is not None:
                s['checked'] += 1
                s['retokenized_mismatch'] += mism
            for k, v in extra.items():
                s[k] = s.get(k, 0) + v

    # -- greedy
    def _one(self, q):
        text, pending, dropped = self._anchor_text(q.anchor)
        prompt = self.head[q.direction] + self.marker + self.tail[q.direction, q.mark] + text
        body = dict(prompt=dict(prompt_string=prompt, multimodal_data=[base64.b64encode(wav_f32(q.audio)).decode()]),
                    n_predict=self.max_new + len(pending), temperature=0.0, top_k=1, cache_prompt=self.cache_prompt,
                    return_tokens=True, stream=False)
        slot, = self.slot_pool.acquire(1)
        try:
            body['id_slot'] = slot
            t0 = time.perf_counter()
            r = self._call('POST', '/completion', body)
            wall = (time.perf_counter() - t0) * 1000
        finally:
            self.slot_pool.release([slot])
        ids = list(r.get('tokens') or [])
        ended = r.get('stop_type') == 'eos'
        if ids and ids[-1] == IM_END:
            ids, ended = ids[:-1], True
        kept = True
        if pending:
            if ids[:len(pending)] == pending:
                ids = ids[len(pending):]
            else:
                kept = False
        if len(ids) > self.max_new:
            ids, ended = ids[:self.max_new], False
        mism = self._check_tokens(q, q.anchor, text, dropped, pending)
        self._count(r, wall, mism, requests=1, partial_tail=int(bool(pending)), partial_tail_kept=int(bool(pending) and kept),
                    partial_head=int(bool(dropped)))
        with self.lock:
            self.walls.append(wall)
        return ids, ended

    def greedy(self, reqs):
        """[Request] -> [(suffix, ended)]"""
        return list(self.pool.map(self._one, reqs))

    # -- beam search
    def _next_top(self, q, b64, prefix, k, slot, cache=True):
        """Top-k (token id, log P) of the next token after committed prefix + beam prefix, in a pinned slot.
        cache=False evaluates the whole prompt afresh (first round), exactly like a greedy request."""
        ids = list(q.anchor) + list(prefix)
        text, pending, dropped = self._anchor_text(ids)
        prompt = self.head[q.direction] + self.marker + self.tail[q.direction, q.mark] + text
        body = dict(prompt=dict(prompt_string=prompt, multimodal_data=[b64]), n_predict=len(pending) + 1,
                    temperature=0.0, n_probs=k, cache_prompt=cache, id_slot=slot, return_tokens=True, stream=False)
        t0 = time.perf_counter()
        r = self._call('POST', '/completion', body)
        wall = (time.perf_counter() - t0) * 1000
        got = list(r.get('tokens') or [])
        probs = r.get('completion_probabilities') or []
        dead = bool(pending) and (got[:len(pending)] != pending or len(probs) <= len(pending))
        top = [] if dead or len(probs) <= len(pending) else \
            [(int(t['id']), float(t['logprob'])) for t in probs[len(pending)].get('top_logprobs', [])][:k]
        tm = r.get('timings') or {}
        mism = self._check_tokens(q, ids, text, dropped, pending)
        self._count(r, wall, mism, beam_requests=1, beam_prompt_n=int(tm.get('prompt_n') or 0),
                    beam_cache_n=int(tm.get('cache_n') or 0), beam_partial=int(bool(pending)), beam_partial_dead=int(dead),
                    partial_head=int(bool(dropped)))
        return top, (top[0][0] if top else None)

    def _beam_one(self, q, nb, length_penalty=1.0):
        """Beam search for one request; returns (suffix, ended)."""
        t0 = time.perf_counter()
        b64 = base64.b64encode(wav_f32(q.audio)).decode()
        slots = self.slot_pool.acquire(nb)
        held = {s: None for s in slots}          # slot -> token prefix whose KV it holds
        try:
            g = beam_search(1, nb, self.max_new, length_penalty)
            jobs = next(g)
            rounds = 0
            while True:
                rounds += 1
                prefixes = [p for _, p in jobs]
                free, use = list(held), [None] * len(prefixes)
                for j in sorted(range(len(prefixes)), key=lambda j: -len(prefixes[j])):
                    s = max(free, key=lambda s: _lcp(held[s], prefixes[j]))
                    use[j] = s
                    free.remove(s)
                # first round: fresh prompt evaluation (same numerics as greedy); later rounds reuse the slot's KV
                res = list(self.inner.map(lambda js: self._next_top(q, b64, js[0], 2 * nb, js[1], cache=rounds > 1),
                                          zip(prefixes, use)))
                for (top, best), p, s in zip(res, prefixes, use):
                    held[s] = list(p) + ([best] if best is not None else [])
                try:
                    jobs = g.send([top for top, _ in res])
                except StopIteration as e:
                    out = e.value[0]
                    break
        finally:
            self.slot_pool.release(slots)
        with self.lock:
            self.stats['beam_calls'] += 1
            self.stats['beam_rounds'] += rounds
            self.step_walls.append((time.perf_counter() - t0) * 1000)
        return out

    def beam(self, reqs, nb, length_penalty=1.0):
        """[Request] -> [(suffix, ended)]; at most slots // nb requests run at the same time."""
        nb = max(1, int(nb))
        if nb > self.slots:
            raise ValueError(f'beam width {nb} needs at least {nb} server slots (llama-server --parallel), got {self.slots}')
        if nb not in self.beam_pools:
            self.beam_pools[nb] = ThreadPoolExecutor(max_workers=max(1, self.slots // nb))
        return list(self.beam_pools[nb].map(lambda q: self._beam_one(q, nb, length_penalty), reqs))

    # -- interface shared with the vLLM engines
    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def run(self, reqs, beams=1, length_penalty=1.0):
        if beams > 1:
            return self.beam(reqs, beams, length_penalty)
        return self.greedy(reqs)

    def summary(self):
        def pct(w, name):
            w = sorted(w)
            if not w:
                return {}
            d = {f'{name}_p{q}_ms': round(w[min(len(w) - 1, int(q / 100 * len(w)))], 1) for q in (50, 90, 99)}
            d[f'{name}_mean_ms'] = round(sum(w) / len(w), 1)
            return d
        return dict(self.stats, **pct(self.walls, 'wall'), **pct(self.step_walls, 'beam_step'))


class AsyncLlamaCppEngine:
    """asyncio wrapper for the server: each session step runs in a thread (greedy: one request; beam search: one
    beam call, which reserves `beams` server slots)."""

    def __init__(self, url, tokenizer_dir=None, max_new=96, workers=None):
        self.sync = LlamaCppEngine(url, tokenizer_dir=tokenizer_dir or None, max_new=max_new, workers=workers)
        self.special, self.n_regular = self.sync.special, self.sync.n_regular

    def decode(self, ids):
        return self.sync.decode(ids)

    async def run(self, q, beams=1, length_penalty=1.0):
        import asyncio
        if beams > 1:
            if beams > self.sync.slots:
                raise ValueError(f'beam width {beams} needs llama-server --parallel >= {beams}')
            return await asyncio.to_thread(self.sync._beam_one, q, beams, length_penalty)
        return await asyncio.to_thread(self.sync._one, q)

    def shutdown(self):
        for p in [self.sync.pool, self.sync.inner, *self.sync.beam_pools.values()]:
            p.shutdown(wait=False)
