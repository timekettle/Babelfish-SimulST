"""Window policy shared by offline replay (simulst.infer.stream), the live server and every backend.

Audio arrives in 320 ms blocks. After block k the model sees the audio window [max(0, t_k - W), t_k] and, as the
assistant prefix, the target tokens it has committed inside the window plus up to N earlier tokens of context
(see ctx_rule). It then generates the next target tokens, or <|im_end|> to wait for more audio. Generated tokens are
committed immediately and never revised. After the last block the model is called once more with the end marker in
the instruction, so that it finishes the translation.

Repetition guard (simulst.loop_guard, on by default): a step whose new tokens start a runaway loop is cut where the
copying starts, before anything is committed; other steps are unchanged.
"""
import math
import time

import numpy as np

from simulst import ctx_rule, loop_guard
from simulst.common import (DEFAULT_CTX_TOKENS, DEFAULT_MAX_NEW, DEFAULT_WINDOW_S, SR, STEP_SAMPLES as STEP,
                            TARGET_LANG)

MAX_WINDOW_S, MAX_ANCHOR = 60, 2048


class Policy:
    """Decoding settings. ctx_sets (sentence-final / incomplete-character token ids) are filled in by the engine."""

    def __init__(self, window=DEFAULT_WINDOW_S, ctx_tokens=DEFAULT_CTX_TOKENS, end_marker=True, max_new=DEFAULT_MAX_NEW,
                 loop_guard='on'):
        self.window, self.ctx_tokens, self.end_marker, self.max_new = float(window), int(ctx_tokens), bool(end_marker), int(max_new)
        assert loop_guard in ('on', 'off'), loop_guard
        self.loop_guard = loop_guard
        self.ctx_sets = None

    def guard_for(self, beams):
        return self.loop_guard == 'on'

    def bind_tokenizer(self, decode, n_regular):
        if self.ctx_tokens > 0:
            self.ctx_sets = ctx_rule.token_sets(decode, n_regular)


class Request:
    """One model call: audio window (float32, 16 kHz), committed prefix (token ids), end marker flag."""
    __slots__ = ('audio', 'anchor', 'mark', 'key', 'direction')

    def __init__(self, audio, anchor, mark, key, direction):
        self.audio, self.anchor, self.mark, self.key, self.direction = audio, anchor, mark, key, direction


class Stream:
    """State of one input stream. Block k (1-based) ends at sample min(k * 320 ms, n).

    wave: the whole input (offline replay), or None for a live stream that grows with append().
    decode: token ids -> text (special tokens skipped).
    uid: unique per stream; it keys vLLM's multimodal cache, so two streams must never share it.
    beams: beam size used for this stream (reported in the result).
    """

    def __init__(self, policy, decode, uid, direction, wave=None, item=None, beams=1):
        assert direction in TARGET_LANG, direction
        self.p, self.decode, self.uid, self.direction = policy, decode, uid, direction
        self.beams, self.guard = beams, policy.guard_for(beams)
        self.item = item or dict(id=uid, direction=direction)
        self.wave = np.zeros(0, np.float32) if wave is None else np.asarray(wave, np.float32)
        self.n = len(self.wave)
        self.committed, self.emit = [], []      # committed tokens; window end (sample) at which each was committed
        self.ws, self.anchor = 0, []            # window start (sample); committed prefix sent to the model
        self.disp, self.disp_t, self.emissions = '', [], []
        self.calls = self.limits = self.invalid = self.loop_guards = 0
        self.error, self.stopped_at = None, None
        self.trace = []
        self.started = time.perf_counter()

    # ---- audio
    def append(self, samples):
        if len(samples):
            self.wave = np.concatenate([self.wave, np.asarray(samples, np.float32)])
            self.n = len(self.wave)

    @property
    def blocks(self):
        """Number of blocks in the current input (the last one may be partial)."""
        return max(1, math.ceil(self.n / STEP))

    @property
    def full_blocks(self):
        return self.n // STEP

    def end_of(self, k):
        return min(k * STEP, self.n)

    # ---- policy
    def request(self, k, final):
        """Model input after block k (final = the extra call after the last block), 'limit', or None (no call)."""
        end = self.end_of(k)
        audio = self.wave[self.ws:end]
        if final and not self.p.end_marker:
            return None
        if len(audio) < STEP:
            return None
        if len(audio) > SR * MAX_WINDOW_S or len(self.anchor) > MAX_ANCHOR:
            self.error = 'resource limit reached (window or prefix too long)'
            self.stopped_at = end / SR
            return 'limit'
        return Request(audio, list(self.anchor), bool(final and self.p.end_marker), f'{self.uid}|{self.ws}|{end}|0',
                       self.direction)

    def commit(self, suffix, ended, k, special):
        end = self.end_of(k)
        src_s = end / SR
        suffix = list(suffix)
        hit_max = (not ended) and len(suffix) >= self.p.max_new
        guard = None
        if self.guard and suffix and not (set(suffix) & special):
            keep, rule = loop_guard.check(self.committed, suffix, src_s, hit_max)
            if rule:
                guard = dict(event='loop_guard', source_s=src_s, rule=rule, kept_tokens=keep, removed_tokens=len(suffix) - keep,
                             removed_text=self.decode(suffix[keep:]), original_suffix_ids=suffix)
                suffix, ended, hit_max = suffix[:keep], True, False
        gen = dict(event='generation', source_s=src_s, anchor_tokens=len(self.anchor), generated_suffix_ids=list(suffix),
                   actual_eos_generated=ended, hit_max_new_tokens=hit_max)
        self.calls += 1
        if guard:
            self.loop_guards += 1
            self.trace.append(guard)
            gen.update(loop_guard=guard['rule'])
        if set(suffix) & special:
            self.invalid += 1
            gen.update(candidate_rejected=True)
            suffix = []
        gen['generated_suffix_text'] = self.decode(suffix)
        self.trace.append(gen)
        if hit_max:
            self.limits += 1
        if suffix:
            self.committed += suffix
            self.emit += [end] * len(suffix)
            self.anchor += suffix
            self.emissions.append(dict(source_s=src_s, text=self.decode(suffix), token_ids=list(suffix)))
            new = self.decode(self.committed)
            c = 0
            while c < min(len(self.disp), len(new)) and self.disp[c] == new[c]:
                c += 1
            self.disp_t = self.disp_t[:c] + [src_s] * (len(new) - c)
            self.disp = new
        self.crop(end)

    def crop(self, end):
        """Slide the window to [end - W, end]; keep committed tokens emitted inside it plus the out-of-window context."""
        start = end - int(self.p.window * SR)
        if start <= self.ws:
            return
        times = self.emit[len(self.emit) - len(self.anchor):] if self.anchor else []
        k = next((i for i, t in enumerate(times) if t > start), len(self.anchor))
        kept = list(self.anchor[k:])
        if self.p.ctx_tokens > 0:
            e = len(self.committed) - len(kept)
            h = ctx_rule.history_start(self.committed, e, self.p.ctx_tokens, *self.p.ctx_sets)
            kept = list(self.committed[h:])
        self.ws = start
        self.anchor = kept

    # ---- output
    def text(self):
        return self.decode(self.committed)

    def result(self):
        chars = [(c, t) for c, t in zip(self.disp, self.disp_t) if not c.isspace()]
        it = self.item
        return dict(id=it['id'], direction=self.direction, source=it.get('transcription', ''),
                    reference=(it.get('translations') or {}).get(TARGET_LANG[self.direction], ''),
                    hypothesis=self.text(), prediction_for_alignment=''.join(c for c, _ in chars),
                    delays_ms=[round(t * 1000, 4) for _, t in chars], duration_s=self.n / SR,
                    processed_s=self.stopped_at if self.error else self.n / SR, complete=self.error is None, error=self.error,
                    stopped_at_s=self.stopped_at, wall_s=round(time.perf_counter() - self.started, 3), calls=self.calls,
                    generation_limit_count=self.limits, commits=len(self.emissions), invalid_candidates=self.invalid,
                    loop_guard=self.guard, loop_guards=self.loop_guards,
                    emissions=self.emissions, committed_token_ids=self.committed, beams=self.beams)
