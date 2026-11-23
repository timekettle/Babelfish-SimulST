#!/usr/bin/env python3
"""Offline streaming replay: decode audio files as if they were streamed, 320 ms at a time (simulst.policy).

Many files are decoded together and batched. Latency is the amount of audio received when a character is
committed (computation time is not counted). This is the script behind the reported results.

Manifest rows (JSONL): id, audio_path (16 kHz), direction (en2zh / zh2en), optional transcription and
translations {zh|en: reference}. Results: OUT/<name>/results.<shard>.jsonl (+ per-item traces), one row per input:
hypothesis, per-character delays_ms (non-space characters), emissions, committed token ids, ...

  python -m simulst.infer.stream --model MODEL_DIR --manifests fleurs=fleurs_en2zh.jsonl --out OUT \\
      [--window 24 --ctx-tokens 256 --beams 1] [--shard i --shards n]
  python -m simulst.infer.stream --backend llamacpp --llama-url http://127.0.0.1:8080 --tokenizer TOKENIZER_DIR ...
"""
import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from simulst.common import (DEFAULT_BEAMS, DEFAULT_CTX_TOKENS, DEFAULT_MAX_NEW, DEFAULT_WINDOW_S, SR,
                            STEP_SAMPLES as STEP, TARGET_LANG, drop_dist_env)
from simulst.policy import Policy, Stream


def load_item(item, policy, decode, uid, beams):
    w, sr = sf.read(item['audio_path'], dtype='float32', always_2d=True)
    assert sr == SR, (item['audio_path'], sr)
    tr = item.get('translations') or {}
    direction = item.get('direction') or ('en2zh' if 'zh' in tr else 'zh2en')
    assert direction in TARGET_LANG, (item['id'], direction)
    wave = w.mean(1) if w.shape[1] > 1 else w[:, 0]
    return Stream(policy, decode, uid, direction, wave=wave, item=item, beams=beams)


def build_engine(a):
    if a.backend == 'llamacpp':
        from simulst.engines.llamacpp import LlamaCppEngine
        return LlamaCppEngine(a.llama_url, tokenizer_dir=a.tokenizer or a.model, max_new=a.max_new)
    from simulst.engines.vllm_engine import VLLMEngine
    return VLLMEngine(a.model, dtype=a.dtype, gpu_mem=a.gpu_mem, max_seqs=a.max_seqs, eager=a.eager, tp=a.tp, max_new=a.max_new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', help='model directory or Hugging Face id (vLLM backend)')
    ap.add_argument('--manifests', required=True, help='name=path[,name=path...]; all items are decoded together')
    ap.add_argument('--out', required=True, help='results go to OUT/<name>/results.<shard>.jsonl')
    ap.add_argument('--window', type=float, default=DEFAULT_WINDOW_S, help='audio window in seconds')
    ap.add_argument('--ctx-tokens', type=int, default=DEFAULT_CTX_TOKENS, help='committed tokens kept as context before the window')
    ap.add_argument('--no-end-marker', dest='end_marker', action='store_false', help='skip the final call with the end marker')
    ap.add_argument('--beams', type=int, default=DEFAULT_BEAMS, help='beam size, 1 = greedy (default)')
    ap.add_argument('--length-penalty', type=float, default=1.0)
    ap.add_argument('--max-new', type=int, default=DEFAULT_MAX_NEW, help='max new tokens per step')
    ap.add_argument('--loop-guard', choices=['on', 'off'], default='on', help='repetition guard (simulst.loop_guard)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--ids', default='', help='comma-separated item ids to decode (default all)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--backend', choices=['vllm', 'llamacpp'], default='vllm')
    ap.add_argument('--llama-url', default='http://127.0.0.1:8080', help='llama-server address (llamacpp backend)')
    ap.add_argument('--tokenizer', default='', help='tokenizer directory (llamacpp backend; defaults to --model)')
    ap.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16', 'auto'])
    ap.add_argument('--tp', type=int, default=1, help='tensor parallel size')
    ap.add_argument('--gpu-mem', type=float, default=0.9)
    ap.add_argument('--max-seqs', type=int, default=256)
    ap.add_argument('--max-live', type=int, default=48, help='items decoded concurrently (larger values evict the prefix cache)')
    ap.add_argument('--eager', action='store_true', help='disable CUDA graphs')
    ap.add_argument('--no-traces', dest='traces', action='store_false')
    a = ap.parse_args()
    if a.backend == 'vllm' and not a.model:
        ap.error('--model is required')
    drop_dist_env()
    root = Path(a.out)
    mans = [x.split('=', 1) for x in a.manifests.split(',') if x]
    want = set(a.ids.split(',')) if a.ids else None
    todo = []
    for name, path in mans:
        items = [json.loads(l) for l in open(path, encoding='utf8') if l.strip()]
        if want:
            items = [x for x in items if x['id'] in want]
        items = items[a.shard::a.shards]
        if a.limit:
            items = items[:a.limit]
        rep = root / name
        (rep / 'traces').mkdir(parents=True, exist_ok=True)
        res_p = rep / f'results.{a.shard}.jsonl'
        done = {json.loads(l)['id'] for l in open(res_p, encoding='utf8')} if res_p.exists() else set()
        todo += [(name, x) for x in items if x['id'] not in done]
    eng = build_engine(a)
    policy = Policy(a.window, a.ctx_tokens, a.end_marker, a.max_new, a.loop_guard)
    policy.bind_tokenizer(eng.decode, eng.n_regular)
    t0 = time.perf_counter()
    streams = []
    for name, x in todo:
        s = load_item(x, policy, eng.decode, uid=f"{name}|{x['id']}|r", beams=a.beams)
        s.man = name
        streams.append(s)
    # decode at most --max-live items at a time (longest first); each advances block by block
    pending = sorted(streams, key=lambda s: -s.n)
    max_live = a.max_live if a.max_live > 0 else len(pending)
    live = []
    it = nb = 0
    while live or pending:
        while pending and len(live) < max_live:
            s = pending.pop(0)
            s.k = 1
            live.append(s)
        it += 1
        for phase in ('normal', 'final'):
            batch, meta = [], []
            for s in live:
                k = s.k
                if s.error or k > s.blocks:
                    continue
                final = s.end_of(k) >= s.n
                if phase == 'final' and not final:
                    continue
                r = s.request(k, final=(phase == 'final'))
                if r is None:
                    continue
                if r == 'limit':
                    s.trace.append(dict(event='evaluation_error', error=s.error, source_s=s.end_of(k) / SR))
                    continue
                batch.append(r)
                meta.append(s)
            if batch:
                nb += 1
                outs = eng.run(batch, a.beams, a.length_penalty)
                for s, (suffix, ended) in zip(meta, outs):
                    s.commit(suffix, ended, s.k, eng.special)
        fin = [s for s in live if s.error or s.k >= s.blocks]
        for s in fin:
            rep = root / s.man
            with open(rep / f'results.{a.shard}.jsonl', 'a', encoding='utf8') as f:
                f.write(json.dumps(s.result(), ensure_ascii=False) + '\n')
            if a.traces:
                with open(rep / 'traces' / f"{s.item['id']}.jsonl", 'w', encoding='utf8') as tf:
                    for row in s.trace:
                        tf.write(json.dumps(row, ensure_ascii=False) + '\n')
            s.wave = None
        if fin:
            live = [s for s in live if not (s.error or s.k >= s.blocks)]
        for s in live:
            s.k += 1
        if it % 20 == 0:
            print(json.dumps(dict(event='progress', shard=a.shard, iter=it, live=len(live), pending=len(pending),
                                  done=len(streams) - len(live) - len(pending), wall_s=round(time.perf_counter() - t0, 1))), flush=True)
    print(json.dumps(dict(event='done', shard=a.shard, items=len(streams), wall_s=round(time.perf_counter() - t0, 1), batches=nb)), flush=True)


if __name__ == '__main__':
    main()
