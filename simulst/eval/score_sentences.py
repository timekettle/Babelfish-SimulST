#!/usr/bin/env python3
"""Score sentence-level streaming results (one item = one sentence): BLEU (sacrebleu; zh tokenizer for Chinese
output, 13a for English), COMET (wmt22-comet-da, with source and reference), and per-sentence LAAL / AL in ms
(non-computation-aware; characters for Chinese output, words for English output, a word being emitted when its
last character is).

  LAAL = mean_{i <= tau} [ d_i - (i - 1) * |X| / max(|Y|, |Y*|) ],  tau = first unit with d_i >= |X| (else all);
  AL uses |Y*| (reference length) in the denominator.

  python -m simulst.eval.score_sentences RESULT_DIR [...]   (RESULT_DIR/results.*.jsonl) -> summary.json, sentences.jsonl
"""
import argparse
import json
from pathlib import Path

import sacrebleu


def lagging(delays, src_ms, hyp_len, ref_len, use_ref_only=False):
    if not delays:
        return None
    denom = ref_len if use_ref_only else max(hyp_len, ref_len)
    rate = src_ms / max(denom, 1)
    tau = next((i + 1 for i, d in enumerate(delays) if d >= src_ms), len(delays))
    return sum(delays[i] - i * rate for i in range(tau)) / tau


def word_delays(r):
    hyp, delays = r['hypothesis'], r['delays_ms']
    assert r['prediction_for_alignment'] == ''.join(c for c in hyp if not c.isspace())
    words, out, k = hyp.split(), [], 0
    for w in words:
        k += len(w)
        out.append(delays[k - 1])
    assert k == len(delays)
    return words, out


def load_comet(name):
    from comet import download_model, load_from_checkpoint
    return load_from_checkpoint(name if name.endswith('.ckpt') else download_model(name))


def load_results(d):
    rows = []
    for p in sorted(Path(d).glob('results.*.jsonl')):
        rows += [json.loads(l) for l in p.read_text(encoding='utf8').splitlines() if l.strip()]
    rows.sort(key=lambda r: r['id'])
    assert len({r['id'] for r in rows}) == len(rows), f'duplicate ids in {d}'
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--comet-model', default='Unbabel/wmt22-comet-da', help='model name or path to a .ckpt')
    ap.add_argument('--batch-size', type=int, default=64)
    a = ap.parse_args()
    model = load_comet(a.comet_model)
    for d in [Path(x).resolve() for x in a.dirs]:
        rows = load_results(d)
        zh = rows[0]['direction'] == 'en2zh'
        bad = [r['id'] for r in rows if not r['complete'] or r['error']]
        hyps = [r['hypothesis'].strip() if zh else ' '.join(r['hypothesis'].split()) for r in rows]
        refs = [r['reference'] for r in rows]
        srcs = [r['source'] for r in rows]
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize='zh' if zh else '13a')
        out = model.predict([dict(src=s, mt=h, ref=r) for s, h, r in zip(srcs, hyps, refs)], batch_size=a.batch_size, gpus=1,
                            progress_bar=False)
        sent = []
        for r, c in zip(rows, out.scores):
            src_ms = r['duration_s'] * 1000
            if zh:
                ref_len = len([ch for ch in r['reference'] if not ch.isspace()])
                dl = r['delays_ms']
            else:
                ref_len = len(r['reference'].split())
                _, dl = word_delays(r)
            sent.append(dict(id=r['id'], comet=c, laal=lagging(dl, src_ms, len(dl), ref_len),
                             al=lagging(dl, src_ms, len(dl), ref_len, use_ref_only=True),
                             first_ms=dl[0] if dl else None, src_ms=src_ms, hyp_units=len(dl), ref_units=ref_len, wall_s=r['wall_s']))
        lat = [s for s in sent if s['laal'] is not None]
        summ = dict(dir=d.name, direction=rows[0]['direction'], unit='char' if zh else 'word', n=len(rows), incomplete=len(bad),
                    empty=sum(1 for h in hyps if not h), bleu=round(bleu.score, 2), comet=round(sum(out.scores) / len(out.scores), 4),
                    laal_ms=round(sum(s['laal'] for s in lat) / len(lat), 1), al_ms=round(sum(s['al'] for s in lat) / len(lat), 1),
                    first_unit_ms=round(sum(s['first_ms'] for s in lat) / len(lat), 1),
                    len_ratio=round(sum(s['hyp_units'] for s in sent) / sum(s['ref_units'] for s in sent), 3),
                    mean_src_s=round(sum(s['src_ms'] for s in sent) / len(sent) / 1000, 2))
        (d / 'sentences.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in sent), encoding='utf8')
        (d / 'summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))
        print(json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
