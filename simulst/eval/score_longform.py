#!/usr/bin/env python3
"""Score long-form streaming results (whole recordings) against reference segments with OmniSTEval: the output is
re-segmented to the reference segments, then BLEU, chrF, COMET (wmt22-comet-da, source = source segment) and
LongLAAL (StreamLAAL, non-computation-aware) are computed. Chinese output is scored by character (SacreBLEU zh),
English output by word (13a; a word is emitted when its last character is).

OFFICIAL = directory with audio-segments.yaml (IWSLT format: wav, offset, duration per segment) and one
text file per language (en.txt, zh.txt) aligned with it; recording ids = wav names without extension.

  python -m simulst.eval.score_longform RESULT_DIR OFFICIAL [--comet-model Unbabel/wmt22-comet-da]
  -> RESULT_DIR/evaluation/{summary.json, scores/resegmented.jsonl}
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from omnisteval.data import get_segmentation_order, load_hypothesis_jsonl, load_reference
from omnisteval.resegment import resegment
from omnisteval.scoring import ChrFScorer, COMETScorer, LAALScorer, SacreBLEUScorer


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('result_dir')
    ap.add_argument('official')
    ap.add_argument('--comet-model', default='Unbabel/wmt22-comet-da')
    a = ap.parse_args()
    cfg, test = Path(a.result_dir).resolve(), Path(a.official).resolve()
    OUT = cfg / 'evaluation'
    INP = OUT / 'official_inputs'
    SCORES = OUT / 'scores'
    SCORES.mkdir(parents=True, exist_ok=True)
    INP.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(cfg.glob('results.*.jsonl')):
        rows += [json.loads(l) for l in p.read_text(encoding='utf8').splitlines() if l.strip()]
    byid = {r['id']: r for r in rows}
    assert len(rows) == len(byid), 'duplicate ids'
    dirs = {r['direction'] for r in rows}
    assert len(dirs) == 1, dirs
    zh_out = dirs.pop() == 'en2zh'
    src_lang, tgt_lang = ('en', 'zh') if zh_out else ('zh', 'en')
    for n in ('audio-segments.yaml', f'{src_lang}.txt', f'{tgt_lang}.txt'):
        (INP / n).write_text((test / n).read_text(encoding='utf8'), encoding='utf8')
    ref_words, segmentation, refs = load_reference(str(INP / 'audio-segments.yaml'), str(INP / f'{tgt_lang}.txt'),
                                                  char_level=zh_out, offset_delays=False)
    order = get_segmentation_order(segmentation)
    assert set(order) == set(byid), ('results do not match the reference recordings', sorted(set(order) ^ set(byid))[:5])
    sources = [line.strip() for line in (INP / f'{src_lang}.txt').read_text(encoding='utf8').splitlines()]
    assert len(sources) == len(refs) == len(segmentation)

    def hyp_of(r):
        hyp, delays = r['hypothesis'], r['delays_ms']
        pred = r['prediction_for_alignment']
        assert pred == ''.join(c for c in hyp if not c.isspace()) and len(pred) == len(delays)
        if zh_out:
            return pred, delays
        words, out, k = hyp.split(), [], 0
        for w in words:
            k += len(w)
            out.append(delays[k - 1])
        return ' '.join(words), out

    hyps = []
    for name in order:
        r = byid[name]
        pred, d = hyp_of(r)
        hyps.append(dict(source=name + '.wav', prediction=pred, delays=d, source_length=r['duration_s'] * 1000))
    (INP / 'hypotheses.jsonl').write_text(''.join(json.dumps(h, ensure_ascii=False) + '\n' for h in hyps), encoding='utf8')
    (INP / 'ids.txt').write_text('\n'.join(order) + '\n', encoding='utf8')
    hyp_words = load_hypothesis_jsonl(str(INP / 'hypotheses.jsonl'), char_level=zh_out, segmentation_order=order,
                                      fix_emission_ca_flag=False, emission_cu_key='delays', emission_ca_key='elapsed')
    instances, dictionaries = resegment(ref_words, hyp_words, segmentation, refs, char_level=zh_out, lang='zh' if zh_out else None)
    assert len(instances) == len(refs)
    (SCORES / 'resegmented.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in dictionaries), encoding='utf8')
    bleu = SacreBLEUScorer('zh' if zh_out else '13a')
    chrf = ChrFScorer()
    laal = LAALScorer(computation_aware=False)
    scorer = COMETScorer(a.comet_model)
    scorer._load_model()
    native_predict = scorer.model.predict
    captured = {}

    def predict_and_capture(*args, **kwargs):
        captured['output'] = native_predict(*args, **kwargs)
        return captured['output']

    scorer.model.predict = predict_and_capture
    global_comet = float(scorer(instances, sources))
    seg_comet = [float(v) for v in captured['output'].scores]
    assert abs(statistics.mean(seg_comet) - global_comet) < 1e-6
    per_recording = []
    for docid, name in enumerate(order):
        idx = [i for i, d in enumerate(dictionaries) if d['docid'] == docid]
        sub = [instances[i] for i in idx]
        r = byid[name]
        per_recording.append(dict(id=name, complete=r['complete'], error=r['error'], segments=len(idx), bleu=bleu(sub),
                                  comet=statistics.mean(seg_comet[i] for i in idx), long_laal_ms=laal(sub),
                                  empty_segments=sum(not s.prediction for s in sub)))
    summary = dict(direction='en2zh' if zh_out else 'zh2en', evaluated_recordings=len(order), completed_recordings=sum(r['complete'] for r in rows),
                   bleu=bleu(instances), chrf=chrf(instances), comet=global_comet, long_laal_ms=laal(instances), reference_segments=len(instances),
                   empty_segments=sum(not i.prediction for i in instances), comet_model=a.comet_model,
                   per_recording=per_recording, segment_comet=seg_comet, scored_at=time.time())
    save(OUT / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('direction', 'completed_recordings', 'bleu', 'comet', 'long_laal_ms', 'empty_segments')}), flush=True)


if __name__ == '__main__':
    main()
