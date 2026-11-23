#!/usr/bin/env python3
"""Build evaluation manifests (one JSON per line: id, audio_path, duration, transcription, translations, direction).

  fleurs   --fleurs-en DIR --fleurs-zh DIR : FLEURS test, both directions. DIR = a FLEURS language directory with
           test.tsv and the extracted test audio (audio/test/*.wav), e.g. from the google/fleurs dataset (en_us,
           cmn_hans_cn). The reference of a sentence is the raw transcription of the same sentence id in the other
           language.
  covost   --covost-tsv DIR --cv-root DIR : CoVoST 2 test sets, both directions. --covost-tsv = the split TSVs made by
           the official CoVoST 2 script (covost_v2.en_zh-CN.test.tsv, covost_v2.zh-CN_en.test.tsv); --cv-root = the
           Common Voice 4 release (en/clips/*.mp3, zh-CN/clips/*.mp3). Clips are converted to 16 kHz mono WAV under
           OUT/covost_wav (torchaudio's sinc resampler).
  realsi   --realsi DIR : RealSI long-form recordings (DIR/<en2zh|zh2en>/{json,wav}). Writes the manifest and
           OUT/realsi_<dir>/official_inputs/{audio-segments.yaml, en.txt, zh.txt, ids.txt} for long-form scoring
           (simulst.eval.score_longform), plus a sentence-level version cut at the reference segments.
"""
import argparse
import csv
import json
from pathlib import Path

import soundfile as sf

SR = 16000


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf8')
    print(json.dumps(dict(out=str(path), n=len(rows), hours=round(sum(r['duration'] for r in rows) / 3600, 2))), flush=True)


def fleurs_rows(d):
    rows = []
    with open(Path(d) / 'test.tsv', encoding='utf8') as f:
        for row in csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            rows.append(dict(id=int(row[0]), file=row[1], raw=row[2].strip(), num_samples=int(row[5])))
    return rows


def build_fleurs(a, out):
    for src_dir, tgt_dir, src, tgt, d in ((a.fleurs_en, a.fleurs_zh, 'en', 'zh', 'en2zh'), (a.fleurs_zh, a.fleurs_en, 'zh', 'en', 'zh2en')):
        refs = {}
        for r in fleurs_rows(tgt_dir):
            refs.setdefault(r['id'], r['raw'])
        rows = []
        for r in fleurs_rows(src_dir):
            wav = Path(src_dir) / 'audio' / 'test' / r['file']
            if not wav.exists() or not refs.get(r['id']):
                continue
            rows.append(dict(id=f'fleurs_{src}_{r["file"][:-4]}', sentence_id=r['id'], audio_path=str(wav),
                             duration=round(r['num_samples'] / SR, 3), transcription=r['raw'], translations={tgt: refs[r['id']]},
                             direction=d))
        write(out / f'fleurs_{d}.jsonl', rows)


def convert_clip(job):
    """mp3 -> 16 kHz mono 16-bit WAV (first channel). Returns the duration in seconds."""
    src, dst = job
    dst = Path(dst)
    if not dst.is_file():
        import torch
        import torchaudio.functional as AF
        wav, sr = sf.read(src, dtype='float32', always_2d=True)
        wav = torch.from_numpy(wav[:, 0].copy()).unsqueeze(0)
        if sr != SR:
            wav = AF.resample(wav, sr, SR)
        sf.write(str(dst), wav[0].numpy(), SR, subtype='PCM_16')
    return sf.info(str(dst)).frames / SR


def build_covost(a, out):
    from concurrent.futures import ProcessPoolExecutor
    for pair, lang, tgt, d in (('en_zh-CN', 'en', 'zh', 'en2zh'), ('zh-CN_en', 'zh-CN', 'en', 'zh2en')):
        tsv = Path(a.covost_tsv) / f'covost_v2.{pair}.test.tsv'
        with open(tsv, encoding='utf8') as f:   # text kept exactly as in the TSV
            rows = list(csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE))
        wav_dir = out / 'covost_wav' / lang
        wav_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(str(Path(a.cv_root) / lang / 'clips' / r['path']), str(wav_dir / (Path(r['path']).stem + '.wav'))) for r in rows]
        with ProcessPoolExecutor(a.workers) as ex:
            durs = list(ex.map(convert_clip, jobs, chunksize=16))
        items = [dict(id=Path(r['path']).stem, audio_path=dst, duration=round(dur, 3), transcription=r['sentence'],
                      translations={tgt: r['translation']}, direction=d) for r, (_, dst), dur in zip(rows, jobs, durs)]
        write(out / f'covost_{d}.jsonl', items)


def build_realsi(a, out):
    import yaml
    root = Path(a.realsi)
    for d, (src, tgt) in dict(en2zh=('en', 'zh'), zh2en=('zh', 'en')).items():
        items, segs, srcs, refs = [], [], [], []
        for jp in sorted((root / d / 'json').glob('*.json')):
            j = json.loads(jp.read_text(encoding='utf8'))
            vid = j['vid']
            wav = root / d / 'wav' / f'{vid}.wav'
            info = sf.info(str(wav))
            assert info.samplerate == SR, (wav, info.samplerate)
            dur = info.frames / info.samplerate
            ss = [s for s in j['segment'] if s['src_text'].strip() and s['trg_text'].strip()]
            for s in ss:
                st, en = s['start_time'] / 1000, s['end_time'] / 1000
                assert 0 <= st < en, (vid, s)
                segs.append(dict(duration=round(min(en, dur) - st, 3), offset=round(st, 3), speaker_id='1', wav=f'{vid}.wav'))
                srcs.append(' '.join(s['src_text'].split()))
                refs.append(' '.join(s['trg_text'].split()))
            items.append(dict(id=vid, audio_path=str(wav), duration=round(dur, 3), transcription=' '.join(s['src_text'] for s in ss),
                              translations={tgt: ' '.join(s['trg_text'] for s in ss)}, direction=d, segments=len(ss)))
        name = f'realsi_{d}'
        write(out / f'{name}.jsonl', items)
        oi = out / name / 'official_inputs'
        oi.mkdir(parents=True, exist_ok=True)
        (oi / 'audio-segments.yaml').write_text(yaml.safe_dump(segs, allow_unicode=True, default_flow_style=None, width=1000), encoding='utf8')
        (oi / f'{src}.txt').write_text('\n'.join(srcs) + '\n', encoding='utf8')
        (oi / f'{tgt}.txt').write_text('\n'.join(refs) + '\n', encoding='utf8')
        (oi / 'ids.txt').write_text('\n'.join(x['id'] for x in items) + '\n', encoding='utf8')
        # sentence-level version: every reference segment as its own item
        wav_dir = out / f'realsi_sent_{d}_wav'
        wav_dir.mkdir(parents=True, exist_ok=True)
        cache, sent, k_of = {}, [], {}
        paths = {x['id']: x['audio_path'] for x in items}
        for s, x, y in zip(segs, srcs, refs):
            rid = s['wav'].rsplit('.', 1)[0]
            if rid not in cache:
                w, sr = sf.read(paths[rid], dtype='float32', always_2d=True)
                cache = {rid: w.mean(1)}
            k = k_of[rid] = k_of.get(rid, -1) + 1
            b0, b1 = int(round(s['offset'] * SR)), int(round((s['offset'] + s['duration']) * SR))
            p = wav_dir / f'{rid}_{k:03d}.wav'
            if not p.exists():
                sf.write(p, cache[rid][b0:b1], SR, subtype='PCM_16')
            sent.append(dict(id=f'{rid}_{k:03d}', audio_path=str(p), duration=round((b1 - b0) / SR, 3), transcription=x,
                             translations={tgt: y}, direction=d))
        write(out / f'realsi_sent_{d}.jsonl', sent)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['fleurs', 'covost', 'realsi'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--fleurs-en')
    ap.add_argument('--fleurs-zh')
    ap.add_argument('--covost-tsv')
    ap.add_argument('--cv-root')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--realsi')
    a = ap.parse_args()
    out = Path(a.out)
    dict(fleurs=build_fleurs, covost=build_covost, realsi=build_realsi)[a.what](a, out)


if __name__ == '__main__':
    main()
