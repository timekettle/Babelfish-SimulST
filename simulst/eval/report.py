#!/usr/bin/env python3
"""Summary table of an evaluation directory (one sub-directory per test set, scored by score_sentences or
score_longform) -> OUT/REPORT.md."""
import json
import sys
from pathlib import Path


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def main():
    out = Path(sys.argv[1])
    lines = ['| Test set | Direction | Items | COMET | BLEU | Latency (ms) | Length ratio |', '|---|---|---|---|---|---|---|']
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        s = load(d / 'evaluation/summary.json')
        if s:
            lines.append(f"| {d.name} | {s['direction']} | {s['evaluated_recordings']} | {s['comet']:.4f} | {s['bleu']:.2f} | "
                         f"{s['long_laal_ms']:.0f} (LongLAAL) | |")
            continue
        s = load(d / 'summary.json')
        if s:
            lines.append(f"| {d.name} | {s['direction']} | {s['n']} | {s['comet']:.4f} | {s['bleu']:.2f} | {s['laal_ms']:.0f} (LAAL) | "
                         f"{s['len_ratio']:.3f} |")
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
