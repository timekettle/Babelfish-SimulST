#!/usr/bin/env python3
"""Command-line client for the server: stream an audio file and print the translation as it arrives.

  python -m simulst.client audio.wav --direction en2zh [--url http://127.0.0.1:8080] [--realtime] [--save out.json]
  python -m simulst.client audio.wav --direction zh2en --transport sse

--transport ws (default) sends 16 kHz PCM (float32 by default) over the WebSocket API, 100 ms per message; with --realtime it waits
between messages like a live speaker (the server then skips blocks it cannot keep up with), otherwise it sends as
fast as possible and every block is decoded. --transport sse uploads the file to /translate/stream.
Needs: numpy, soundfile, soxr, websockets, httpx.
"""
import argparse
import asyncio
import json
import sys
import time

import numpy as np

SR = 16000


def load(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype='float32', always_2d=True)
    w = w.mean(1)
    if sr != SR:
        import soxr
        w = soxr.resample(w, sr, SR, quality='HQ').astype(np.float32)
    return w


def show(ev, state):
    if ev['event'] == 'translation':
        sys.stdout.write(ev['text'][len(state['printed']):] if ev['text'].startswith(state['printed']) else '\n' + ev['text'])
        sys.stdout.flush()
        state['printed'] = ev['text']
    elif ev['event'] == 'done':
        rest = ev['text'][len(state['printed']):] if ev['text'].startswith(state['printed']) else '\n' + ev['text']
        print(rest)
        p50, p90 = ev.get('step_ms_p50'), ev.get('step_ms_p90')
        step = f', step p50 {p50:.0f} ms / p90 {p90:.0f} ms' if p50 is not None else ''
        guard = f', repetition guard {ev["loop_guards"]}x' if ev.get('loop_guards') else ''
        print(f'[done] audio {ev["audio_s"]:.1f} s, wall {ev["elapsed_s"]:.1f} s{step}, skipped blocks {ev.get("skipped_blocks", 0)}{guard}',
              file=sys.stderr)
    elif ev['event'] == 'error':
        print(f'\n[error] {ev["message"]}', file=sys.stderr)


async def run_ws(a, wave, events):
    import websockets
    url = a.url.rstrip('/').replace('https://', 'wss://').replace('http://', 'ws://') + '/translate/ws'
    state = dict(printed='')
    async with websockets.connect(url, max_size=16 * 2 ** 20) as ws:
        cfg = dict(direction=a.direction, sample_rate=SR, format=a.format, channels=1, realtime=a.realtime)
        if a.beams:
            cfg['beams'] = a.beams
        await ws.send(json.dumps(cfg))
        ready = json.loads(await ws.recv())
        events.append(ready)
        if ready['event'] != 'ready':
            show(ready, state)
            return

        async def send():
            if a.format == 's16le':
                pcm = np.clip(np.round(wave * 32768), -32768, 32767).astype('<i2')
            else:
                pcm = wave.astype('<f4')
            hop = SR // 10
            t0 = time.perf_counter()
            for i in range(0, len(pcm), hop):
                if a.realtime:
                    await asyncio.sleep(max(0.0, t0 + i / SR - time.perf_counter()))
                await ws.send(pcm[i:i + hop].tobytes())
            await ws.send(json.dumps(dict(eof=True)))

        sender = asyncio.create_task(send())
        async for msg in ws:
            ev = json.loads(msg)
            ev['client_s'] = round(time.perf_counter() - a.t0, 3)
            events.append(ev)
            show(ev, state)
            if ev['event'] in ('done', 'error'):
                break
        sender.cancel()


async def run_sse(a, events):
    import httpx
    state = dict(printed='')
    data = dict(direction=a.direction, realtime=str(a.realtime).lower())
    if a.beams:
        data['beams'] = str(a.beams)
    with open(a.audio, 'rb') as f:
        files = dict(file=(a.audio, f.read()))
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream('POST', a.url.rstrip('/') + '/translate/stream', data=data, files=files) as r:
            if r.status_code != 200:
                show(json.loads(await r.aread()), state)
                return
            async for line in r.aiter_lines():
                if line.startswith('data: '):
                    ev = json.loads(line[6:])
                    ev['client_s'] = round(time.perf_counter() - a.t0, 3)
                    events.append(ev)
                    if ev['event'] != 'ready':
                        show(ev, state)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('audio')
    ap.add_argument('--direction', default='en2zh', help='en2zh or zh2en')
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--transport', choices=['ws', 'sse'], default='ws')
    ap.add_argument('--realtime', action='store_true', help='send audio at real-time speed')
    ap.add_argument('--beams', type=int, default=0, help='override the server default beam size')
    ap.add_argument('--format', choices=['f32le', 's16le'], default='f32le', help='PCM format sent over the WebSocket')
    ap.add_argument('--save', default='', help='write every event (JSON lines) to this file')
    a = ap.parse_args()
    a.t0 = time.perf_counter()
    events = []
    if a.transport == 'ws':
        asyncio.run(run_ws(a, load(a.audio), events))
    else:
        asyncio.run(run_sse(a, events))
    if a.save:
        with open(a.save, 'w', encoding='utf8') as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
