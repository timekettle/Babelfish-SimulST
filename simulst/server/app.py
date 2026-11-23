#!/usr/bin/env python3
"""Live simultaneous translation server: web page, real-time WebSocket API and file upload (SSE).

  python -m simulst.server --model MODEL_DIR [--host 0.0.0.0] [--port 8080] [--beams 1] [--max-sessions 4]
  python -m simulst.server --backend llamacpp --llama-url http://127.0.0.1:8081 --tokenizer TOKENIZER_DIR

Endpoints (see docs/DEPLOYMENT.md#api):
  GET  /                  web page (microphone, file upload)
  GET  /health            model and load information
  WS   /translate/ws      real-time: JSON config, then binary PCM frames, then {"eof": true}
  POST /translate/stream  multipart upload (file, direction[, realtime]) -> server-sent events
  POST /translate         multipart upload -> one JSON result

Each session follows simulst.policy exactly like the offline replay (one model call per 320 ms block). A
real-time session that falls behind merges the pending blocks and decodes only the newest one, so the delay
never accumulates; file uploads decode every block, which reproduces the offline results.
"""
import argparse
import asyncio
import contextlib
import io
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from simulst import __version__
from simulst.common import (DEFAULT_BEAMS, DEFAULT_CTX_TOKENS, DEFAULT_MAX_NEW, DEFAULT_WINDOW_S, SR,
                            STEP_SAMPLES as STEP, normalize_direction)
from simulst.policy import Policy, Stream

STATIC = Path(__file__).with_name('static')


class Resampler:
    """Raw PCM bytes (s16le / f32le, any rate, interleaved channels) -> float32 mono 16 kHz, chunk by chunk."""

    def __init__(self, rate, channels=1, fmt='s16le'):
        if fmt not in ('s16le', 'f32le'):
            raise ValueError('format must be s16le or f32le')
        self.rate, self.ch, self.fmt = int(rate), int(channels), fmt
        self.width = (2 if fmt == 's16le' else 4) * self.ch
        self.rest = b''
        self.rs = None
        if self.rate != SR:
            import soxr
            self.rs = soxr.ResampleStream(self.rate, SR, 1, dtype='float32', quality='HQ')

    def __call__(self, data=b'', last=False):
        data = self.rest + data
        cut = len(data) - len(data) % self.width
        self.rest = data[cut:]
        data = data[:cut]
        if self.fmt == 's16le':
            x = np.frombuffer(data, '<i2').astype(np.float32) / 32768.0
        else:
            x = np.frombuffer(data, '<f4').astype(np.float32)
        if self.ch > 1:
            x = x.reshape(-1, self.ch).mean(1)
        if self.rs is not None:
            x = self.rs.resample_chunk(x, last=last)
        return x


def read_upload(raw):
    """Decode an uploaded audio file to float32 mono 16 kHz."""
    import soundfile as sf
    try:
        w, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    except Exception as e:
        raise ValueError(f'cannot decode the audio file ({e}); use wav / flac / ogg') from None
    w = w.mean(1)
    if sr != SR:
        import soxr
        w = soxr.resample(w, sr, SR, quality='HQ').astype(np.float32)
    return w


class State:
    def __init__(self, a):
        self.a = a
        self.engine = None
        self.policy = Policy(a.window, a.ctx_tokens, True, a.max_new, a.loop_guard)
        self.sessions = {}
        self.model_name = Path(a.model).name if a.model else 'llama.cpp'


class Session:
    def __init__(self, st, direction, beams, realtime):
        self.st, self.beams, self.realtime = st, beams, realtime
        self.sid = uuid.uuid4().hex[:12]
        self.stream = Stream(st.policy, st.engine.decode, uid=f'live|{self.sid}', direction=direction, beams=beams)
        self.k = 0               # last decoded block
        self.sent = ''           # text already sent to the client
        self.eof = False
        self.wake = asyncio.Event()
        self.events = asyncio.Queue()
        self.t0 = time.perf_counter()
        self.step_ms = []
        self.skipped = 0
        self.task = None

    def feed(self, samples):
        self.stream.append(samples)
        self.wake.set()

    def finish(self):
        self.eof = True
        self.wake.set()

    def put(self, ev):
        self.events.put_nowait(ev)

    def emit(self, k, final, ms):
        s = self.stream
        full = s.text()
        stable = full if final else full.rstrip('\ufffd')   # wait for the rest of a character split across tokens
        c = 0
        if stable.startswith(self.sent):
            c = len(self.sent)
        else:
            while c < min(len(stable), len(self.sent)) and stable[c] == self.sent[c]:
                c += 1
        delta = stable[c:]
        self.sent = stable
        if delta:
            self.put(dict(event='translation', delta=delta, text=stable, audio_s=round(s.end_of(k) / SR, 3),
                          elapsed_s=round(time.perf_counter() - self.t0, 3), step_ms=ms))

    async def step(self, k, final):
        s = self.stream
        req = s.request(k, final)
        if req is None:
            return
        if req == 'limit':
            raise RuntimeError(s.error)
        t = time.perf_counter()
        suffix, ended = await self.st.engine.run(req, self.beams, self.st.a.length_penalty)
        ms = round((time.perf_counter() - t) * 1000, 1)
        self.step_ms.append(ms)
        s.commit(suffix, ended, k, self.st.engine.special)
        self.emit(k, final, ms)

    async def run(self):
        s = self.stream
        try:
            while True:
                await self.wake.wait()
                self.wake.clear()
                while True:
                    avail = s.blocks if self.eof else s.full_blocks
                    if avail <= self.k:
                        break
                    k = avail if self.realtime else self.k + 1
                    self.skipped += k - self.k - 1
                    await self.step(k, final=False)
                    self.k = k
                if self.eof and self.k >= s.blocks:
                    await self.step(s.blocks, final=True)
                    ms = sorted(self.step_ms)
                    self.put(dict(event='done', text=s.text(), audio_s=round(s.n / SR, 3),
                                  elapsed_s=round(time.perf_counter() - self.t0, 3), steps=s.calls,
                                  skipped_blocks=self.skipped, loop_guards=s.loop_guards,
                                  step_ms_p50=ms[len(ms) // 2] if ms else None,
                                  step_ms_p90=ms[int(len(ms) * 0.9)] if ms else None,
                                  emissions=[dict(audio_s=e['source_s'], text=e['text']) for e in s.emissions]))
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:   # report to the client instead of dying silently
            self.put(dict(event='error', message=f'{type(e).__name__}: {e}'))
        finally:
            self.put(None)


def build_app(a):
    from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    st = State(a)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if a.backend == 'llamacpp':
            from simulst.engines.llamacpp import AsyncLlamaCppEngine
            st.engine = AsyncLlamaCppEngine(a.llama_url, tokenizer_dir=a.tokenizer or a.model, max_new=a.max_new)
        else:
            from simulst.engines.vllm_engine import AsyncVLLMEngine
            st.engine = AsyncVLLMEngine(a.model, dtype=a.dtype, gpu_mem=a.gpu_mem, max_seqs=a.max_seqs, eager=a.eager,
                                        tp=a.tp, max_new=a.max_new, max_model_len=a.max_model_len)
        st.policy.bind_tokenizer(st.engine.decode, st.engine.n_regular)
        rng = np.random.default_rng(0)   # warm-up: compile kernels / capture graphs before the first user
        for d in ('en2zh', 'zh2en'):
            w = Session(st, d, a.beams, realtime=False)
            w.feed((rng.standard_normal(3 * STEP) * 0.01).astype(np.float32))
            w.finish()
            await w.run()
        print(f'Ready: http://{"127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host}:{a.port}/', flush=True)
        yield
        st.engine.shutdown()

    app = FastAPI(title='Babelfish', version=__version__, lifespan=lifespan)
    app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')

    def admit(direction, beams, realtime):
        if len(st.sessions) >= a.max_sessions:
            raise RuntimeError(f'server busy: {a.max_sessions} sessions already running')
        beams = a.beams if beams is None else int(beams)
        if not 1 <= beams <= 4:
            raise ValueError('beams must be between 1 and 4')
        s = Session(st, normalize_direction(direction), beams, realtime)
        st.sessions[s.sid] = s
        s.task = asyncio.create_task(s.run())
        return s

    def close(s):
        st.sessions.pop(s.sid, None)
        if s.task and not s.task.done():
            s.task.cancel()

    @app.get('/')
    async def index():
        return FileResponse(STATIC / 'index.html')

    @app.get('/health')
    async def health():
        return dict(status='ok' if st.engine else 'loading', version=__version__, model=st.model_name, backend=a.backend,
                    sessions=len(st.sessions), max_sessions=a.max_sessions, beams=a.beams,
                    loop_guard=a.loop_guard,
                    window_s=a.window, ctx_tokens=a.ctx_tokens, directions=['en2zh', 'zh2en'], sample_rate=SR)

    @app.websocket('/translate/ws')
    async def ws_translate(ws: WebSocket):
        await ws.accept()
        s = None
        try:
            cfg = json.loads(await ws.receive_text())
            rs = Resampler(cfg.get('sample_rate', SR), cfg.get('channels', 1), cfg.get('format', 's16le'))
            s = admit(cfg.get('direction', 'en2zh'), cfg.get('beams'), realtime=bool(cfg.get('realtime', True)))
        except Exception as e:
            await ws.send_json(dict(event='error', message=str(e)))
            await ws.close()
            return
        await ws.send_json(dict(event='ready', session=s.sid, direction=s.stream.direction, beams=s.beams,
                                model=st.model_name, sample_rate=SR))

        async def sender():
            while True:
                ev = await s.events.get()
                if ev is None:
                    return
                await ws.send_json(ev)

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await ws.receive()
                if msg['type'] == 'websocket.disconnect':
                    break
                if msg.get('bytes'):
                    s.feed(rs(msg['bytes']))
                elif msg.get('text'):
                    d = json.loads(msg['text'])
                    if d.get('eof') or d.get('type') == 'eof':
                        s.feed(rs(last=True))
                        s.finish()
                        await send_task
                        break
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            close(s)
            with contextlib.suppress(Exception):
                await ws.close()

    async def file_session(raw, direction, realtime, beams):
        wave = read_upload(raw)
        s = admit(direction, beams, realtime)

        async def feeder():
            if realtime:   # pace the file like a live speaker
                t0 = time.perf_counter()
                for i in range(0, len(wave), STEP):
                    await asyncio.sleep(max(0.0, t0 + i / SR - time.perf_counter()))
                    s.feed(wave[i:i + STEP])
            else:
                s.feed(wave)
            s.finish()
        return s, asyncio.create_task(feeder())

    @app.post('/translate/stream')
    async def sse_translate(file: UploadFile = File(...), direction: str = Form('en2zh'), realtime: bool = Form(False),
                            beams: int | None = Form(None)):
        try:
            s, feed_task = await file_session(await file.read(), direction, realtime, beams)
        except Exception as e:
            return JSONResponse(dict(event='error', message=str(e)), status_code=400)

        async def gen():
            try:
                yield f'data: {json.dumps(dict(event="ready", session=s.sid, direction=s.stream.direction, beams=s.beams), ensure_ascii=False)}\n\n'
                while True:
                    ev = await s.events.get()
                    if ev is None:
                        break
                    yield f'data: {json.dumps(ev, ensure_ascii=False)}\n\n'
            finally:
                feed_task.cancel()
                close(s)
        return StreamingResponse(gen(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})

    @app.post('/translate')
    async def translate(file: UploadFile = File(...), direction: str = Form('en2zh'), beams: int | None = Form(None)):
        try:
            s, feed_task = await file_session(await file.read(), direction, False, beams)
        except Exception as e:
            return JSONResponse(dict(event='error', message=str(e)), status_code=400)
        try:
            last = None
            while True:
                ev = await s.events.get()
                if ev is None:
                    break
                last = ev
        finally:
            feed_task.cancel()
            close(s)
        return JSONResponse(last or dict(event='error', message='no result'), status_code=200 if last and last['event'] == 'done' else 500)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='', help='model directory or Hugging Face id (vLLM backend)')
    ap.add_argument('--backend', choices=['vllm', 'llamacpp'], default='vllm')
    ap.add_argument('--llama-url', default='http://127.0.0.1:8081', help='llama-server address (llamacpp backend)')
    ap.add_argument('--tokenizer', default='', help='tokenizer directory (llamacpp backend; defaults to --model)')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--beams', type=int, default=DEFAULT_BEAMS, help='default beam size, 1 = greedy (default)')
    ap.add_argument('--length-penalty', type=float, default=1.0)
    ap.add_argument('--window', type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument('--ctx-tokens', type=int, default=DEFAULT_CTX_TOKENS)
    ap.add_argument('--max-new', type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument('--loop-guard', choices=['on', 'off'], default='on', help='repetition guard (simulst.loop_guard)')
    ap.add_argument('--max-sessions', type=int, default=4, help='concurrent sessions')
    ap.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16', 'auto'])
    ap.add_argument('--tp', '--tensor-parallel-size', dest='tp', type=int, default=1)
    ap.add_argument('--gpu-mem', '--gpu-memory-utilization', dest='gpu_mem', type=float, default=0.9)
    ap.add_argument('--max-seqs', type=int, default=64)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--eager', action='store_true', help='disable CUDA graphs (slower, saves memory)')
    ap.add_argument('--ssl-certfile', default=None, help='serve HTTPS (browsers only allow the microphone on https or localhost)')
    ap.add_argument('--ssl-keyfile', default=None)
    a = ap.parse_args()
    if a.backend == 'vllm' and not a.model:
        ap.error('--model is required')
    import uvicorn
    uvicorn.run(build_app(a), host=a.host, port=a.port, ws_max_size=16 * 2 ** 20, log_level='warning',
                ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile)


if __name__ == '__main__':
    main()
