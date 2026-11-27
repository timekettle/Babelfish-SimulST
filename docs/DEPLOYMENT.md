**English** | [简体中文](DEPLOYMENT_zh.md) | [Back to README](../README.md)

# Deployment guide

How to deploy the Babelfish inference service: choosing a model version, installing, starting the server, using the web page and the APIs, the FP8 and GGUF versions, and common problems.

- [1. Versions and hardware](#1-versions-and-hardware)
- [2. Installation](#2-installation)
- [3. Starting the server](#3-starting-the-server)
- [4. Web page](#4-web-page)
- [5. API](#api)
- [6. FP8 version](#6-fp8-version)
- [7. llama.cpp (GGUF version)](#7-llamacpp-gguf-version)
- [8. Offline batch inference](#8-offline-batch-inference)
- [9. FAQ](#9-faq)

## 1. Versions and hardware

| Version | Weights | GPU memory | Tested | Framework | Default decoding |
|---|---|---|---|---|---|
| Babelfish-30B-A3B (FP16) | 58 GiB (13 shards) | 80 GB × 1, or 48 GB × 2 with tensor parallelism | A800 80GB; two GPUs with tensor parallelism limited to 46 GB each | vLLM | greedy |
| Babelfish-30B-A3B-FP8 | 30 GiB (7 shards) | 48 GB × 1 | A800 (weight-only Marlin), limited to 44 GB | vLLM | greedy |
| Babelfish-30B-A3B-GGUF Q8_0 | 30.3 GiB + 1.2 GiB audio encoder | about 36 GB (8 slots) / 46 GB (32 slots) | A800 (CUDA) | llama.cpp | greedy |
| Babelfish-30B-A3B-GGUF Q4_K_M | 17.3 GiB + 1.2 GiB audio encoder | about 23 GB (8 slots) / 32 GB (32 slots) | A800 (CUDA) | llama.cpp | greedy |

Every version defaults to greedy decoding with the repetition guard; `--beams 2` switches to beam search.

The service runs **stateful streaming sessions**: each session calls the model every 320 ms, and vLLM batches the calls of all sessions together. A step must take clearly less than 320 ms for a real-time session to keep up. When a session falls behind, the server merges the pending audio blocks into one step, so the delay does not grow, at a small cost in quality.

Measured on one A800 80GB: every session streams 60 s of audio in real time (RealSI excerpts, half English → Chinese, half Chinese → English). "Merged blocks" is the share of 320 ms blocks merged because the session fell behind; "text behind audio" is when a piece of text arrives minus the audio time it belongs to, i.e. what a deployment adds on top of the offline latency.

| Version | Decoding | Sessions | Step time p50 / p90 | Merged blocks | Text behind audio p50 / p90 |
|---|---|---|---|---|---|
| FP16 | beam 2 | 1 | 143 / 225 ms | 0 | 0.25 / 0.39 s |
| FP16 | beam 2 | 2 | 224 / 312 ms | 0 | 0.32 / 0.50 s |
| FP16 | beam 2 | 4 | 248 / 492 ms | 7.3% | 0.37 / 0.72 s |
| FP16 | greedy + guard | 1 | 111 / 149 ms | 0 | 0.18 / 0.24 s |
| FP16 | greedy + guard | 4 | 195 / 247 ms | 0.7% | 0.29 / 0.37 s |
| FP16 | greedy + guard | 8 | 280 / 484 ms | 9.8% | 0.56 / 0.90 s |
| FP8 | beam 2 | 1 | 132 / 185 ms | 0 | 0.20 / 0.27 s |
| FP8 | beam 2 | 2 | 211 / 282 ms | 0 | 0.28 / 0.38 s |
| FP8 | beam 2 | 4 | 242 / 467 ms | 6.5% | 0.35 / 0.72 s |

Rule of thumb for one A800: about 2 sessions with beam size 2, or about 4 with greedy decoding + repetition guard, without falling behind; run more instances for more sessions.

## 2. Installation

### 2.1 vLLM environment

Linux, Conda (Python 3.10+) and an NVIDIA driver. Tested: vLLM 0.28.0 (torch 2.13) and a vLLM 0.23 nightly (torch 2.11), both with transformers 5.13.1.

```bash
conda create -n babelfish python=3.12 -y
conda activate babelfish
pip install -U pip

# driver supports CUDA 13 (>= 580)
pip install vllm==0.28.0

# driver supports CUDA 12.x only (e.g. 535 / 550): CUDA 12.9 builds
pip install "vllm==0.28.0+cu129" "torch==2.13.0+cu129" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://wheels.vllm.ai/0.28.0/cu129

# this project (server, client, offline inference)
git clone https://github.com/timekettle/Babelfish-SimulST.git && cd Babelfish-SimulST
pip install -e .
```

### 2.2 Downloading the model

```bash
# Hugging Face (downloads only the FP16 subfolder into models/Babelfish-30B-A3B)
hf download timekettle/Babelfish-SimulST --include "Babelfish-30B-A3B/*" --local-dir models
```

Use `Babelfish-30B-A3B-FP8` for the FP8 version. The GGUF version is covered in section 7.

## 3. Starting the server

### 3.1 One-command script

`scripts/deploy.sh` (optionally) creates / uses a conda environment and installs the dependencies, downloads the model if given an ID, and starts the server in the foreground. It is ready when it prints `Ready: http://127.0.0.1:8080/`; Ctrl+C stops it. Downloaded models and logs are kept in `.simulst/`.

```bash
# start directly inside an active conda environment
conda activate babelfish
bash scripts/deploy.sh --model models/Babelfish-30B-A3B --gpus 0
bash scripts/deploy.sh --model models/Babelfish-30B-A3B --gpus 0,1                                # tensor parallel

# or pass --install to create the `babelfish` conda environment automatically
bash scripts/deploy.sh --install --model timekettle/Babelfish-SimulST/Babelfish-30B-A3B           # CUDA 13 driver
bash scripts/deploy.sh --install --cu129 --model timekettle/Babelfish-SimulST/Babelfish-30B-A3B   # CUDA 12.x driver
bash scripts/deploy.sh --install --model timekettle/Babelfish-SimulST/Babelfish-30B-A3B-FP8       # FP8 version
bash scripts/deploy.sh --model models/Babelfish-30B-A3B --dry-run                                 # print the commands only
```

| Option | Meaning | Default |
|---|---|---|
| `--model` | Hugging Face model ID (`repo` or `repo/subfolder`), or a local directory | required |
| `--install` | create / use a conda environment and install vLLM and this project | off |
| `--conda-env` | conda environment name for `--install` (reuses active non-base conda env by default) | `babelfish` |
| `--cu129` | install the CUDA 12.9 builds of vLLM / torch (for CUDA 12.x drivers) | off |
| `--python` | explicit Python of an existing environment (defaults to active conda `python3`) | `python3` |
| `--gpus` | GPUs to use, e.g. `0` or `0,1`; tensor parallel size defaults to the number of GPUs | all visible |
| `--tp` | tensor parallel size | number of GPUs |
| `--host` / `--port` | listen address / port | `0.0.0.0` / `8080` |
| `--beams` | default beam size: 1 = greedy (default, faster), 2 = beam search | `1` |
| `--max-sessions` | maximum concurrent sessions | `4` |
| `--gpu-mem` | vLLM GPU memory fraction | `0.9` |
| `--runtime-dir` | downloaded models and logs | `.simulst` |
| after `--` | passed to `simulst.server` unchanged | |

### 3.2 Starting directly

```bash
python -m simulst.server --model models/Babelfish-30B-A3B --host 0.0.0.0 --port 8080
```

| Option | Meaning | Default |
|---|---|---|
| `--model` | model directory or Hugging Face ID | required (vLLM) |
| `--beams` | default beam size (1 = greedy; clients may override it per session, 1–4) | `1` |
| `--max-sessions` | maximum concurrent sessions; further sessions get "server busy" | `4` |
| `--tp` | tensor parallel size | `1` |
| `--gpu-mem` | vLLM GPU memory fraction | `0.9` |
| `--max-seqs` | vLLM maximum concurrent sequences | `64` |
| `--dtype` | compute precision; the model is released and evaluated in FP16, keep `float16` | `float16` |
| `--window` / `--ctx-tokens` | audio window (s) / context tokens; keep the defaults to match the evaluation | `24` / `256` |
| `--loop-guard` | repetition guard (cuts runaway repetition only; see "Inference" in the README) | `on` |
| `--eager` | disable CUDA graphs (less memory, faster start, slower steps) | off |
| `--ssl-certfile` / `--ssl-keyfile` | serve HTTPS directly | none |
| `--backend llamacpp` | use the llama.cpp backend (section 7) | `vllm` |

### 3.3 Decoding: greedy or beam size 2

Greedy decoding with the repetition guard is the default: a step is about 25% faster than with beam size 2 and one GPU serves twice the sessions, at comparable quality. The results on the README use beam size 2 (`--beams 2`, or `"beams": 2` in the client configuration). Greedy decoding loops more often, which the repetition guard cuts before commit. FP16, measured:

| Test set | beam size 2 (README results) | greedy, no guard | greedy + repetition guard (default) |
|---|---|---|---|
| FLEURS en→zh | 0.8833 / 1.66 | 0.8857 / 1.67 | 0.8857 / 1.67 (never fired) |
| FLEURS zh→en | 0.8570 / 2.07 | 0.8595 / 1.90 | 0.8595 / 1.90 (never fired) |
| RealSI long-form en→zh | 0.8283 / 1.74 | 0.7931 / — (1 recording looped until stopped) | 0.8281 / 1.90 (fired on that recording only) |
| RealSI long-form zh→en | 0.7784 / 1.97 | 0.7805 / 1.99 | 0.7781 / 1.98 (never fired) |
| MCIF | 0.8680 / 1.50 | 0.8641 / 1.53 | 0.8630 / 1.53 (never fired) |

Cells are COMET / latency (s), FP16. Where the guard never fired, the two greedy columns differ only by vLLM batching noise.

### 3.4 Multiple GPUs and instances

- If the FP16 model does not fit on one 48 GB GPU, use two GPUs with tensor parallelism: `--gpus 0,1` (or `--tp 2`).
- For more concurrent sessions, run one instance per GPU (or GPU group) on different ports behind a load balancer. A WebSocket connection always stays on one instance, so no session affinity is needed.

## 4. Web page

Open `http://SERVER:8080/`:

1. pick a direction (English → Chinese / Chinese → English);
2. click "Start microphone" and speak, then "Finish"; or choose an audio / video file and click "Translate file": with "Play in real time" it is translated while playing, otherwise as fast as possible;
3. the bottom line shows how much audio has been processed and the time per step.

Browsers allow the microphone only on **HTTPS or localhost**. For a remote server:

```bash
# 1. SSH port forwarding (simplest), then open http://127.0.0.1:8080/ locally
ssh -N -L 8080:127.0.0.1:8080 user@server
# 2. HTTPS served directly
python -m simulst.server --model models/Babelfish-30B-A3B --ssl-certfile cert.pem --ssl-keyfile key.pem
```

3. HTTPS from a reverse proxy such as Nginx; forward WebSocket upgrades:

```nginx
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
    proxy_buffering off;          # deliver server-sent events one by one
}
```

The page uses relative URLs, so it also works under a sub-path (for example `https://host/proxy/8080/`).

## API

Interactive API docs are served at `/docs`.

### 5.1 Real-time WebSocket: `/translate/ws`

1. Send a JSON configuration first:

   | Field | Meaning | Default |
   |---|---|---|
   | `direction` | `en2zh` or `zh2en` (`en-zh` / `zh-en` also accepted) | `en2zh` |
   | `sample_rate` | audio sample rate; the server resamples to 16 kHz | `16000` |
   | `format` | `s16le` (16-bit integer) or `f32le` (32-bit float), little endian | `s16le` |
   | `channels` | number of channels; channels are averaged | `1` |
   | `realtime` | `true`: real-time pacing, pending blocks are merged when the server falls behind; `false`: every block is decoded (for streams faster than real time; identical to the offline evaluation) | `true` |
   | `beams` | beam size for this session (1–4) | server default |

2. After `{"event":"ready", ...}`, send audio as binary messages (any size; about 100 ms per message is a good choice).
3. When the audio ends, send `{"eof": true}` and keep reading until `done`.

Server messages:

| `event` | Fields | Meaning |
|---|---|---|
| `ready` | `session`, `direction`, `beams`, `model` | session started |
| `translation` | `delta` (new text), `text` (full translation so far), `audio_s` (seconds of audio received when the text was output), `elapsed_s`, `step_ms` | new text |
| `done` | `text`, `audio_s`, `elapsed_s`, `steps`, `skipped_blocks`, `loop_guards` (repetition-guard cuts), `step_ms_p50`, `step_ms_p90`, `emissions` | finished |
| `error` | `message` | error (e.g. server busy); the connection is closed |

Output is never revised, so a client can append `delta`; in rare cases (a Chinese character split across two tokens) rely on `text`.

Python example (`pip install websockets numpy soundfile`):

```python
import asyncio, json
import numpy as np, soundfile as sf, websockets

async def main(path, direction='en2zh', url='ws://127.0.0.1:8080/translate/ws'):
    wav, sr = sf.read(path, dtype='int16')
    if wav.ndim > 1:
        wav = wav.mean(1).astype('int16')
    async with websockets.connect(url, max_size=2**24) as ws:
        await ws.send(json.dumps(dict(direction=direction, sample_rate=sr, format='s16le', channels=1)))
        print(await ws.recv())                       # ready
        async def send():
            hop = sr // 10                           # 100 ms
            for i in range(0, len(wav), hop):
                await ws.send(wav[i:i + hop].tobytes())
                await asyncio.sleep(0.1)             # real-time pacing
            await ws.send(json.dumps({'eof': True}))
        task = asyncio.create_task(send())
        async for msg in ws:
            ev = json.loads(msg)
            if ev['event'] == 'translation':
                print(ev['delta'], end='', flush=True)
            elif ev['event'] in ('done', 'error'):
                print('\n', ev.get('text') or ev.get('message'))
                break
        task.cancel()

asyncio.run(main('audio.wav'))
```

The bundled client does more (fast or real-time streaming, saving all events):

```bash
python -m simulst.client audio.wav --direction en2zh --url http://127.0.0.1:8080 [--realtime] [--save events.jsonl]
```

### 5.2 File upload with streamed results: `POST /translate/stream`

Multipart form: `file` (any format libsndfile can read: wav, flac, ogg, ...), `direction`, optional `realtime` (`true` = process at real-time speed) and `beams`. Returns `text/event-stream`; every `data:` line is an event as on the WebSocket.

```bash
curl -N http://127.0.0.1:8080/translate/stream -F direction=zh2en -F file=@audio.wav
```

### 5.3 File upload, single response: `POST /translate`

Same parameters; returns the final `done` event (full translation and the time of every output).

```bash
curl http://127.0.0.1:8080/translate -F direction=en2zh -F file=@audio.wav
```

### 5.4 Status: `GET /health`

Model name, backend, active sessions, default beam size, window settings.

## 6. FP8 version

`Babelfish-30B-A3B-FP8` stores the language-model attention projections and all expert weights in FP8 (E4M3, one scale per 128×128 block) and quantizes activations dynamically at run time; router gates, embeddings and the audio encoder stay FP16. Weights shrink from 58 GiB to 30 GiB.

- On Hopper / Ada (H100, H20, L40S, ...) vLLM computes in FP8;
- Ampere (A100, A800, ...) has no FP8 units: vLLM falls back to Marlin kernels (weight-only FP8, FP16 compute), which saves memory but is not faster.

Usage is the same as for the FP16 model:

```bash
python -m simulst.server --model models/Babelfish-30B-A3B-FP8 --port 8080
```

The FP8 weights are converted directly from the FP16 version (one scale per 128×128 block, no calibration data). Quality:

| Test set | FP16: COMET / latency (s) | FP8: COMET / latency (s) | ΔCOMET |
|---|---|---|---|
| MCIF | 0.8680 / 1.50 | 0.8684 / 1.48 | +0.0004 |
| RealSI long-form en→zh | 0.8283 / 1.74 | 0.8264 / 1.70 | -0.0019 |
| FLEURS en→zh | 0.8833 / 1.66 | 0.8811 / 1.62 | -0.0022 |
| CoVoST 2 en→zh | 0.8706 / 1.81 | 0.8702 / 1.78 | -0.0004 |
| ACL 60/60 dev | 0.8566 / 1.81 | 0.8555 / 1.77 | -0.0011 |
| ACL 60/60 eval | 0.8517 / 1.92 | 0.8504 / 1.84 | -0.0013 |
| RealSI sentences en→zh | 0.8294 / 1.75 | 0.8302 / 1.73 | +0.0008 |
| RealSI long-form zh→en | 0.7784 / 1.97 | 0.7749 / 1.92 | -0.0035 |
| FLEURS zh→en | 0.8570 / 2.07 | 0.8566 / 2.03 | -0.0004 |
| CoVoST 2 zh→en | 0.8061 / 2.11 | 0.8059 / 2.08 | -0.0002 |
| RealSI sentences zh→en | 0.7948 / 2.15 | 0.7947 / 2.11 | -0.0001 |

Mean over the 11 sets: COMET −0.0009, latency 40 ms lower (both with beam size 2, as the README results; FP8 run with weight-only Marlin kernels on A800).

## 7. llama.cpp (GGUF version)

The GGUF version runs on llama.cpp: `llama-server` loads the language model and the audio encoder, and this project's server (`--backend llamacpp`) provides the same web page and APIs on top of it, with the same streaming logic and repetition guard as the vLLM version.

**Files** (`Babelfish-30B-A3B-GGUF`): `Babelfish-30B-A3B-Q8_0.gguf` (30.3 GiB), `Babelfish-30B-A3B-Q4_K_M.gguf` (17.3 GiB), `mmproj-Babelfish-30B-A3B-F16.gguf` (audio encoder, 1.2 GiB); checksums in the repository's `SHA256SUMS`.

### 7.1 Building llama.cpp (with the audio patch)

For Qwen3-Omni audio, llama.cpp pads the last partial second to a full second and encodes the padding, adding 0–12 padding tokens at the end of the audio of every step, so the model believes it has heard more audio and speaks too early. The patch in [patches/llama.cpp](../patches/llama.cpp) keeps only the valid tokens of the last chunk (as vLLM / Hugging Face do). Without it the model still runs, but quality and latency deviate (see 7.3).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 3f8af15a76ac8675812e856da45454bafc90f12c        # the patch is made against this commit (2026-11-02)
git apply /path/to/Babelfish-SimulST/patches/llama.cpp/qwen3-omni-audio-valid-tokens.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```

- Building with CUDA 12.8+ for a driver older than r570: add `-DGGML_CUDA_COMPRESSION_MODE=none`, otherwise it fails at run time with `device kernel image is invalid`.
- Without internet access during the build, add `-DLLAMA_BUILD_HTML=OFF` to skip downloading the web UI.
- Do not run two llama-server instances on one GPU: we saw occasional `illegal memory access` crashes in the audio encoder when two shared a GPU, and none with one server per GPU.

### 7.2 Starting

One-command script (starts llama-server, then the web / API service; Ctrl+C stops both):

```bash
bash scripts/deploy_llamacpp.sh --model timekettle/Babelfish-SimulST/Babelfish-30B-A3B-GGUF --quant Q4_K_M \
  --llama-server /path/to/llama.cpp/build/bin/llama-server --gpus 0 [--sessions 4] [--beams 1]
```

Or in two steps:

```bash
llama-server -m Babelfish-30B-A3B-Q4_K_M.gguf --mmproj mmproj-Babelfish-30B-A3B-F16.gguf -ngl 999 \
  --parallel 8 -c 32768 --host 127.0.0.1 --port 8081
python -m simulst.server --backend llamacpp --llama-url http://127.0.0.1:8081 --port 8080 --max-sessions 8
```

- `--parallel` (slots) must be at least concurrent sessions × beam size; use `-c` = 4096 × slots. Memory: Q8_0 about 36 GB with 8 slots and 46 GB with 32; Q4_K_M about 23 GB with 8 slots and 32 GB with 32.
- The server reads the vocabulary from the GGUF file at the path llama-server reports; if llama-server runs on another machine, point `--tokenizer` to the FP16 version's directory (only its tokenizer files are used).

### 7.3 Quality and speed

| System | FLEURS en→zh | FLEURS zh→en | RealSI long-form en→zh (1 recording) |
|---|---|---|---|
| vLLM FP16 greedy | 0.8886 / 1.67 | 0.8497 / 2.18 | 0.8341 / 1.95 |
| vLLM FP16 beam size 2 | 0.8877 / 1.62 | 0.8464 / 2.30 | 0.8451 / 1.98 |
| **GGUF Q8_0 greedy (patched)** | **0.8876 / 1.65** | **0.8483 / 2.16** | **0.8295 / 1.89** |
| GGUF Q8_0 greedy (official llama.cpp) | 0.8740 / 0.99 | 0.8480 / 1.63 | 0.8172 / 1.50 |
| GGUF Q8_0 beam size 2 (official) | 0.8742 / 0.97 | 0.8352 / 1.78 | 0.8136 / 1.57 |
| GGUF Q4_K_M greedy (official) | 0.8746 / 1.11 | 0.8459 / 1.75 | 0.8212 / 1.52 |
| GGUF Q4_K_M beam size 2 (official) | 0.8777 / 1.07 | 0.8369 / 1.73 | 0.8113 / 1.60 |

Cells are COMET / latency (s) on the same items: the first 100 FLEURS sentences per direction and the first RealSI long-form English → Chinese recording (205 s), repetition guard on (it never fired). The low latency of the official llama.cpp is the padding making the model speak too early, at a cost in quality. Q8_0 and Q4_K_M differ by less than 0.004. One long recording can differ by about 0.017 between concurrent runs, so read it as a direction only.

**Decoding**: greedy with the repetition guard by default. Beam search on llama.cpp uses the same algorithm (each round asks for the top 2 × beam-size next tokens; every beam keeps its own slot and cache), and with beam size 1 it reproduces greedy decoding token for token; but on llama.cpp beam size 2 was not better than greedy (about 0.01 lower for Chinese → English) and doubles the step time, so it is optional (`--beams 2`, with twice the slots).

**Speed** (A800, one server per GPU, Q8_0): with a 24 s window, a greedy step takes 530 ms at the median (1,030 ms with beam size 2); about 190 ms for short sentences. llama.cpp does not batch the audio prefill of several sessions, serving about 4 requests per second under the same load (vLLM about 50). Real-time sessions on long audio therefore fall behind the 320 ms cadence and merge blocks, with more delay than the offline measurement (measured: one Q4_K_M session streaming 60 s of audio in real time, 481 ms per step at the median, about 25% of the blocks merged): the GGUF version suits offline jobs, short utterances and environments without vLLM; prefer the vLLM version for real-time service.

## 8. Offline batch inference

`simulst.infer.stream` processes audio files in batch: it reads an audio manifest, simulates streaming input 320 ms at a time, batches many files together and records when every character was output, with the same streaming logic as the live server.

One JSON object per line (`audio_path`: 16 kHz mono audio; `direction`: `en2zh` or `zh2en`):

```json
{"id": "talk_0001", "audio_path": "/data/talk_0001.wav", "direction": "en2zh"}
```

```bash
python -m simulst.infer.stream --model models/Babelfish-30B-A3B \
  --manifests talks_en=talks_en.jsonl,talks_zh=talks_zh.jsonl --out outputs/ [--beams 2] [--shard i --shards n]
```

Results go to `outputs/<name>/results.<shard>.jsonl`: the translation (`hypothesis`), every output (`emissions`, with time and text) and the output time of every character (`delays_ms`). `--shard / --shards` spread the items over GPUs; with the FP16 model, decoding one file at a time (`--max-live 1`) is fully deterministic (the FP8 model on A800 still varies slightly between runs); batching many files causes tiny numerical differences.

Scoring (COMET, BLEU, latency) and reproducing the README results: [EVALUATION.md](EVALUATION.md).

## 9. FAQ

**CUDA out of memory**: use the FP8 version, or two GPUs with tensor parallelism, or lower `--gpu-mem` / `--max-seqs`, or add `--eager` to save the CUDA-graph memory.

**The microphone does not work in the browser**: browsers only allow it on HTTPS or localhost, see section 4.

**Text arrives late, many `skipped_blocks`**: steps take longer than 320 ms. Check `step_ms_p50 / p90` in the `done` event; reduce concurrent sessions or run more instances; if you use beam size 2, go back to the default greedy decoding.

**The same sentence is repeated over and over**: a decoding loop. The repetition guard is on by default (`--loop-guard on`) and cuts a step where runaway copying starts; turn it back on if it was disabled (`off`).

**Can I use bfloat16?** The model is released and evaluated in FP16; keep the default `--dtype float16`.

**Loading the weights without this repository**: the weights contain only the audio encoder and the language model (no vision encoder, no speech output). With vLLM directly, pass `--limit-mm-per-prompt '{"audio": 1, "image": 0, "video": 0}'` so that vLLM does not build the vision encoder; transformers warns that the vision weights are missing and initializes them randomly, which does not affect speech input.

**vLLM from pip fails with CUDA / driver errors**: the default PyPI builds need a CUDA 13 driver (>= 580); with an older driver install the CUDA 12.9 builds (section 2.1).

**Audio formats**: the WebSocket takes raw PCM (`s16le` / `f32le`, any sample rate and channel count); the upload endpoints take what libsndfile decodes (wav, flac, ogg, ...); files on the web page are decoded by the browser, so common audio and video formats work.
