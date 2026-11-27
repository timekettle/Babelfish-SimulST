**English** | [简体中文](README_zh.md)

<h1 align="center">Babelfish</h1>

<p align="center"><b>Low-latency, high-quality streaming simultaneous translation between English and Chinese</b></p>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Babelfish-ffc107)](https://huggingface.co/timekettle/Babelfish-SimulMT)
[![GitHub](https://img.shields.io/badge/GitHub-Babelfish-181717?logo=github)](https://github.com/timekettle/Babelfish-SimulMT)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

</div>

<!-- Demo video source backups are kept in assets/demo_en2zh.mp4 and assets/demo_zh2en.mp4 -->
<div align="center">
<table>
  <tr>
    <td align="center"><b>English → Chinese</b></td>
    <td align="center"><b>Chinese → English</b></td>
  </tr>
  <tr>
    <td align="center"><video src="https://github.com/user-attachments/assets/166bb4c8-92e4-46b6-a2a8-ae51116bcb96" width="380" controls></video></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/ae1ecb7b-96f1-4a4c-b120-3bff67feb0d6" width="380" controls></video></td>
  </tr>
</table>
<sub>The translations in the demos are generated in real time by the Babelfish-30B-A3B model released here (beam size 2).<br>The source transcript and translated speech (with speaker voice cloning) shown in the UI come from separate ASR and TTS models not part of this release.<br>Demo translation latency: English → Chinese StreamLAAL 1.20 s (1.48 s with computation; first character 1.33 s), Chinese → English StreamLAAL 1.57 s (1.81 s with computation; first word 2.22 s); first-audio latency is 2.10 s and 4.47 s, respectively. First-audio latency is longer because the external voice-cloning TTS first buffers enough source speech to capture the speaker's timbre and waits for a complete initial phrase before synthesizing and playing audio (the Chinese → English demo opens with a longer clause), so voice quality and prosody remain natural.</sub>
</div>

## Introduction

**Babelfish-30B-A3B** is a speech-to-text simultaneous interpretation model for **English → Chinese** and **Chinese → English**: speech streams in, the model translates while listening, and text that has been output is never revised. It is fine-tuned for streaming from Qwen3-Omni-30B-A3B-Instruct (mixture of experts, about 3B parameters active per token); see the technical report (link to be added) for the method and training details.

This repository releases:

- model weights: the original FP16 model, an FP8 version (vLLM) and GGUF versions (llama.cpp);
- inference and deployment code: a web page (microphone / file upload), a real-time WebSocket API, a file upload API, a command-line client and a one-command deployment script;
- offline batch inference and evaluation scripts.

Training code and training-data construction are not released at this time.

## News

- **2026.11.XX | Release**: Babelfish-30B-A3B with FP8 and GGUF versions, plus inference and deployment code; technical report (link to be added).

## Results

We evaluated on every test set reported by Hy-SimulMT (FLEURS, CoVoST 2, ACL 60/60, RealSI sentences, RealSI long-form) plus MCIF long-form, 11 test sets in total, against the open Hy-SimulMT and four commercial simultaneous interpretation services.

- **Quality**: the highest COMET on 10 of the 11 test sets and the highest CometKiwi on 9. Average COMET is 85.5 for English → Chinese and 80.9 for Chinese → English; the next best, Qwen3.8-LiveTranslate, has 84.3 and 79.7. The only set without the highest COMET is Chinese → English RealSI long-form, 0.2 below Seed LiveInterpret 2.0.
- **Latency**: LAAL of 1.50–1.92 s for English → Chinese and 1.97–2.15 s for Chinese → English, 1.74 s and 2.08 s on average.

COMET and CometKiwi are on a 0–100 scale. The best value in each column of each metric is in bold (the lowest for latency). † marks commercial services, whose latency is measured differently (see the notes below the tables).

**English → Chinese**

| System | MCIF | RealSI long-form | RealSI sentences | FLEURS | CoVoST 2 | ACL 60/60 dev | ACL 60/60 eval |
|---|---|---|---|---|---|---|---|
| *COMET ↑* | | | | | | | |
| **Babelfish-30B-A3B** | **86.8** | **82.8** | **82.9** | **88.3** | **87.1** | **85.7** | **85.2** |
| Hy-SimulMT-30B-A3B | 85.2 | 79.4 | 77.7 | 86.7 | 83.9 | 83.3 | 82.5 |
| Qwen3.8-LiveTranslate † | 86.6 | 82.2 | 80.4 | 86.9 | 85.6 | 84.3 | 84.0 |
| Seed LiveInterpret 2.0 † | 86.2 | 81.7 | 79.2 | 85.8 | 81.2 | 83.3 | 81.2 |
| GPT-Realtime-Translate † | 84.1 | 77.5 | 77.6 | 83.0 | 75.0\* | 80.6 | 80.6 |
| Gemini 3.5 Live Translate † | 83.1 | 76.3 | 75.6 | 84.0 | 80.4 | 80.8 | 78.9 |
| *CometKiwi ↑* | | | | | | | |
| **Babelfish-30B-A3B** | **79.7** | **73.2** | **75.5** | **84.2** | **80.8** | **78.4** | **76.6** |
| Hy-SimulMT-30B-A3B | 76.9 | 70.4 | 70.6 | 81.9 | 76.8 | 74.4 | 72.5 |
| Qwen3.8-LiveTranslate † | 78.5 | 71.3 | 72.1 | 81.3 | 78.0 | 75.3 | 73.0 |
| Seed LiveInterpret 2.0 † | 77.5 | 71.4 | 69.7 | 80.4 | 72.4 | 73.7 | 69.7 |
| GPT-Realtime-Translate † | 76.7 | 67.5 | 69.8 | 78.0 | 66.7\* | 72.5 | 70.9 |
| Gemini 3.5 Live Translate † | 74.2 | 67.8 | 67.7 | 78.3 | 72.0 | 70.7 | 67.8 |
| *BLEU ↑* | | | | | | | |
| **Babelfish-30B-A3B** | 40.2 | 25.2 | 24.0 | 40.3 | **41.4** | 41.8 | 41.9 |
| Hy-SimulMT-30B-A3B | 52.3 | 29.8 | 28.5 | 42.6 | 38.2 | 49.9 | 48.3 |
| Qwen3.8-LiveTranslate † | 45.1 | 27.0 | 25.3 | 38.2 | 38.1 | 43.1 | 43.0 |
| Seed LiveInterpret 2.0 † | **55.6** | **33.3** | **31.3** | **43.1** | 39.6 | **51.9** | **48.9** |
| GPT-Realtime-Translate † | 38.4 | 25.0 | 22.8 | 31.1 | 28.0\* | 33.8 | 34.6 |
| Gemini 3.5 Live Translate † | 48.3 | 28.4 | 27.0 | 40.3 | 38.7 | 46.5 | 46.1 |
| *LAAL (s) ↓* | | | | | | | |
| **Babelfish-30B-A3B** | **1.50** | **1.74** | **1.75** | **1.66** | **1.81** | 1.81 | 1.92 |
| Hy-SimulMT-30B-A3B | 2.15 | 2.66 | 1.88 | 2.09 | 1.96 | **1.77** | **1.74** |
| Qwen3.8-LiveTranslate † | 4.07 | 3.69 | 2.18 | 2.30 | 2.66 | 2.44 | 2.51 |
| Seed LiveInterpret 2.0 † | 2.51 | 2.78 | 2.02 | 2.15 | 1.86 | 2.10 | 2.02 |
| GPT-Realtime-Translate † | 4.50 | 4.86 | 2.92 | 3.55 | 3.31\* | 2.87 | 2.92 |
| Gemini 3.5 Live Translate † | 1.98 | 2.24 | 2.74 | 2.53 | 2.95 | 2.67 | 2.55 |

**Chinese → English**

| System | RealSI long-form | RealSI sentences | FLEURS | CoVoST 2 |
|---|---|---|---|---|
| *COMET ↑* | | | | |
| **Babelfish-30B-A3B** | 77.8 | **79.5** | **85.7** | **80.6** |
| Hy-SimulMT-30B-A3B | 75.1 | 75.9 | 85.2 | 78.3 |
| Qwen3.8-LiveTranslate † | 77.5 | 77.7 | 84.4 | 79.0 |
| Seed LiveInterpret 2.0 † | **78.0** | 77.8 | 84.5 | 75.2 |
| GPT-Realtime-Translate † | 74.5 | 73.3 | 79.6 | 65.8\* |
| Gemini 3.5 Live Translate † | 72.2 | 72.3 | 80.4 | 66.0 |
| *CometKiwi ↑* | | | | |
| **Babelfish-30B-A3B** | 67.8 | 73.5 | **83.6** | **77.3** |
| Hy-SimulMT-30B-A3B | **68.7** | 72.5 | 82.5 | 74.1 |
| Qwen3.8-LiveTranslate † | 67.9 | **73.9** | 81.9 | 74.8 |
| Seed LiveInterpret 2.0 † | 67.0 | 69.5 | 81.4 | 69.9 |
| GPT-Realtime-Translate † | 65.1 | 67.8 | 76.2 | 59.8\* |
| Gemini 3.5 Live Translate † | 65.6 | 67.2 | 75.8 | 59.7 |
| *BLEU ↑* | | | | |
| **Babelfish-30B-A3B** | 21.0 | 20.3 | 22.7 | **23.6** |
| Hy-SimulMT-30B-A3B | 22.1 | 19.7 | **23.7** | 22.9 |
| Qwen3.8-LiveTranslate † | 20.4 | 19.5 | 20.7 | 21.5 |
| Seed LiveInterpret 2.0 † | **28.3** | **25.3** | 22.7 | 22.2 |
| GPT-Realtime-Translate † | 17.2 | 14.4 | 16.1 | 11.8\* |
| Gemini 3.5 Live Translate † | 16.8 | 16.8 | 19.4 | 14.6 |
| *LAAL (s) ↓* | | | | |
| **Babelfish-30B-A3B** | 1.97 | 2.15 | **2.07** | **2.11** |
| Hy-SimulMT-30B-A3B | **1.91** | **1.59** | 2.30 | 2.20 |
| Qwen3.8-LiveTranslate † | 3.56 | 2.42 | 2.73 | 3.24 |
| Seed LiveInterpret 2.0 † | 2.82 | 2.19 | 2.70 | 2.13 |
| GPT-Realtime-Translate † | 4.15 | 3.10 | 3.56 | 3.79\* |
| Gemini 3.5 Live Translate † | 2.06 | 2.80 | 2.82 | 3.05 |

<details>
<summary>Test conditions and metrics</summary>

- **COMET**: wmt22-comet-da; **CometKiwi**: wmt22-cometkiwi-da (reference-free); **BLEU**: SacreBLEU (zh tokenizer for Chinese output, 13a for English). The raw output is scored (punctuation kept); sentence-level sets are scored per sentence; long-form sets (MCIF, RealSI long-form) are first re-segmented to the official reference sentences with OmniSTEval.
- **Latency**: LAAL for sentence-level sets, StreamLAAL (LAAL after re-segmentation) for long-form sets; characters for Chinese output, words for English output.
- **The two groups of latencies are measured differently and are not directly comparable**: Babelfish and Hy-SimulMT are simulated offline on the audio timeline, without computation time; the commercial services were fed audio in real time and their latency is the wall-clock time at which the text arrived, including network round trips and server-side computation.
- Babelfish results use beam size 2 (`--beams 2`); the released default is greedy decoding with the repetition guard, of comparable quality (slightly higher on FLEURS, 0.5 lower on MCIF; see [deployment guide 3.3](docs/DEPLOYMENT.md#33-decoding-greedy-or-beam-size-2)). Hy-SimulMT-30B-A3B uses the official SimulInfer defaults (1.6 s chunks, VAD on).
- \* Approximate: GPT-Realtime-Translate produced valid results for only part of CoVoST 2; the values measured on those sentences are extrapolated to the full set with the average difference of the other systems (about ±0.5 COMET).
- Test set sizes: MCIF 21 talks; RealSI long-form 10 recordings per direction; FLEURS 646 / 945 sentences; CoVoST 2 15,530 / 4,898; ACL 60/60 dev 468, eval 416; RealSI sentences 346 / 431.
- Raw scores on a 0–1 scale, quantized versions and decoding settings: [docs/EVALUATION.md](docs/EVALUATION.md).

</details>

## Model links

| Model | Description | Framework | Download |
|---|---|---|---|
| Babelfish-30B-A3B | original FP16 model | vLLM | [Hugging Face](https://huggingface.co/timekettle/Babelfish-SimulMT/tree/main/Babelfish-30B-A3B) |
| Babelfish-30B-A3B-FP8 | FP8 (one scale per 128×128 block) | vLLM | [Hugging Face](https://huggingface.co/timekettle/Babelfish-SimulMT/tree/main/Babelfish-30B-A3B-FP8) |
| Babelfish-30B-A3B-GGUF | Q8_0 / Q4_K_M + audio encoder | llama.cpp | [Hugging Face](https://huggingface.co/timekettle/Babelfish-SimulMT/tree/main/Babelfish-30B-A3B-GGUF) |

| Version | Weights | GPU memory | Quality | Use |
|---|---|---|---|---|
| FP16 | 58 GiB | 80 GB × 1, or 48 GB × 2 | used for the reported results | servers |
| FP8 | 30 GiB | 48 GB × 1 | mean COMET 0.09 lower over the 11 test sets (0–100 scale) | one 48 GB GPU; native FP8 on H100 / H20 / L40S |
| GGUF Q8_0 / Q4_K_M | 30.3 / 17.3 GiB + 1.2 GiB audio encoder | about 36 / 23 GB | on par with vLLM with the llama.cpp audio patch | llama.cpp; environments without vLLM, offline jobs |

## Deployment

> Details (environment, GPU memory, multi-GPU, HTTPS, API protocol, FAQ) are in the **[deployment guide](docs/DEPLOYMENT.md)**.

### Requirements

- Linux, Conda (Python 3.10+), NVIDIA GPU: one 80 GB GPU for the FP16 model (or two 48 GB GPUs with tensor parallelism), 48 GB or more for the FP8 model;
- vLLM (tested with 0.28.0 and a 0.23 nightly, with transformers 5.13.1);
- see below for the GGUF version with llama.cpp.

### Setup and deployment (vLLM)

Create and activate a conda environment, then install the dependencies:

```bash
git clone https://github.com/timekettle/Babelfish-SimulMT.git
cd Babelfish-SimulMT

conda create -n babelfish python=3.12 -y
conda activate babelfish

# driver supports CUDA 13 (>= 580)
pip install vllm==0.28.0
# driver supports CUDA 12.x only (e.g. 535 / 550): use this instead
# pip install "vllm==0.28.0+cu129" "torch==2.13.0+cu129" \
#   --extra-index-url https://download.pytorch.org/whl/cu129 \
#   --extra-index-url https://wheels.vllm.ai/0.28.0/cu129

pip install -e .
```

With the `babelfish` environment active, start the server (passing a model ID downloads it automatically; once you see `Ready: http://127.0.0.1:8080/` it is ready; Ctrl+C stops it):

```bash
# download from Hugging Face and start (FP16)
bash scripts/deploy.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B
# FP8 version
bash scripts/deploy.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B-FP8
# local model, single GPU or tensor parallelism over two GPUs
bash scripts/deploy.sh --model /path/to/Babelfish-30B-A3B --gpus 0
bash scripts/deploy.sh --model /path/to/Babelfish-30B-A3B --gpus 0,1
```

You can also pass `--install` to let the script create and set up the `babelfish` conda environment automatically (add `--cu129` for CUDA 12.x drivers):

```bash
bash scripts/deploy.sh --install --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B
```

Or start the server directly inside the `babelfish` conda environment:

```bash
python -m simulst.server --model /path/to/Babelfish-30B-A3B --port 8080
```

### Usage

- **Web page**: open `http://SERVER:8080/`, pick a direction, click "Start microphone", or choose an audio file ("Play in real time" translates while playing). Browsers allow the microphone only on HTTPS or localhost; the simplest way for a remote server is SSH port forwarding: `ssh -N -L 8080:127.0.0.1:8080 user@server`, then open `http://127.0.0.1:8080/`.
- **Command-line client**:

  ```bash
  python -m simulst.client audio.wav --direction en2zh --url http://127.0.0.1:8080             # as fast as possible
  python -m simulst.client audio.wav --direction zh2en --url http://127.0.0.1:8080 --realtime  # real-time streaming
  ```

- **File upload with streamed results (SSE)**:

  ```bash
  curl -N http://127.0.0.1:8080/translate/stream -F direction=en2zh -F file=@audio.wav
  ```

- **Real-time WebSocket**: connect to `ws://SERVER:8080/translate/ws`, send `{"direction":"en2zh","sample_rate":16000,"format":"s16le","channels":1}`, wait for `ready`, send binary PCM audio, send `{"eof":true}` at the end and read events until `done`. See [deployment guide · API](docs/DEPLOYMENT.md#api).

### llama.cpp (GGUF)

The GGUF version runs on llama.cpp (Q8_0 30.3 GiB / Q4_K_M 17.3 GiB, plus a 1.2 GiB audio encoder) with the same streaming logic and repetition guard as the vLLM version.

**To match the vLLM version, llama.cpp needs an audio patch** ([patches/llama.cpp](patches/llama.cpp)): llama.cpp pads the last partial second of audio and encodes the padding, so the model believes it has heard more audio than it has and speaks too early. With the patch, Q8_0 is on par with vLLM (FLEURS English → Chinese COMET 88.8 vs 88.9); without it the model still runs, with COMET about 1.4 lower and output about 0.65 s earlier.

```bash
# build llama.cpp with the patch
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 3f8af15a76ac8675812e856da45454bafc90f12c
git apply ../patches/llama.cpp/qwen3-omni-audio-valid-tokens.patch      # path when cloned inside the Babelfish-SimulMT repository
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server && cd ..
# start llama-server and the web / API service (same APIs as the vLLM version)
bash scripts/deploy_llamacpp.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B-GGUF --quant Q4_K_M --llama-server llama.cpp/build/bin/llama-server
```

Greedy decoding with the repetition guard is the default. Beam search also works on llama.cpp (`--beams 2`), but it was not better than greedy there and doubles the step time, so it is not the default. llama.cpp does not batch the audio prefill of several sessions, so on an A800 a step with a 24 s window takes about 0.5 s (more than the 0.32 s cadence): real-time sessions on long audio merge blocks and fall behind, which makes this version a better fit for offline jobs, short utterances and environments without vLLM. Details: [deployment guide · llama.cpp](docs/DEPLOYMENT.md#7-llamacpp-gguf-version).

### Offline batch inference and evaluation

Translate a batch of audio files in streaming mode (each output has the translation and the time every character was output); manifest format in the [deployment guide · offline batch inference](docs/DEPLOYMENT.md#8-offline-batch-inference):

```bash
python -m simulst.infer.stream --model /path/to/Babelfish-30B-A3B --manifests talks=talks.jsonl --out outputs/
```

To reproduce the COMET, BLEU and latency results above, decode with beam size 2 and score:

```bash
python -m simulst.infer.stream --model /path/to/Babelfish-30B-A3B --beams 2 --manifests fleurs=fleurs_en2zh.jsonl --out outputs/
python -m simulst.eval.score_sentences outputs/fleurs
```

Test-set preparation, long-form scoring and the full results: [docs/EVALUATION.md](docs/EVALUATION.md).

## License

Code and model weights are released under the [Apache License 2.0](LICENSE). The model is fine-tuned from Qwen3-Omni-30B-A3B-Instruct (Apache License 2.0); see [NOTICE](NOTICE).

## Contributors

Shiyang Deng, Nanjie Li, Xiaoyong Guo, Haihua Xu, Wei Shi, Hao Huang, Xiansu Deng, Jun Guo, Yuanming Luo, Wangzheng Wu

## Citation

If Babelfish helps your work, please cite:

```bibtex
@misc{babelfish2026,
  title  = {Babelfish: Low-Latency Streaming Simultaneous Speech Translation between English and Chinese},
  author = {Shiyang Deng and Nanjie Li and Xiaoyong Guo and Haihua Xu and Wei Shi and Hao Huang and Xiansu Deng and Jun Guo and Yuanming Luo and Wangzheng Wu},
  year   = {2026},
  url    = {https://github.com/timekettle/Babelfish-SimulMT}
}
```

## Contact

Please use GitHub Issues for questions and feedback.
