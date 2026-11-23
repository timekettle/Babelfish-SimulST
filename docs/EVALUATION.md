**English** | [简体中文](EVALUATION_zh.md) | [Back to README](../README.md)

# Evaluation

How to reproduce the reported results: test-set manifests, offline streaming decoding, scoring, and the full comparison.

## 1. Manifest format

One JSON object per line:

```json
{"id": "fleurs_en_0001", "audio_path": "/data/fleurs/en_us/audio/test/0001.wav", "direction": "en2zh",
 "transcription": "source text (optional, needed by COMET)", "translations": {"zh": "reference (needed for scoring)"}}
```

- `audio_path`: 16 kHz mono audio;
- `direction`: `en2zh` or `zh2en`;
- long-form recordings are also one line each; scoring needs the sentence-level references (section 3).

## 2. Test sets

`simulst.eval.build_testsets` writes the manifests:

```bash
# FLEURS (en_us and cmn_hans_cn from google/fleurs, with test.tsv and the extracted audio/test/*.wav)
python -m simulst.eval.build_testsets fleurs --fleurs-en fleurs/en_us --fleurs-zh fleurs/cmn_hans_cn --out testsets
# CoVoST 2 (test-split TSVs from the official script + Common Voice 4 audio; mp3 is converted to 16 kHz wav)
python -m simulst.eval.build_testsets covost --covost-tsv covost_tsv --cv-root cv-corpus-4 --out testsets
# RealSI long-form (also writes official_inputs for long-form scoring and a sentence-level version)
python -m simulst.eval.build_testsets realsi --realsi RealSI --out testsets
```

ACL 60/60 and MCIF manifests follow the same format; long-form scoring needs `official_inputs` (`audio-segments.yaml` plus one text file per language, aligned line by line).

## 3. Decoding and scoring

```bash
# decode (many files are batched; --shard/--shards split the items over GPUs); the README results use beam size 2
# (--beams 2); the default is greedy decoding with the repetition guard
python -m simulst.infer.stream --model models/Babelfish-30B-A3B --beams 2 \
  --manifests fleurs_en2zh=testsets/fleurs_en2zh.jsonl,realsi_en2zh=testsets/realsi_en2zh.jsonl --out outputs

# sentence-level: BLEU, COMET (wmt22-comet-da), LAAL / AL
python -m simulst.eval.score_sentences outputs/fleurs_en2zh
# long-form: OmniSTEval re-segmentation to the reference sentences, then BLEU, chrF, COMET, StreamLAAL
python -m simulst.eval.score_longform outputs/realsi_en2zh testsets/realsi_en2zh/official_inputs
# summary table
python -m simulst.eval.report outputs
```

All GPUs at once (one decoding process per GPU, scoring afterwards):

```bash
MODEL=models/Babelfish-30B-A3B OUT=outputs DECODE_ARGS="--beams 2" bash scripts/evaluate.sh \
  fleurs_en2zh=testsets/fleurs_en2zh.jsonl realsi_en2zh=testsets/realsi_en2zh.jsonl@testsets/realsi_en2zh/official_inputs
```

Scoring dependencies: `pip install -r requirements/eval.txt`.

**Latency**: the delay of a character is the amount of audio received when it is output, without computation time (non-computation-aware). Characters for Chinese output, words for English output (a word counts as output with its last letter). LAAL for sentence-level sets, StreamLAAL after re-segmentation for long-form sets. A real deployment adds the per-step computation time; see the speed table in the [deployment guide](DEPLOYMENT.md).

**Reproducibility**: when vLLM batches many files together, the batch composition causes tiny numerical differences, so single outputs may differ in a few characters between runs; overall scores move by about ±0.001 COMET. With the FP16 model, decoding one file at a time (`--max-live 1`) is fully deterministic; the FP8 model on A800 (Marlin kernels) still varies slightly between runs. The results above were produced before the vision encoder was removed from the released weights; the removal causes tiny numerical differences, so a few sentences may differ in wording or output time when you rerun them.

## 4. Full results

Cells are **COMET / CometKiwi / latency (s) / BLEU**. COMET = wmt22-comet-da, CometKiwi = wmt22-cometkiwi-da (reference-free), BLEU = SacreBLEU (zh tokenizer for Chinese, 13a for English).

**English → Chinese**

| System | MCIF | RealSI long-form | FLEURS | CoVoST 2 | ACL 60/60 dev | ACL 60/60 eval | RealSI sentences |
|---|---|---|---|---|---|---|---|
| Babelfish-30B-A3B | 0.8680 / 0.7968 / 1.50 / 40.2 | 0.8283 / 0.7317 / 1.74 / 25.2 | 0.8833 / 0.8417 / 1.66 / 40.3 | 0.8706 / 0.8079 / 1.81 / 41.4 | 0.8566 / 0.7844 / 1.81 / 41.8 | 0.8517 / 0.7657 / 1.92 / 41.9 | 0.8294 / 0.7550 / 1.75 / 24.0 |
| Hy-SimulMT-30B-A3B | 0.8524 / 0.7686 / 2.15 / 52.3 | 0.7942 / 0.7037 / 2.66 / 29.8 | 0.8668 / 0.8189 / 2.09 / 42.6 | 0.8387 / 0.7680 / 1.96 / 38.2 | 0.8328 / 0.7435 / 1.77 / 49.9 | 0.8248 / 0.7250 / 1.74 / 48.3 | 0.7771 / 0.7064 / 1.88 / 28.5 |
| Qwen3.8-LiveTranslate | 0.8658 / 0.7852 / 4.07 / 45.1 | 0.8219 / 0.7131 / 3.69 / 27.0 | 0.8688 / 0.8133 / 2.30 / 38.2 | 0.8558 / 0.7797 / 2.66 / 38.1 | 0.8430 / 0.7531 / 2.44 / 43.1 | 0.8396 / 0.7298 / 2.51 / 43.0 | 0.8039 / 0.7211 / 2.18 / 25.3 |
| Seed LiveInterpret 2.0 | 0.8624 / 0.7748 / 2.51 / 55.6 | 0.8167 / 0.7136 / 2.78 / 33.3 | 0.8580 / 0.8036 / 2.15 / 43.1 | 0.8122 / 0.7238 / 1.86 / 39.6 | 0.8327 / 0.7371 / 2.10 / 51.9 | 0.8117 / 0.6970 / 2.02 / 48.9 | 0.7916 / 0.6970 / 2.02 / 31.3 |
| GPT-Realtime-Translate | 0.8413 / 0.7667 / 4.50 / 38.4 | 0.7750 / 0.6747 / 4.86 / 25.0 | 0.8297 / 0.7802 / 3.55 / 31.1 | ≈0.750 / 0.667 / 3.31 / 28.0 * | 0.8063 / 0.7252 / 2.87 / 33.8 | 0.8062 / 0.7091 / 2.92 / 34.6 | 0.7757 / 0.6983 / 2.92 / 22.8 |
| Gemini 3.5 Live Translate | 0.8308 / 0.7418 / 1.98 / 48.3 | 0.7629 / 0.6777 / 2.24 / 28.4 | 0.8396 / 0.7827 / 2.53 / 40.3 | 0.8038 / 0.7201 / 2.95 / 38.7 | 0.8084 / 0.7071 / 2.67 / 46.5 | 0.7892 / 0.6782 / 2.55 / 46.1 | 0.7555 / 0.6770 / 2.74 / 27.0 |

**Chinese → English**

| System | RealSI long-form | FLEURS | CoVoST 2 | RealSI sentences |
|---|---|---|---|---|
| Babelfish-30B-A3B | 0.7784 / 0.6779 / 1.97 / 21.0 | 0.8570 / 0.8357 / 2.07 / 22.7 | 0.8061 / 0.7728 / 2.11 / 23.6 | 0.7948 / 0.7349 / 2.15 / 20.3 |
| Hy-SimulMT-30B-A3B | 0.7514 / 0.6869 / 1.91 / 22.1 | 0.8518 / 0.8250 / 2.30 / 23.7 | 0.7830 / 0.7409 / 2.20 / 22.9 | 0.7592 / 0.7251 / 1.59 / 19.7 |
| Qwen3.8-LiveTranslate | 0.7752 / 0.6792 / 3.56 / 20.4 | 0.8441 / 0.8187 / 2.73 / 20.7 | 0.7897 / 0.7478 / 3.24 / 21.5 | 0.7771 / 0.7392 / 2.42 / 19.5 |
| Seed LiveInterpret 2.0 | 0.7803 / 0.6704 / 2.82 / 28.3 | 0.8450 / 0.8144 / 2.70 / 22.7 | 0.7518 / 0.6993 / 2.13 / 22.2 | 0.7775 / 0.6945 / 2.19 / 25.3 |
| GPT-Realtime-Translate | 0.7448 / 0.6509 / 4.15 / 17.2 | 0.7961 / 0.7618 / 3.56 / 16.1 | ≈0.658 / 0.598 / 3.79 / 11.8 * | 0.7330 / 0.6779 / 3.10 / 14.4 |
| Gemini 3.5 Live Translate | 0.7218 / 0.6559 / 2.06 / 16.8 | 0.8042 / 0.7579 / 2.82 / 19.4 | 0.6601 / 0.5972 / 3.05 / 14.6 | 0.7225 / 0.6723 / 2.80 / 16.8 |

Notes:

- Sentence-level sets are scored per sentence; long-form sets (MCIF, RealSI long-form) are re-segmented to the official sentences with OmniSTEval first; output is scored as is, punctuation kept; empty outputs count as empty strings for quality and are skipped for latency.
- **The two groups of latencies are measured differently**: Babelfish and Hy-SimulMT are simulated offline on the audio timeline without computation time; the commercial APIs were fed audio in real time and timed by wall clock, including network round trips and server-side computation.
- Babelfish uses beam size 2 (24 s window, 256 context tokens, `--beams 2`; without the repetition guard, which would change 1 of these 23,721 outputs); Hy-SimulMT-30B-A3B uses the official SimulInfer defaults (1.6 s chunks, VAD on).
- \* GPT-Realtime-Translate has valid results for only part of CoVoST 2: measured on those sentences, then extrapolated with the average "full set minus the same sentences" difference of the other five systems, about ±0.005 COMET; for English → Chinese about 0.54 s of latency caused by server overload was also removed.
- GPT-Realtime-Translate writes Chinese with English punctuation; it is scored as is.

## 5. Quantized versions and decoding

**FP8 vs FP16** (11 test sets, beam size 2):

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

Mean over the 11 sets: COMET −0.0009, latency 40 ms lower (both with beam size 2; FP8 runs with weight-only Marlin kernels on A800).

**Decoding** (FP16):

| Test set | beam size 2 (README results) | greedy, no guard | greedy + repetition guard (default) |
|---|---|---|---|
| FLEURS en→zh | 0.8833 / 1.66 | 0.8857 / 1.67 | 0.8857 / 1.67 (never fired) |
| FLEURS zh→en | 0.8570 / 2.07 | 0.8595 / 1.90 | 0.8595 / 1.90 (never fired) |
| RealSI long-form en→zh | 0.8283 / 1.74 | 0.7931 / — (1 recording looped until stopped) | 0.8281 / 1.90 (fired on that recording only) |
| RealSI long-form zh→en | 0.7784 / 1.97 | 0.7805 / 1.99 | 0.7781 / 1.98 (never fired) |
| MCIF | 0.8680 / 1.50 | 0.8641 / 1.53 | 0.8630 / 1.53 (never fired) |

Cells are COMET / latency (s), FP16. Where the guard never fired, the two greedy columns differ only by vLLM batching noise.

**GGUF (llama.cpp)**:

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

