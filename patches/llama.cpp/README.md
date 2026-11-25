# llama.cpp audio patch

`qwen3-omni-audio-valid-tokens.patch` fixes how llama.cpp encodes the end of a Qwen3-Omni audio input
(`tools/mtmd`, 5 files, about 30 lines). It applies to llama.cpp commit
`3f8af15a76ac8675812e856da45454bafc90f12c` (2026-11-02); it also applies cleanly to master as of 2026-11-03 (`ca8daad`).

**Why.** The audio encoder works on chunks of 100 mel frames (1 s), 13 tokens per chunk. The reference implementation
(Hugging Face / vLLM) gives a partial last chunk only the tokens that its real frames produce. llama.cpp pads the last
chunk to a full second and always produces 13 tokens, so every streaming step sees 0–12 extra tokens of padding at the
end of the audio. The model takes them as more audio than it has received and starts to speak too early: without the
patch, outputs come about 650 ms earlier and COMET is about 0.014 lower (English → Chinese FLEURS); with the patch,
the GGUF model matches vLLM (see docs/DEPLOYMENT.md, section 7).

**How.**

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 3f8af15a76ac8675812e856da45454bafc90f12c
git apply /path/to/Babelfish-SimulMT/patches/llama.cpp/qwen3-omni-audio-valid-tokens.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```

The unpatched llama.cpp also runs the GGUF files, with the lower quality described above.

The patch modifies llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors).
