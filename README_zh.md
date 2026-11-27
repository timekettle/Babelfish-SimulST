[English](README.md) | **简体中文**

<h1 align="center">Babelfish</h1>

<p align="center"><b>低延迟、高质量的英中双向流式同声传译模型</b></p>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Babelfish-ffc107)](https://huggingface.co/timekettle/Babelfish-SimulMT)
[![GitHub](https://img.shields.io/badge/GitHub-Babelfish-181717?logo=github)](https://github.com/timekettle/Babelfish-SimulMT)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

</div>

<!-- 演示视频原文件备份在 assets/demo_en2zh.mp4 和 assets/demo_zh2en.mp4 -->
<div align="center">
<table>
  <tr>
    <td align="center"><b>英文 → 中文</b></td>
    <td align="center"><b>中文 → 英文</b></td>
  </tr>
  <tr>
    <td align="center"><video src="https://github.com/user-attachments/assets/166bb4c8-92e4-46b6-a2a8-ae51116bcb96" width="380" controls></video></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/ae1ecb7b-96f1-4a4c-b120-3bff67feb0d6" width="380" controls></video></td>
  </tr>
</table>
<sub>演示中的译文由本仓库开源的 Babelfish-30B-A3B 实时生成（束宽 2）。<br>界面上的原文转写与译文语音（克隆原说话人音色）由另接的语音识别和语音合成模型生成，不在本次开源范围内。<br>两段演示的译文延迟：英→中 StreamLAAL 1.20 秒（含计算 1.48 秒，首字 1.33 秒），中→英 StreamLAAL 1.57 秒（含计算 1.81 秒，首词 2.22 秒）；首音延迟分别为 2.10 秒与 4.47 秒。首音延迟较长，是因为外接音色克隆语音合成需先采集足够的原说话人语音以提取音色，并等待首个完整语义短句译出后再合成播报（中→英开场从句较长，因此首音等待更久），以保证音色还原与韵律自然。</sub>
</div>

## 模型介绍

**Babelfish-30B-A3B** 是一个语音到文字的同声传译模型，支持**英文 → 中文**和**中文 → 英文**两个方向：说话人的声音持续输入，模型边听边译，已经输出的译文不再修改。模型由 Qwen3-Omni-30B-A3B-Instruct 流式微调得到（MoE 结构，每个 token 约激活 3B 参数），方法与训练细节见技术报告（链接待补）。

本仓库开源：

- 模型权重：FP16 原版、FP8 量化版（vLLM）、GGUF 量化版（llama.cpp）；
- 推理与部署代码：网页（麦克风 / 上传文件）、实时 WebSocket 接口、文件上传接口、命令行客户端、一键部署脚本；
- 离线批量推理与评测脚本。

训练代码和训练数据构造暂不开源。

## 新闻

- **2026.11.XX｜发布**：Babelfish-30B-A3B 及其 FP8、GGUF 量化版发布，推理与部署代码同步开源；技术报告（链接待补）。

## 效果

我们在 Hy-SimulMT 公布的全部测试集（FLEURS、CoVoST 2、ACL 60/60、RealSI 单句、RealSI 长音频）加上 MCIF 长音频，共 11 个测试集上，对比了开源的 Hy-SimulMT 和四家商业同传服务。

- **质量**：COMET 在 11 个测试集中有 10 个最高，CometKiwi 有 9 个最高；平均 COMET 英→中 85.5、中→英 80.9，次高的 Qwen3.8-LiveTranslate 为 84.3 和 79.7。唯一没有拿到最高 COMET 的是中→英 RealSI 长音频，比 Seed LiveInterpret 2.0 低 0.2。
- **延迟**：LAAL 英→中 1.50–1.92 秒、中→英 1.97–2.15 秒，平均 1.74 秒和 2.08 秒。

COMET 和 CometKiwi 为百分制（0–100）；每个指标每列最好的值加粗（延迟取最低）；† 为商业同传服务，延迟测法不同，见表下说明。

**英文 → 中文**

| 系统 | MCIF | RealSI 长音频 | RealSI 单句 | FLEURS | CoVoST 2 | ACL 60/60 dev | ACL 60/60 eval |
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
| *LAAL（秒）↓* | | | | | | | |
| **Babelfish-30B-A3B** | **1.50** | **1.74** | **1.75** | **1.66** | **1.81** | 1.81 | 1.92 |
| Hy-SimulMT-30B-A3B | 2.15 | 2.66 | 1.88 | 2.09 | 1.96 | **1.77** | **1.74** |
| Qwen3.8-LiveTranslate † | 4.07 | 3.69 | 2.18 | 2.30 | 2.66 | 2.44 | 2.51 |
| Seed LiveInterpret 2.0 † | 2.51 | 2.78 | 2.02 | 2.15 | 1.86 | 2.10 | 2.02 |
| GPT-Realtime-Translate † | 4.50 | 4.86 | 2.92 | 3.55 | 3.31\* | 2.87 | 2.92 |
| Gemini 3.5 Live Translate † | 1.98 | 2.24 | 2.74 | 2.53 | 2.95 | 2.67 | 2.55 |

**中文 → 英文**

| 系统 | RealSI 长音频 | RealSI 单句 | FLEURS | CoVoST 2 |
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
| *LAAL（秒）↓* | | | | |
| **Babelfish-30B-A3B** | 1.97 | 2.15 | **2.07** | **2.11** |
| Hy-SimulMT-30B-A3B | **1.91** | **1.59** | 2.30 | 2.20 |
| Qwen3.8-LiveTranslate † | 3.56 | 2.42 | 2.73 | 3.24 |
| Seed LiveInterpret 2.0 † | 2.82 | 2.19 | 2.70 | 2.13 |
| GPT-Realtime-Translate † | 4.15 | 3.10 | 3.56 | 3.79\* |
| Gemini 3.5 Live Translate † | 2.06 | 2.80 | 2.82 | 3.05 |

<details>
<summary>测试条件与指标口径</summary>

- **COMET**：wmt22-comet-da；**CometKiwi**：wmt22-cometkiwi-da（不看参考译文）；**BLEU**：SacreBLEU（英→中 zh 分词，中→英 13a）。译文保留标点原样打分；单句集逐句打分，长音频（MCIF、RealSI 长音频）先用 OmniSTEval 按官方句子重新切分再打分。
- **延迟**：单句集为 LAAL，长音频为重新切分后计算的 StreamLAAL；英→中按字、中→英按词计算。
- **两组延迟测法不同，不能直接比较**：Babelfish 和 Hy-SimulMT 按音频时间离线模拟，不含计算耗时；四家商业服务按实时速度送入音频，记录收到译文时的墙钟时间，包含网络往返和服务端计算。
- Babelfish 的成绩用束宽 2 解码（`--beams 2`）；发布版默认改为贪心解码 + 复读保护，质量相当（FLEURS 略高、MCIF 低 0.5），见[部署教程 3.3](docs/DEPLOYMENT_zh.md#33-解码方式贪心还是束宽-2)。Hy-SimulMT-30B-A3B 使用其 SimulInfer 官方默认设置（1.6 秒一块，开启 VAD）。
- 带 * 的格子为近似值：GPT-Realtime-Translate 在 CoVoST 2 上只有部分句子有有效结果，按其余系统在同一批句子上的差值推算到全集（COMET 误差约 ±0.5）。
- 测试集规模：MCIF 21 段、RealSI 长音频每个方向 10 段、FLEURS 646 / 945 句、CoVoST 2 15,530 / 4,898 句、ACL 60/60 dev 468 句、eval 416 句、RealSI 单句 346 / 431 句。
- 0–1 分制的原始分数、量化版和不同解码方式的对比见 [docs/EVALUATION_zh.md](docs/EVALUATION_zh.md)。

</details>

## 模型链接

| 模型名 | 简介 | 推理框架 | 下载 |
|---|---|---|---|
| Babelfish-30B-A3B | FP16 原版 | vLLM | [Hugging Face](https://huggingface.co/timekettle/Babelfish-SimulMT/tree/main/Babelfish-30B-A3B) |
| Babelfish-30B-A3B-FP8 | FP8 量化（逐 128×128 块缩放） | vLLM | [Hugging Face](https://huggingface.co/timekettle/Babelfish-SimulMT/tree/main/Babelfish-30B-A3B-FP8) |
| Babelfish-30B-A3B-GGUF | Q8_0 / Q4_K_M + 音频编码器 | llama.cpp | [Hugging Face](https://huggingface.co/timekettle/Babelfish-SimulMT/tree/main/Babelfish-30B-A3B-GGUF) |

| 版本 | 权重大小 | 显存 | 质量 | 适合 |
|---|---|---|---|---|
| FP16 | 58 GiB | 80 GB × 1，或 48 GB × 2 | 发布成绩所用版本 | 服务器部署 |
| FP8 | 30 GiB | 48 GB × 1 | 11 个测试集 COMET 平均低 0.09（百分制） | 单张 48 GB 卡；H100 / H20 / L40S 原生 FP8 |
| GGUF Q8_0 / Q4_K_M | 30.3 / 17.3 GiB + 1.2 GiB 音频编码器 | 约 36 / 23 GB | 打上 llama.cpp 音频补丁后与 vLLM 持平 | llama.cpp；没有 vLLM 的环境、离线转写 |

## 部署推理服务

> 详细说明（环境、显存、多卡、HTTPS、接口协议、常见问题）见 **[部署教程](docs/DEPLOYMENT_zh.md)**。

### 环境要求

- Linux，Conda（Python 3.10 及以上），NVIDIA GPU：FP16 版需要 1 张 80 GB 显卡（或 2 张 48 GB 做张量并行），FP8 版需要 48 GB 及以上显存；
- vLLM（已测试 0.28.0 和 0.23 nightly，配 transformers 5.13.1）；
- llama.cpp 部署 GGUF 版的要求见下文。

### 环境配置与部署（vLLM）

先创建并激活 conda 环境，再安装依赖：

```bash
git clone https://github.com/timekettle/Babelfish-SimulMT.git
cd Babelfish-SimulMT

conda create -n babelfish python=3.12 -y
conda activate babelfish

# 驱动支持 CUDA 13（>= 580）
pip install vllm==0.28.0
# 显卡驱动只支持到 CUDA 12.x（例如 535 / 550）时，改用下面这行：
# pip install "vllm==0.28.0+cu129" "torch==2.13.0+cu129" \
#   --extra-index-url https://download.pytorch.org/whl/cu129 \
#   --extra-index-url https://wheels.vllm.ai/0.28.0/cu129

pip install -e .
```

激活 `babelfish` 环境后启动服务（传入模型 ID 会自动下载模型；看到 `Ready: http://127.0.0.1:8080/` 后即可使用，按 Ctrl+C 停止）：

```bash
# 从 Hugging Face 下载并启动（FP16 原版）
bash scripts/deploy.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B
# FP8 版
bash scripts/deploy.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B-FP8
# 本地权重，指定单卡或两卡张量并行
bash scripts/deploy.sh --model /path/to/Babelfish-30B-A3B --gpus 0
bash scripts/deploy.sh --model /path/to/Babelfish-30B-A3B --gpus 0,1
```

也可以给脚本加 `--install`，自动创建并配置名为 `babelfish` 的 conda 环境（驱动只到 CUDA 12.x 时加 `--cu129`）：

```bash
bash scripts/deploy.sh --install --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B
```

或不用脚本，在 `babelfish` conda 环境里直接启动服务：

```bash
python -m simulst.server --model /path/to/Babelfish-30B-A3B --port 8080
```

### 使用

- **网页**：浏览器打开 `http://服务器地址:8080/`，选择方向，点“开始麦克风”说话，或选择音频文件（可勾选“按实时速度播放”边播边译）。远程使用麦克风需要 HTTPS 或 localhost，最简单的做法是 SSH 端口转发：`ssh -N -L 8080:127.0.0.1:8080 user@server`，再打开 `http://127.0.0.1:8080/`。
- **命令行客户端**：

  ```bash
  python -m simulst.client audio.wav --direction en2zh --url http://127.0.0.1:8080             # 尽快解完
  python -m simulst.client audio.wav --direction zh2en --url http://127.0.0.1:8080 --realtime  # 按实时速度推流
  ```

- **上传文件（SSE 流式返回）**：

  ```bash
  curl -N http://127.0.0.1:8080/translate/stream -F direction=en2zh -F file=@audio.wav
  ```

- **实时 WebSocket**：连接 `ws://服务器地址:8080/translate/ws`，先发 `{"direction":"en2zh","sample_rate":16000,"format":"s16le","channels":1}`，收到 `ready` 后发送 PCM 二进制音频，结束时发 `{"eof":true}`，一直接收到 `done`。协议见 [部署教程 · 接口](docs/DEPLOYMENT_zh.md#接口)。

### llama.cpp（GGUF）

GGUF 版用 llama.cpp 运行（Q8_0 30.3 GiB / Q4_K_M 17.3 GiB，外加 1.2 GiB 的音频编码器），推理逻辑和复读保护与 vLLM 版相同。

**要达到 vLLM 版的质量，需要给 llama.cpp 打一个音频补丁**（[patches/llama.cpp](patches/llama.cpp)）：llama.cpp 会把最后不满 1 秒的音频补零后照常编码，模型以为听到了更多音频，于是过早开口。打补丁后 Q8_0 与 vLLM 持平（FLEURS 英→中 COMET 88.8 vs 88.9）；不打也能运行，但 COMET 低约 1.4、出字早约 0.65 s。

```bash
# 编译打过补丁的 llama.cpp
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 3f8af15a76ac8675812e856da45454bafc90f12c
git apply ../patches/llama.cpp/qwen3-omni-audio-valid-tokens.patch      # 在 Babelfish-SimulMT 仓库目录下 clone 时的相对路径
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server && cd ..
# 一键启动 llama-server 和网页 / 接口服务（接口与 vLLM 版相同）
bash scripts/deploy_llamacpp.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B-GGUF --quant Q4_K_M --llama-server llama.cpp/build/bin/llama-server
```

默认贪心解码 + 复读保护。llama.cpp 上也能用束搜索（`--beams 2`），但实测质量没有比贪心更好、每步耗时翻倍，所以不作为默认。llama.cpp 不会把多路音频的预填拼成一批，A800 上 24 s 窗口每步约 0.5 s（超过 0.32 s 的节拍），长音频的实时会话会合并音频块、延迟变大，更适合离线转写、短句或没有 vLLM 的环境。详见[部署教程 · llama.cpp](docs/DEPLOYMENT_zh.md#7-llamacppgguf-版)。

### 离线批量推理与评测

一批音频文件按流式方式译完（每条输出译文和每个字的输出时刻），清单格式见[部署教程 · 离线批量推理](docs/DEPLOYMENT_zh.md#8-离线批量推理)：

```bash
python -m simulst.infer.stream --model /path/to/Babelfish-30B-A3B --manifests talks=talks.jsonl --out outputs/
```

复现上文的 COMET、BLEU 和延迟：解码时用束宽 2，再打分：

```bash
python -m simulst.infer.stream --model /path/to/Babelfish-30B-A3B --beams 2 --manifests fleurs=fleurs_en2zh.jsonl --out outputs/
python -m simulst.eval.score_sentences outputs/fleurs
```

测试集准备、长音频打分和完整结果见 [docs/EVALUATION_zh.md](docs/EVALUATION_zh.md)。

## 许可证

代码与模型权重以 [Apache License 2.0](LICENSE) 发布。模型由 Qwen3-Omni-30B-A3B-Instruct（Apache License 2.0）微调得到，见 [NOTICE](NOTICE)。

## 贡献者

邓士洋，李男杰，郭晓勇，徐海华，石伟，黄浩，邓贤苏，郭俊，罗元明，吴王政

## 引用

如果 Babelfish 对你有帮助，欢迎引用：

```bibtex
@misc{babelfish2026,
  title  = {Babelfish: Low-Latency Streaming Simultaneous Speech Translation between English and Chinese},
  author = {Shiyang Deng and Nanjie Li and Xiaoyong Guo and Haihua Xu and Wei Shi and Hao Huang and Xiansu Deng and Jun Guo and Yuanming Luo and Wangzheng Wu},
  year   = {2026},
  url    = {https://github.com/timekettle/Babelfish-SimulMT}
}
```

## 联系我们

欢迎通过 GitHub Issues 反馈问题和建议。
