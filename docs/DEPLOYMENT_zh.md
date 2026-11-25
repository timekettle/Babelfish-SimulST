[English](DEPLOYMENT.md) | **简体中文** | [返回首页](../README_zh.md)

# 部署教程

本文介绍如何部署 Babelfish 推理服务：选择模型版本、安装环境、启动服务、访问网页与接口，以及 FP8、GGUF 量化版和常见问题。

- [1. 选择版本与硬件](#1-选择版本与硬件)
- [2. 安装](#2-安装)
- [3. 启动服务](#3-启动服务)
- [4. 访问网页](#4-访问网页)
- [5. 接口](#接口)
- [6. FP8 版](#6-fp8-版)
- [7. llama.cpp（GGUF 版）](#7-llamacppgguf-版)
- [8. 离线批量推理](#8-离线批量推理)
- [9. 常见问题](#9-常见问题)

## 1. 选择版本与硬件

| 版本 | 权重 | 显存 | 已测试 | 推理框架 | 默认解码 |
|---|---|---|---|---|---|
| Babelfish-30B-A3B（FP16） | 58 GiB（13 个分片） | 80 GB × 1；或 48 GB × 2 张量并行 | A800 80GB 单卡；两卡张量并行、每卡限 46 GB | vLLM | 贪心 |
| Babelfish-30B-A3B-FP8 | 30 GiB（7 个分片） | 48 GB × 1 | A800（Marlin，只量化权重），显存限在 44 GB 内 | vLLM | 贪心 |
| Babelfish-30B-A3B-GGUF Q8_0 | 30.3 GiB + 音频编码器 1.2 GiB | 约 36 GB（8 个槽）/ 46 GB（32 个槽） | A800（CUDA） | llama.cpp | 贪心 |
| Babelfish-30B-A3B-GGUF Q4_K_M | 17.3 GiB + 音频编码器 1.2 GiB | 约 23 GB（8 个槽）/ 32 GB（32 个槽） | A800（CUDA） | llama.cpp | 贪心 |

所有版本默认贪心解码 + 复读保护，都可以用 `--beams 2` 改成束搜索。

服务是**有状态的流式会话**：每个会话每 320 ms 调用一次模型，多个会话的请求由 vLLM 自动拼批。单步耗时要明显小于 320 ms，实时会话才不会积压；积压时服务会把多块音频合并成一步，延迟不会越来越大，但会略微影响质量。

单张 A800 80GB 实测：每路按实时速度推送 60 s 音频（RealSI 录音片段，英→中、中→英各半）。“合并块”是跟不上时被合并掉的 320 ms 块所占比例；“译文到达比音频晚”是收到一段译文的时刻减去这段译文对应的音频时刻，也就是部署时在离线延迟之外多出来的部分。

| 版本 | 解码 | 并发路数 | 每步耗时 p50 / p90 | 合并块 | 译文到达比音频晚 p50 / p90 |
|---|---|---|---|---|---|
| FP16 | 两束 | 1 | 143 / 225 ms | 0 | 0.25 / 0.39 s |
| FP16 | 两束 | 2 | 224 / 312 ms | 0 | 0.32 / 0.50 s |
| FP16 | 两束 | 4 | 248 / 492 ms | 7.3% | 0.37 / 0.72 s |
| FP16 | 贪心 + 复读保护 | 1 | 111 / 149 ms | 0 | 0.18 / 0.24 s |
| FP16 | 贪心 + 复读保护 | 4 | 195 / 247 ms | 0.7% | 0.29 / 0.37 s |
| FP16 | 贪心 + 复读保护 | 8 | 280 / 484 ms | 9.8% | 0.56 / 0.90 s |
| FP8 | 两束 | 1 | 132 / 185 ms | 0 | 0.20 / 0.27 s |
| FP8 | 两束 | 2 | 211 / 282 ms | 0 | 0.28 / 0.38 s |
| FP8 | 两束 | 4 | 242 / 467 ms | 6.5% | 0.35 / 0.72 s |

建议：一张 A800 上束宽 2 带 2 路左右、贪心 + 复读保护带 4 路左右不会积压；更多并发请多开实例。

## 2. 安装

### 2.1 vLLM 环境

需要 Linux、Conda（Python 3.10 及以上）和 NVIDIA 驱动。已测试的组合：vLLM 0.28.0（torch 2.13）和 vLLM 0.23 nightly（torch 2.11），都配 transformers 5.13.1。

```bash
conda create -n babelfish python=3.12 -y
conda activate babelfish
pip install -U pip

# 驱动支持 CUDA 13（>= 580）：直接安装
pip install vllm==0.28.0

# 驱动只到 CUDA 12.x（例如 535 / 550）：安装 CUDA 12.9 构建
pip install "vllm==0.28.0+cu129" "torch==2.13.0+cu129" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://wheels.vllm.ai/0.28.0/cu129

# 本项目（服务、客户端、离线推理）
git clone https://github.com/timekettle/Babelfish-SimulMT.git && cd Babelfish-SimulMT
pip install -e .
```

国内网络可以给 pip 加镜像，例如 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（PyTorch 和 vLLM 的 CUDA 构建仍从上面两个地址下载）。

### 2.2 下载模型

```bash
# Hugging Face（仅下载 FP16 子目录到 models/Babelfish-30B-A3B）
hf download timekettle/Babelfish-SimulMT --include "Babelfish-30B-A3B/*" --local-dir models
```

FP8 版把名字换成 `Babelfish-30B-A3B-FP8`。GGUF 版见第 7 节。

## 3. 启动服务

### 3.1 一键部署脚本

`scripts/deploy.sh` 依次完成：（可选）创建 / 使用 conda 环境并安装依赖 → 下载模型（如果给的是模型 ID）→ 前台启动服务。看到 `Ready: http://127.0.0.1:8080/` 后即可访问，Ctrl+C 停止。自动下载的模型和日志保存在 `.simulst/`。

```bash
# 已激活 babelfish conda 环境时直接启动
conda activate babelfish
bash scripts/deploy.sh --model models/Babelfish-30B-A3B --gpus 0
bash scripts/deploy.sh --model models/Babelfish-30B-A3B --gpus 0,1                                # 两卡张量并行

# 也可加 --install 自动创建名为 babelfish 的 conda 环境并安装依赖
bash scripts/deploy.sh --install --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B           # 驱动支持 CUDA 13
bash scripts/deploy.sh --install --cu129 --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B   # 驱动只到 CUDA 12.x
bash scripts/deploy.sh --install --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B-FP8       # FP8 版
bash scripts/deploy.sh --model models/Babelfish-30B-A3B --dry-run                                 # 只打印将执行的命令
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--model` | Hugging Face 模型 ID（`仓库` 或 `仓库/子目录`），或本地目录 | 必填 |
| `--install` | 创建 / 使用 conda 环境并安装 vLLM 和本项目 | 不安装 |
| `--conda-env` | `--install` 使用的 conda 环境名（已激活非 base 环境时默认复用当前环境） | `babelfish` |
| `--cu129` | 安装 CUDA 12.9 构建的 vLLM / torch（驱动只到 CUDA 12.x 时用） | 否 |
| `--python` | 显式指定已有环境的 Python（默认使用当前激活的 conda 环境 `python3`） | `python3` |
| `--gpus` | 使用的 GPU，例如 `0` 或 `0,1`；张量并行数默认等于卡数 | 全部可见 |
| `--tp` | 张量并行数 | 卡数 |
| `--host` / `--port` | 监听地址 / 端口 | `0.0.0.0` / `8080` |
| `--beams` | 默认束宽：1 为贪心（默认，更快），2 为束搜索 | `1` |
| `--max-sessions` | 最大并发会话数 | `4` |
| `--gpu-mem` | vLLM 显存占比 | `0.9` |
| `--runtime-dir` | 自动下载的模型和日志目录 | `.simulst` |
| `--` 之后 | 原样传给 `simulst.server` | |

### 3.2 直接启动

```bash
python -m simulst.server --model models/Babelfish-30B-A3B --host 0.0.0.0 --port 8080
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--model` | 模型目录或 Hugging Face ID | 必填（vLLM） |
| `--beams` | 默认束宽（1 = 贪心；客户端可按会话覆盖，1–4） | `1` |
| `--max-sessions` | 最大并发会话数，超出时新会话收到“服务繁忙” | `4` |
| `--tp` | 张量并行数 | `1` |
| `--gpu-mem` | vLLM 显存占比 | `0.9` |
| `--max-seqs` | vLLM 同时处理的序列数上限 | `64` |
| `--dtype` | 计算精度；模型以 FP16 发布和评测，请保持 `float16` | `float16` |
| `--window` / `--ctx-tokens` | 音频窗口（秒）/ 前文 token 数，保持默认即与评测一致 | `24` / `256` |
| `--loop-guard` | 复读保护（只在失控复读时截断，见首页“推理方式”） | `on` |
| `--eager` | 关闭 CUDA Graph（省显存、启动快，单步更慢） | 否 |
| `--ssl-certfile` / `--ssl-keyfile` | 直接提供 HTTPS | 无 |
| `--backend llamacpp` | 改用 llama.cpp 后端（见第 7 节） | `vllm` |

### 3.3 解码方式：贪心还是束宽 2

默认贪心解码 + 复读保护：比束宽 2 每步快约 25%，同一张卡能多带一倍会话，质量相当。首页成绩表用的是束宽 2（`--beams 2`，或客户端配置里传 `"beams": 2`）。贪心更容易复读，复读保护会在提交前截掉。FP16 实测：

| 测试集 | 束宽 2（首页成绩） | 贪心，不加保护 | 贪心 + 复读保护（默认） |
|---|---|---|---|
| FLEURS 英→中 | 0.8833 / 1.66 | 0.8857 / 1.67 | 0.8857 / 1.67（0 次触发） |
| FLEURS 中→英 | 0.8570 / 2.07 | 0.8595 / 1.90 | 0.8595 / 1.90（0 次触发） |
| RealSI 长音频 英→中 | 0.8283 / 1.74 | 0.7931 / —（1 段复读到中止） | 0.8281 / 1.90（只在那 1 段触发） |
| RealSI 长音频 中→英 | 0.7784 / 1.97 | 0.7805 / 1.99 | 0.7781 / 1.98（0 次触发） |
| MCIF | 0.8680 / 1.50 | 0.8641 / 1.53 | 0.8630 / 1.53（0 次触发） |

格子为 COMET / 延迟（秒），FP16。没有触发的集合，两列贪心的差异来自 vLLM 拼批的数值波动。

### 3.4 多卡与多实例

- FP16 版放不进单张 48 GB 显卡时，用两张卡张量并行：`--gpus 0,1`（或 `--tp 2`）。
- 要服务更多并发，可以每张卡（或每组卡）起一个实例，用不同端口，前面加负载均衡。一个 WebSocket 连接始终落在同一个实例上，不需要额外的会话保持。

## 4. 访问网页

浏览器打开 `http://服务器地址:8080/`：

1. 选择方向（英文 → 中文 / 中文 → 英文）；
2. 点“开始麦克风”说话，说完点“结束并收尾”；或者选择音频 / 视频文件后点“翻译文件”：勾选“按实时速度播放”时边播边译，不勾选时尽快解完；
3. 页面下方显示已处理的音频时长和每步耗时。

浏览器只允许在 **HTTPS 或 localhost** 下使用麦克风。远程服务器有三种做法：

```bash
# 1. SSH 端口转发（最简单），然后在本机打开 http://127.0.0.1:8080/
ssh -N -L 8080:127.0.0.1:8080 user@server
# 2. 服务直接提供 HTTPS
python -m simulst.server --model models/Babelfish-30B-A3B --ssl-certfile cert.pem --ssl-keyfile key.pem
```

3. 用 Nginx 等反向代理提供 HTTPS，注意转发 WebSocket：

```nginx
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
    proxy_buffering off;          # 让 SSE 逐条到达
}
```

页面用相对路径访问接口，挂在子路径下（例如 `https://host/proxy/8080/`）也能用。

## 接口

服务启动后，交互式接口文档在 `/docs`。

### 5.1 实时 WebSocket：`/translate/ws`

1. 连接后先发一条 JSON 配置：

   | 字段 | 说明 | 默认 |
   |---|---|---|
   | `direction` | `en2zh` 或 `zh2en`（也接受 `en-zh`、`zh-en`） | `en2zh` |
   | `sample_rate` | 音频采样率，服务端会重采样到 16 kHz | `16000` |
   | `format` | `s16le`（16 位整数）或 `f32le`（32 位浮点），小端 | `s16le` |
   | `channels` | 声道数，多声道会取平均 | `1` |
   | `realtime` | `true`：按实时节奏处理，跟不上时合并积压的块；`false`：每块都解码（推流快于实时时用，结果与离线评测一致） | `true` |
   | `beams` | 本会话的束宽（1–4） | 服务端默认 |

2. 收到 `{"event":"ready", ...}` 后，用二进制消息发送音频（大小不限，建议每 100 ms 左右一条）。
3. 音频发完后发 `{"eof": true}`，继续接收直到 `done`。

服务端发出的消息：

| `event` | 字段 | 含义 |
|---|---|---|
| `ready` | `session`、`direction`、`beams`、`model` | 会话已建立 |
| `translation` | `delta`（新增文字）、`text`（目前的完整译文）、`audio_s`（这段文字输出时已收到的音频秒数）、`elapsed_s`、`step_ms` | 有新译文时发送 |
| `done` | `text`、`audio_s`、`elapsed_s`、`steps`、`skipped_blocks`、`loop_guards`（复读保护截断次数）、`step_ms_p50`、`step_ms_p90`、`emissions` | 全部完成 |
| `error` | `message` | 出错（例如服务繁忙），随后关闭连接 |

已经输出的文字不会修改，所以客户端直接把 `delta` 接在后面即可；个别情况下（被拆成两个 token 的汉字）请以 `text` 为准。

Python 示例（`pip install websockets numpy soundfile`）：

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
                await asyncio.sleep(0.1)             # 实时节奏
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

仓库自带的命令行客户端功能更全（尽快 / 实时推流、保存全部事件）：

```bash
python -m simulst.client audio.wav --direction en2zh --url http://127.0.0.1:8080 [--realtime] [--save events.jsonl]
```

### 5.2 上传文件，流式返回：`POST /translate/stream`

multipart 表单：`file`（wav / flac / ogg 等 libsndfile 能读的格式）、`direction`、可选 `realtime`（`true` 时按实时速度处理）、`beams`。返回 `text/event-stream`，每条 `data:` 是一个与 WebSocket 相同的事件。

```bash
curl -N http://127.0.0.1:8080/translate/stream -F direction=zh2en -F file=@audio.wav
```

### 5.3 上传文件，一次返回：`POST /translate`

参数同上，返回最后的 `done` 事件（含完整译文和每次输出的时间点）。

```bash
curl http://127.0.0.1:8080/translate -F direction=en2zh -F file=@audio.wav
```

### 5.4 状态：`GET /health`

返回模型名、后端、当前会话数、默认束宽、窗口设置等。

## 6. FP8 版

`Babelfish-30B-A3B-FP8` 把语言模型的注意力投影和全部专家的权重存成 FP8（E4M3，每 128×128 块一个缩放系数），激活在运行时动态量化；路由器、词表和音频编码器保持 FP16。权重从 58 GiB 降到 30 GiB。

- Hopper / Ada（H100、H20、L40S 等）上 vLLM 直接用 FP8 计算；
- Ampere（A100、A800 等）没有 FP8 计算单元，vLLM 自动改用 Marlin 内核（只量化权重，计算仍为 FP16），省显存但不提速。

使用方法与原版完全相同，只是模型换成 FP8 目录：

```bash
python -m simulst.server --model models/Babelfish-30B-A3B-FP8 --port 8080
```

FP8 权重从 FP16 版直接转换得到（每 128×128 块一个缩放系数，不需要校准数据）。质量对比：

| 测试集 | FP16：COMET / 延迟（秒） | FP8：COMET / 延迟（秒） | ΔCOMET |
|---|---|---|---|
| MCIF | 0.8680 / 1.50 | 0.8684 / 1.48 | +0.0004 |
| RealSI 长音频 英→中 | 0.8283 / 1.74 | 0.8264 / 1.70 | -0.0019 |
| FLEURS 英→中 | 0.8833 / 1.66 | 0.8811 / 1.62 | -0.0022 |
| CoVoST 2 英→中 | 0.8706 / 1.81 | 0.8702 / 1.78 | -0.0004 |
| ACL 60/60 dev | 0.8566 / 1.81 | 0.8555 / 1.77 | -0.0011 |
| ACL 60/60 eval | 0.8517 / 1.92 | 0.8504 / 1.84 | -0.0013 |
| RealSI 单句 英→中 | 0.8294 / 1.75 | 0.8302 / 1.73 | +0.0008 |
| RealSI 长音频 中→英 | 0.7784 / 1.97 | 0.7749 / 1.92 | -0.0035 |
| FLEURS 中→英 | 0.8570 / 2.07 | 0.8566 / 2.03 | -0.0004 |
| CoVoST 2 中→英 | 0.8061 / 2.11 | 0.8059 / 2.08 | -0.0002 |
| RealSI 单句 中→英 | 0.7948 / 2.15 | 0.7947 / 2.11 | -0.0001 |

11 个测试集 COMET 平均 −0.0009，延迟平均快 40 ms（两者都用束宽 2，与首页成绩同设置；FP8 在 A800 上经 Marlin 只量化权重运行）。

## 7. llama.cpp（GGUF 版）

GGUF 版用 llama.cpp 运行：`llama-server` 加载语言模型和音频编码器，本项目的服务（`--backend llamacpp`）在它之上提供同样的网页和接口，推理逻辑、复读保护都和 vLLM 版一样。

**文件**（`Babelfish-30B-A3B-GGUF`）：`Babelfish-30B-A3B-Q8_0.gguf`（30.3 GiB）、`Babelfish-30B-A3B-Q4_K_M.gguf`（17.3 GiB）、`mmproj-Babelfish-30B-A3B-F16.gguf`（音频编码器，1.2 GiB），校验值见仓库里的 `SHA256SUMS`。

### 7.1 编译 llama.cpp（带音频补丁）

llama.cpp 处理 Qwen3-Omni 音频时，会把最后不满 1 秒的音频补零到整秒照常编码，每一步音频末尾多出 0–12 个补零 token，模型以为听到了更多音频而过早开口。[patches/llama.cpp](../patches/llama.cpp) 里的补丁让最后一块只保留有效 token（与 vLLM / Hugging Face 一致）。不打补丁也能运行，但质量和延迟都会偏离（见 7.3）。

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 3f8af15a76ac8675812e856da45454bafc90f12c        # 补丁基于这个提交（2026-11-02）
git apply /path/to/Babelfish-SimulMT/patches/llama.cpp/qwen3-omni-audio-valid-tokens.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```

- 用 CUDA 12.8 及以上编译、而驱动低于 r570 时，加 `-DGGML_CUDA_COMPRESSION_MODE=none`，否则运行时报 `device kernel image is invalid`。
- 编译时如果连不上外网，加 `-DLLAMA_BUILD_HTML=OFF` 跳过下载网页界面。
- 不要让两个 llama-server 共用一张卡：实测共用时偶发音频编码器 `illegal memory access` 崩溃，一卡一个服务时没有出现。

### 7.2 启动

一键脚本（先启动 llama-server，就绪后启动网页 / 接口服务；Ctrl+C 一起停止）：

```bash
bash scripts/deploy_llamacpp.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B-GGUF --quant Q4_K_M \
  --llama-server /path/to/llama.cpp/build/bin/llama-server --gpus 0 [--sessions 4] [--beams 1]
```

也可以分两步手动启动：

```bash
llama-server -m Babelfish-30B-A3B-Q4_K_M.gguf --mmproj mmproj-Babelfish-30B-A3B-F16.gguf -ngl 999 \
  --parallel 8 -c 32768 --host 127.0.0.1 --port 8081
python -m simulst.server --backend llamacpp --llama-url http://127.0.0.1:8081 --port 8080 --max-sessions 8
```

- `--parallel`（槽数）要不少于 并发会话数 × 束宽；`-c` 取 4096 × 槽数。显存：Q8_0 8 个槽约 36 GB、32 个槽约 46 GB；Q4_K_M 8 个槽约 23 GB、32 个槽约 32 GB。
- 服务端从 llama-server 报告的模型路径读取 GGUF 里的词表；llama-server 在另一台机器上时，用 `--tokenizer` 指向 FP16 版目录（只用到其中的分词器文件）。

### 7.3 质量与速度

| 系统 | FLEURS 英→中 | FLEURS 中→英 | RealSI 长音频 英→中（1 段） |
|---|---|---|---|
| vLLM FP16 贪心 | 0.8886 / 1.67 | 0.8497 / 2.18 | 0.8341 / 1.95 |
| vLLM FP16 束宽 2 | 0.8877 / 1.62 | 0.8464 / 2.30 | 0.8451 / 1.98 |
| **GGUF Q8_0 贪心（打补丁）** | **0.8876 / 1.65** | **0.8483 / 2.16** | **0.8295 / 1.89** |
| GGUF Q8_0 贪心（官方 llama.cpp） | 0.8740 / 0.99 | 0.8480 / 1.63 | 0.8172 / 1.50 |
| GGUF Q8_0 束宽 2（官方） | 0.8742 / 0.97 | 0.8352 / 1.78 | 0.8136 / 1.57 |
| GGUF Q4_K_M 贪心（官方） | 0.8746 / 1.11 | 0.8459 / 1.75 | 0.8212 / 1.52 |
| GGUF Q4_K_M 束宽 2（官方） | 0.8777 / 1.07 | 0.8369 / 1.73 | 0.8113 / 1.60 |

格子为 COMET / 延迟（秒）；同一批条目：FLEURS 两个方向各前 100 句、RealSI 长音频英→中第 1 段（205 s），都开着复读保护（一次都没触发）。官方 llama.cpp 出字偏早（延迟低）是补零造成的“抢答”，质量随之下降。Q8_0 与 Q4_K_M 相差在 0.004 以内。单段长音频在并发下重复跑可差约 0.017，只能看方向。

**解码方式**：默认贪心 + 复读保护。llama.cpp 上的束搜索复用同一套算法（每轮取下一个 token 的前 2 × 束宽 个候选，每条束固定一个槽复用缓存），束宽 1 时与贪心逐 token 相同；但在 llama.cpp 上束宽 2 没有比贪心更好（中→英还低约 0.01），每步耗时翻倍，所以只作为可选项（`--beams 2`，槽数也要翻倍）。

**速度**（A800，单卡单服务，Q8_0）：单路、24 s 窗口时贪心每步 p50 530 ms（束宽 2 为 1,030 ms）；短句约 190 ms。llama.cpp 不会把多路音频的预填拼成一批，同样负载每秒约 4 个请求（vLLM 约 50）。所以长音频的实时会话会跟不上 320 ms 的节拍、合并音频块，延迟比离线测得的大（实测：Q4_K_M 单路按实时推 60 s 音频，每步 p50 481 ms，约 25% 的块被合并）；GGUF 版更适合离线转写、短句或没有 vLLM 的环境，实时服务优先用 vLLM 版。

## 8. 离线批量推理

`simulst.infer.stream` 批量处理音频文件：读入音频清单，按 320 ms 一块模拟流式输入，多条音频一起拼批解码，记录每个字的输出时刻。和实时服务用的是同一套推理逻辑。

清单每行一个 JSON（`audio_path` 为 16 kHz 单声道音频，`direction` 为 `en2zh` 或 `zh2en`）：

```json
{"id": "talk_0001", "audio_path": "/data/talk_0001.wav", "direction": "en2zh"}
```

```bash
python -m simulst.infer.stream --model models/Babelfish-30B-A3B \
  --manifests talks_en=talks_en.jsonl,talks_zh=talks_zh.jsonl --out outputs/ [--beams 2] [--shard i --shards n]
```

结果写在 `outputs/<名字>/results.<分片>.jsonl`，每条包括译文 `hypothesis`、每次输出 `emissions`（输出时刻和文字）、每个字的输出时刻 `delays_ms`。`--shard / --shards` 把条目分到多张卡上并行；FP16 模型一次只解一条（`--max-live 1`）时结果完全确定（FP8 模型在 A800 上重复运行仍有很小的差异），多条一起拼批时会有很小的数值差异。

打分（COMET、BLEU、延迟）和复现首页成绩见 [EVALUATION_zh.md](EVALUATION_zh.md)。

## 9. 常见问题

**显存不足（CUDA out of memory）**：换 FP8 版；或用两张卡张量并行；或降低 `--gpu-mem`、`--max-seqs`；或加 `--eager` 省下 CUDA Graph 的显存。

**网页上麦克风不可用**：浏览器只在 HTTPS 或 localhost 下开放麦克风，见第 4 节。

**译文出得慢、`skipped_blocks` 很多**：单步耗时超过 320 ms 了。看 `done` 事件里的 `step_ms_p50 / p90`；可以减少并发会话、或多开实例；如果用了束宽 2，换回默认的贪心。

**译文出现一句话反复重复**：这是模型复读。复读保护默认开启（`--loop-guard on`），检测到失控复读就在提交前截断；如果被关掉了（`off`），请打开。

**能不能用 bfloat16**：模型以 FP16 发布和评测，请保持默认的 `--dtype float16`。

**不用本仓库代码、直接加载权重**：权重只含音频编码器和语言模型（没有视觉编码器，也没有语音输出部分）。直接用 vLLM 时加 `--limit-mm-per-prompt '{"audio": 1, "image": 0, "video": 0}'`，vLLM 就不会建视觉编码器；transformers 会提示视觉部分的权重缺失并随机初始化，只用语音输入时没有影响。

**pip 装的 vLLM 报 CUDA / 驱动错误**：PyPI 上的默认构建需要支持 CUDA 13 的驱动（>= 580）；旧驱动请装 CUDA 12.9 构建，见 2.1 节。

**支持哪些音频格式**：WebSocket 接收原始 PCM（`s16le` / `f32le`，任意采样率和声道数）；上传接口接收 libsndfile 能解码的格式（wav、flac、ogg 等）；网页上的文件由浏览器解码，常见音视频格式都可以。
