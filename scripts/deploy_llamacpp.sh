#!/usr/bin/env bash
# One-command deployment of the GGUF version: start llama-server with the model and its audio encoder, then the
# web / API service on top of it (repetition guard on). Ctrl+C stops both. For results matching the vLLM version,
# build llama-server with the audio patch in patches/llama.cpp (see docs/DEPLOYMENT.md).
#
#   bash scripts/deploy_llamacpp.sh --model /path/to/Babelfish-30B-A3B-GGUF --quant Q4_K_M --llama-server /path/to/llama-server
#   bash scripts/deploy_llamacpp.sh --model TimeKettle/Babelfish-30B-A3B-GGUF --quant Q8_0 --gpus 0 --beams 2
#
# Options: --model DIR|ID  --quant Q8_0|Q4_K_M  --source hf|modelscope  --llama-server PATH  --python PY  --gpus LIST
#          --sessions N (concurrent sessions, default 4)  --beams N (default 1 = greedy)  --ngl N (layers on the GPU)
#          --host H  --port P  --llama-port P  --runtime-dir DIR  --dry-run   (anything after -- goes to llama-server)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL='' QUANT=Q4_K_M SOURCE=hf LLAMA=${LLAMA_SERVER:-llama-server} PY=python3 GPUS='' SESSIONS=4 BEAMS=1 NGL=999
HOST=0.0.0.0 PORT=8080 LPORT=8081 RUNTIME=$REPO/.simulst DRY=0 EXTRA=()
usage() { sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while (( $# )); do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --quant) QUANT=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --llama-server) LLAMA=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --sessions) SESSIONS=$2; shift 2 ;;
    --beams) BEAMS=$2; shift 2 ;;
    --ngl) NGL=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --llama-port) LPORT=$2; shift 2 ;;
    --runtime-dir) RUNTIME=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n $MODEL ]] || { echo "--model is required" >&2; usage; exit 2; }
say() { echo "[deploy-llamacpp] $*"; }
GGUF=Babelfish-30B-A3B-$QUANT.gguf
MMPROJ=mmproj-Babelfish-30B-A3B-F16.gguf
mkdir -p "$RUNTIME"

# 1. model files
if [[ -d $MODEL ]]; then
  DIR=$(cd "$MODEL" && pwd)
else
  DIR=$RUNTIME/models/$(basename "$MODEL")
  if [[ ! -f $DIR/$GGUF || ! -f $DIR/$MMPROJ ]]; then
    say "downloading $GGUF and $MMPROJ from $MODEL ($SOURCE)"
    if (( ! DRY )); then
      if [[ $SOURCE == modelscope ]]; then
        "$PY" -c "import sys; from modelscope import snapshot_download as d; d(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])" "$MODEL" "$DIR" "$GGUF" "$MMPROJ"
      else
        "$PY" -c "import sys; from huggingface_hub import snapshot_download as d; d(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])" "$MODEL" "$DIR" "$GGUF" "$MMPROJ"
      fi
    fi
  fi
fi
(( DRY )) || [[ -f $DIR/$GGUF && -f $DIR/$MMPROJ ]] || { echo "missing $DIR/$GGUF or $DIR/$MMPROJ" >&2; exit 1; }
(( DRY )) || command -v "$LLAMA" > /dev/null || { echo "llama-server not found: build llama.cpp and pass --llama-server PATH" >&2; exit 1; }

# 2. llama-server (background, one slot per session and beam) + web / API service (foreground)
PARALLEL=$((SESSIONS * BEAMS))
LCMD=("$LLAMA" -m "$DIR/$GGUF" --mmproj "$DIR/$MMPROJ" -ngl "$NGL" --parallel "$PARALLEL" -c $((PARALLEL * 4096))
      --host 127.0.0.1 --port "$LPORT" ${EXTRA[@]+"${EXTRA[@]}"})
BCMD=("$PY" -m simulst.server --backend llamacpp --llama-url "http://127.0.0.1:$LPORT" --host "$HOST" --port "$PORT"
      --max-sessions "$SESSIONS" --beams "$BEAMS")
say "${GPUS:+CUDA_VISIBLE_DEVICES=$GPUS }${LCMD[*]}"
say "${BCMD[*]}"
(( DRY )) && exit 0
[[ -n $GPUS ]] && export CUDA_VISIBLE_DEVICES=$GPUS
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
"${LCMD[@]}" > "$RUNTIME/llama-server.log" 2>&1 &
LPID=$!
cleanup() {   # stop llama-server; force it after 5 s
  kill "$LPID" 2> /dev/null || return 0
  for _ in 1 2 3 4 5; do kill -0 "$LPID" 2> /dev/null || return 0; sleep 1; done
  kill -9 "$LPID" 2> /dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM
say "loading $GGUF (log: $RUNTIME/llama-server.log)"
for _ in $(seq 1 600); do
  curl -s -m 2 "http://127.0.0.1:$LPORT/health" 2> /dev/null | grep -q '"ok"' && break
  kill -0 $LPID 2> /dev/null || { echo "llama-server exited:" >&2; tail -20 "$RUNTIME/llama-server.log" >&2; exit 1; }
  sleep 2
done
"${BCMD[@]}" 2>&1 | tee -a "$RUNTIME/server.log"
