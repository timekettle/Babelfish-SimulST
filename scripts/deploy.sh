#!/usr/bin/env bash
# One-command deployment of the vLLM server: (optionally) create an environment, fetch the model, start the
# web / API service in the foreground (Ctrl+C stops it).
#
#   bash scripts/deploy.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B --install          # new env + HF
#   bash scripts/deploy.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B --install --cu129  # CUDA 12.x driver
#   bash scripts/deploy.sh --model timekettle/Babelfish-SimulMT/Babelfish-30B-A3B-FP8 --install      # FP8 subfolder
#   bash scripts/deploy.sh --model /path/to/Babelfish-30B-A3B --python /path/to/vllm-env/bin/python --gpus 0
#   bash scripts/deploy.sh --model /path/to/Babelfish-30B-A3B --gpus 0,1              # tensor parallel over 2 GPUs
#
# Options: --model ID|DIR  --source hf|modelscope  --install  --cu129  --python PY  --gpus LIST  --tp N  --host H
#          --port P  --beams N  --max-sessions N  --gpu-mem F  --runtime-dir DIR  --no-start  --dry-run  (after --: server args)
# Environment: VLLM_VERSION (default 0.28.0), TORCH_VERSION (default 2.13.0, used with --cu129),
#              PIP_INDEX_URL (pip mirror), SIMULST_PYTHON (python used to create the environment, default python3).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL='' SOURCE=hf INSTALL=0 CU129=0 PY='' GPUS='' TP='' HOST=0.0.0.0 PORT=8080 BEAMS=1 SESSIONS=4 GPU_MEM=0.9 DRY=0 START=1
RUNTIME=$REPO/.simulst
EXTRA=()
VLLM_VERSION=${VLLM_VERSION:-0.28.0}
TORCH_VERSION=${TORCH_VERSION:-2.13.0}
usage() { sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while (( $# )); do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --cu129) CU129=1; shift ;;
    --python) PY=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --tp|--tensor-parallel-size) TP=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --beams) BEAMS=$2; shift 2 ;;
    --max-sessions) SESSIONS=$2; shift 2 ;;
    --gpu-mem|--gpu-memory-utilization) GPU_MEM=$2; shift 2 ;;
    --runtime-dir) RUNTIME=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --no-start) START=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n $MODEL ]] || { echo "--model is required" >&2; usage; exit 2; }
[[ $SOURCE == hf || $SOURCE == modelscope ]] || { echo "--source must be hf or modelscope" >&2; exit 2; }
if [[ -z $TP ]]; then
  if [[ -n $GPUS ]]; then IFS=, read -r -a _g <<< "$GPUS"; TP=${#_g[@]}; else TP=1; fi
fi
mkdir -p "$RUNTIME"
say() { echo "[deploy] $*"; }

# 1. Python environment
if (( INSTALL )); then
  ENV=$RUNTIME/env
  if [[ ! -x $ENV/bin/python ]]; then
    say "creating $ENV"
    (( DRY )) || "${SIMULST_PYTHON:-python3}" -m venv "$ENV"
  fi
  PY=$ENV/bin/python
  if (( ! DRY )); then
    if ! "$PY" -m pip --version > /dev/null 2>&1; then
      echo "the new environment has no pip: install the venv module (Ubuntu: sudo apt install python3-venv) or set" >&2
      echo "SIMULST_PYTHON to a Python 3.10+ that has it (e.g. a conda python), then delete $ENV and rerun" >&2
      exit 1
    fi
    "$PY" -m pip install -U pip
    if (( CU129 )); then
      "$PY" -m pip install "vllm==$VLLM_VERSION+cu129" "torch==$TORCH_VERSION+cu129" \
        --extra-index-url https://download.pytorch.org/whl/cu129 --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/cu129"
    else
      "$PY" -m pip install "vllm==$VLLM_VERSION"
    fi
    "$PY" -m pip install -e "$REPO"
    [[ $SOURCE == modelscope ]] && "$PY" -m pip install modelscope
  fi
fi
PY=${PY:-python3}
if (( ! DRY )); then
  "$PY" -c 'import vllm' 2> /dev/null || { echo "vllm cannot be imported with $PY; use --install or --python" >&2; exit 1; }
fi
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

# 2. Model (supports local directory, owner/repo, or owner/repo/subfolder on Hugging Face)
if [[ -d $MODEL ]]; then
  DIR=$(cd "$MODEL" && pwd)
else
  IFS=/ read -r _m_org _m_repo _m_sub <<< "$MODEL"
  if [[ -n ${_m_sub:-} ]]; then
    REPO_ID="$_m_org/$_m_repo"
    SUBDIR="$_m_sub"
    DL_ROOT=$RUNTIME/models/$_m_repo
    DIR=$DL_ROOT/$SUBDIR
  else
    REPO_ID="$MODEL"
    SUBDIR=""
    DL_ROOT=$RUNTIME/models/$(basename "$MODEL")
    DIR=$DL_ROOT
  fi
  if [[ ! -f $DIR/config.json ]]; then
    say "downloading $MODEL ($SOURCE) to $DIR"
    if (( ! DRY )); then
      if [[ $SOURCE == modelscope ]]; then
        "$PY" -c "import sys; from modelscope import snapshot_download as d; d(sys.argv[1], local_dir=sys.argv[2])" "$REPO_ID" "$DL_ROOT"
      elif [[ -n $SUBDIR ]]; then
        "$PY" -c "import sys; from huggingface_hub import snapshot_download as d; d(sys.argv[1], local_dir=sys.argv[2], allow_patterns=[sys.argv[3] + '/*'])" "$REPO_ID" "$DL_ROOT" "$SUBDIR"
      else
        "$PY" -c "import sys; from huggingface_hub import snapshot_download as d; d(sys.argv[1], local_dir=sys.argv[2])" "$REPO_ID" "$DL_ROOT"
      fi
    fi
  fi
fi

# 3. Server (foreground; the log is also kept in $RUNTIME/server.log)
CMD=("$PY" -m simulst.server --model "$DIR" --host "$HOST" --port "$PORT" --tp "$TP" --beams "$BEAMS"
     --max-sessions "$SESSIONS" --gpu-mem "$GPU_MEM" ${EXTRA[@]+"${EXTRA[@]}"})
say "${GPUS:+CUDA_VISIBLE_DEVICES=$GPUS }${CMD[*]}"
(( DRY )) && exit 0
(( START )) || { say "environment and model ready (--no-start)"; exit 0; }
[[ -n $GPUS ]] && export CUDA_VISIBLE_DEVICES=$GPUS
say "loading the model (a few minutes); open the page when you see 'Ready: http://...'"
"${CMD[@]}" 2>&1 | tee -a "$RUNTIME/server.log"
