#!/usr/bin/env bash
# Decode test sets with all GPUs (one vLLM process per GPU, items sharded) and score them.
#   MODEL=work/model OUT=work/eval VLLM_PY=/envs/vllm/bin/python PY=/envs/eval/bin/python \
#   bash scripts/evaluate.sh fleurs_en2zh=testsets/fleurs_en2zh.jsonl realsi_en2zh=testsets/realsi_en2zh.jsonl@testsets/realsi_en2zh/official_inputs
# name=manifest            sentence-level set (simulst.eval.score_sentences)
# name=manifest@official   long-form set scored against reference segments (simulst.eval.score_longform)
# DECODE_ARGS: extra arguments for simulst.infer.stream (defaults: 24 s window, 256 context tokens, end marker, greedy
# decoding with the repetition guard; e.g. DECODE_ARGS="--beams 2" for beam search).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
: "${MODEL:?}" "${OUT:?}"
(( $# )) || { echo "usage: MODEL=... OUT=... $0 name=manifest[@official_dir] ..."; exit 2; }
VLLM_PY=${VLLM_PY:-python}
PY=${PY:-python}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd' ' -)}
read -r -a G <<< "$GPUS"
mkdir -p "$OUT/logs"
MANS=""
declare -A OFFICIAL MANIFEST
for spec in "$@"; do
  name=${spec%%=*}
  rest=${spec#*=}
  if [[ $rest == *@* ]]; then OFFICIAL[$name]=${rest#*@}; rest=${rest%@*}; fi
  MANIFEST[$name]=$rest
  MANS+="${MANS:+,}$name=$rest"
done
pids=()
for i in "${!G[@]}"; do
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${G[$i]} "$VLLM_PY" -m simulst.infer.stream --model "$MODEL" --manifests "$MANS" --out "$OUT" \
    --shard "$i" --shards "${#G[@]}" ${DECODE_ARGS:-} > "$OUT/logs/decode.$i.log" 2>&1 &
  pids+=($!)
  sleep 15
done
for p in "${pids[@]}"; do wait "$p" || echo "decoder $p exited with an error (see $OUT/logs)"; done
for name in "${!MANIFEST[@]}"; do
  want=$(grep -c . "${MANIFEST[$name]}")
  got=$(cat "$OUT/$name"/results.*.jsonl 2> /dev/null | grep -c .)
  (( got == want )) || { echo "$name: $got / $want items decoded; rerun to resume"; exit 1; }
done
for name in "${!MANIFEST[@]}"; do
  if [[ -n ${OFFICIAL[$name]:-} ]]; then
    "$PY" -m simulst.eval.score_longform "$OUT/$name" "${OFFICIAL[$name]}"
  else
    "$PY" -m simulst.eval.score_sentences "$OUT/$name"
  fi
done
"$PY" -m simulst.eval.report "$OUT"
