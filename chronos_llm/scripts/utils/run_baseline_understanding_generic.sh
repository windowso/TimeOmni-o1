#!/bin/bash
# Unified launcher for the understanding-side baselines (chatts_2503 / timemqa / chattime /
# timeomni1 / timeomnivl).
# Usage: BASELINE=chatts_2503 [LIMIT=5] [JSONL_LIST=...] [SHARD_IDX=0 NUM_SHARDS=1] \
#          bash run_baseline_understanding_generic.sh
# Batch runs on the MMTR understanding pool ("agg6"): pass
# JSONL_LIST=understanding_test_jsonl_agg6_parts.txt (the physically split parts, see
# split_agg6_for_baselines.py) and OUT_DIR=outputs/eval/baseline_<name>/understanding_agg6.
set -ex

REPO=.
CK=checkpoints
BL=third_party
BASE_DIR=${BASE_DIR:-data/raw/scits/Release_v1}
JSONL_LIST=${JSONL_LIST:-$REPO/chronos_llm/configs/understanding_test_jsonl.txt}
LIMIT=${LIMIT:-0}
SHARD_IDX=${SHARD_IDX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
BASELINE=${BASELINE:?BASELINE must be set to one of chatts_2503|timemqa|chattime|timeomni1|timeomnivl}

cd "$REPO"
case "$BASELINE" in
  chatts_2503)
    VENV_PY=third_party/.venv_timeomnivl/bin/python
    OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_chatts_2503/understanding}
    mkdir -p "$OUT_DIR"
    "$VENV_PY" chronos_llm/eval/baseline_chatts_2503.py \
      --model_dir "$CK/Chat_TS" \
      --jsonl_list "$JSONL_LIST" --base_dir "$BASE_DIR" --out_dir "$OUT_DIR" \
      --limit "$LIMIT" --file_shard_idx "$SHARD_IDX" --file_num_shards "$NUM_SHARDS" \
      ${PROMPT_STYLE:+--prompt_style $PROMPT_STYLE}
    ;;
  timemqa)
    VENV_PY=third_party/.venv_timeomnivl/bin/python
    OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_timemqa/understanding}
    mkdir -p "$OUT_DIR"
    "$VENV_PY" chronos_llm/eval/baseline_timemqa.py \
      --adapter_dir "$CK/Time-MQA-Qwen2.5-7B" \
      --base_dir_model "$CK/qwen2.5-7b-unsloth-bnb-4bit" \
      --jsonl_list "$JSONL_LIST" --base_dir "$BASE_DIR" --out_dir "$OUT_DIR" \
      --limit "$LIMIT" --file_shard_idx "$SHARD_IDX" --file_num_shards "$NUM_SHARDS"
    ;;
  chattime)
    # The official ChatTime analysis pipeline depends on its repository's utils/ (Discretizer/Serializer/getPrompt)
    VENV_PY=$BL/.venv_doublecast/bin/python
    export PYTHONPATH="$BL/ChatTime:$PYTHONPATH"
    OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_chattime/understanding}
    mkdir -p "$OUT_DIR"
    "$VENV_PY" chronos_llm/eval/baseline_chattime_understanding.py \
      --ckpt checkpoints/ChatTime-1-7B-Chat \
      --jsonl_list "$JSONL_LIST" --base_dir "$BASE_DIR" --out_dir "$OUT_DIR" \
      --limit "$LIMIT" --file_shard_idx "$SHARD_IDX" --file_num_shards "$NUM_SHARDS"
    ;;
  timeomni1)
        OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_timeomni1/understanding}
    mkdir -p "$OUT_DIR"
    python -u chronos_llm/eval/baseline_timeomni1_understanding.py \
      --model_dir "$CK/TimeOmni-1-4B" \
      --jsonl_list "$JSONL_LIST" --base_dir "$BASE_DIR" --out_dir "$OUT_DIR" \
      --limit "$LIMIT" --file_shard_idx "$SHARD_IDX" --file_num_shards "$NUM_SHARDS"
    ;;
  timeomnivl)
    VENV_PY=$BL/.venv_timeomnivl/bin/python
    OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_timeomnivl/understanding}
    mkdir -p "$OUT_DIR"
    "$VENV_PY" chronos_llm/eval/baseline_timeomnivl_understanding.py \
      --model_dir "$CK/TimeOmni-VL" \
      --jsonl_list "$JSONL_LIST" --base_dir "$BASE_DIR" --out_dir "$OUT_DIR" \
      --limit "$LIMIT" --file_shard_idx "$SHARD_IDX" --file_num_shards "$NUM_SHARDS"
    ;;
  *) echo "unknown BASELINE=$BASELINE"; exit 1;;
esac
