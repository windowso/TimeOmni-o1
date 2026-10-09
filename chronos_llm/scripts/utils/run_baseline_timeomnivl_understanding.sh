#!/bin/bash
# Launcher for the TimeOmni-VL understanding-side baseline (works for interactive debugging and batch jobs alike).
set -ex

REPO=.
VENV_PY=third_party/.venv_timeomnivl/bin/python
MODEL_DIR=${MODEL_DIR:-checkpoints/TimeOmni-VL}
JSONL_LIST=${JSONL_LIST:-$REPO/chronos_llm/configs/understanding_test_jsonl.txt}
BASE_DIR=${BASE_DIR:-data/raw/scits/Release_v1}
OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_timeomnivl/understanding}
LIMIT=${LIMIT:-0}
FILE_SHARD_IDX=${FILE_SHARD_IDX:-0}
FILE_NUM_SHARDS=${FILE_NUM_SHARDS:-1}

cd "$REPO"
mkdir -p "$OUT_DIR"

"$VENV_PY" chronos_llm/eval/baseline_timeomnivl_understanding.py \
  --model_dir "$MODEL_DIR" \
  --jsonl_list "$JSONL_LIST" \
  --base_dir "$BASE_DIR" \
  --out_dir "$OUT_DIR" \
  --limit "$LIMIT" \
  --file_shard_idx "$FILE_SHARD_IDX" \
  --file_num_shards "$FILE_NUM_SHARDS"
