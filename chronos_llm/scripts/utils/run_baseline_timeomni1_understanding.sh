#!/bin/bash
# Launcher for the TimeOmni-1 understanding-side baseline (interactive debugging / full runs).
set -ex
# The environment may lack soundfile (required to load the .wav/.flac audio datasets) -- without it
# those series silently degrade to all-zero placeholders. Install it if possible; if the install
# fails, do not block.
python -c "import soundfile" 2>/dev/null || pip install soundfile || true

REPO=.
MODEL_DIR=${MODEL_DIR:-checkpoints/TimeOmni-1-4B}
JSONL_LIST=${JSONL_LIST:-$REPO/chronos_llm/configs/understanding_test_jsonl.txt}
BASE_DIR=${BASE_DIR:-data/raw/scits/Release_v1}
OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_timeomni1/understanding}
LIMIT=${LIMIT:-0}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_LEN=${MAX_LEN:-512}
FILE_SHARD_IDX=${FILE_SHARD_IDX:-0}
FILE_NUM_SHARDS=${FILE_NUM_SHARDS:-1}

cd "$REPO"
mkdir -p "$OUT_DIR"

python -u chronos_llm/eval/baseline_timeomni1_understanding.py \
  --model_dir "$MODEL_DIR" \
  --jsonl_list "$JSONL_LIST" \
  --base_dir "$BASE_DIR" \
  --out_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_len "$MAX_LEN" \
  --limit "$LIMIT" \
  --file_shard_idx "$FILE_SHARD_IDX" \
  --file_num_shards "$FILE_NUM_SHARDS"
