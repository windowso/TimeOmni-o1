#!/bin/bash
# Launch script for the TimeOmni-1 forecasting baseline (interactive debugging and full runs alike).
set -ex

REPO=.
MODEL_DIR=${MODEL_DIR:-checkpoints/TimeOmni-1-4B}
PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
OUTPUT=${OUTPUT:-$REPO/outputs/eval/baseline_timeomni1/forecast_preds.npz}
LIMIT=${LIMIT:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
HIST_LEN=${HIST_LEN:-600}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2560}

cd "$REPO"
mkdir -p "$(dirname "$OUTPUT")"

python -u chronos_llm/eval/baseline_timeomni1.py \
  --model_dir "$MODEL_DIR" \
  --parquet "$PARQUET" \
  --hist_len "$HIST_LEN" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --output "$OUTPUT"
