#!/bin/bash
# Launcher for the TimeOmni-VL forecasting-side baseline.
# Separate venv (third_party/.venv_timeomnivl); does not use the main project environment and does not import chronos_llm.
set -ex

REPO=.
VENV_PY=third_party/.venv_timeomnivl/bin/python
MODEL_DIR=${MODEL_DIR:-checkpoints/TimeOmni-VL}
PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_timeomnivl}
LIMIT=${LIMIT:-0}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-50}
SHARD_IDX=${SHARD_IDX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SAVE_IMAGES_DIR=${SAVE_IMAGES_DIR:-}

cd "$REPO"
mkdir -p "$OUT_DIR"

EXTRA=()
if [ -n "$SAVE_IMAGES_DIR" ]; then
  EXTRA+=(--save_images_dir "$SAVE_IMAGES_DIR")
fi

"$VENV_PY" chronos_llm/eval/baseline_timeomnivl.py \
  --model_dir "$MODEL_DIR" \
  --parquet "$PARQUET" \
  --output "$OUT_DIR/forecast_preds.npz" \
  --limit "$LIMIT" \
  --num_timesteps "$NUM_TIMESTEPS" \
  --shard_idx "$SHARD_IDX" \
  --num_shards "$NUM_SHARDS" \
  "${EXTRA[@]}"
