#!/bin/bash
# TimeOmni-VL forecasting-side baseline inference: consumes the jsonl produced by
# build_timeomnivl_forecast_data.py (image_mask.png + image_full.png + metadata.json + series.npz)
# and calls its official eval/generation_inference.py (Bagel-7B architecture, local large-model
# inference, needs a GPU).
# Usage: bash chronos_llm/scripts/utils/baseline_timeomnivl_forecast.sh
# Smoke test first with MAX_SAMPLES=3.
set -e -o pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
VENV=$REPO_ROOT/third_party/.venv_timeomnivl
REPO_TOVL=$REPO_ROOT/third_party/TimeOmni-VL
CKPT=$REPO_ROOT/checkpoints/TimeOmni-VL
DATA_ROOT=$REPO_ROOT/outputs/eval/baseline_timeomnivl/forecast_data

OUT_DIR=${OUT_DIR:-$REPO_ROOT/outputs/eval/baseline_timeomnivl/forecast_gen}
mkdir -p "$OUT_DIR"

cd "$REPO_TOVL"
"$VENV/bin/python" eval/generation_inference.py \
  --base_model "$CKPT" \
  --jsonl "$DATA_ROOT/forecast_samples.jsonl" \
  --input-root "$DATA_ROOT" \
  --output-root "$OUT_DIR" \
  --output-name edit.png \
  --think \
  --device-ids "${DEVICE_IDS:-0}" \
  --no-shuffle \
  ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES} \
  --metrics-csv "$OUT_DIR/metrics.csv" \
  2>&1 | tee "$OUT_DIR/infer_$(date +%Y%m%d-%H%M%S).log"

echo "[baseline_timeomnivl_forecast] metrics -> $OUT_DIR/metrics.csv"
cat "$OUT_DIR/metrics.csv" 2>/dev/null | head -5
