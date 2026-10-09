#!/bin/bash
# Chronos-2 zero-shot forecasting baseline: the
# original chronos2 with cross_states=None, scored on the forecast parquet split=test with
# MAPE/PCC/CRPS (full window + ROI). Only chronos2 is loaded (small, no 9B LLM), so it also
# runs on CPU:
#   bash chronos_llm/scripts/utils/baseline_forecast_zeroshot.sh
set -e -o pipefail
REPO=.
cd "$REPO"; export PYTHONPATH="$REPO:$PYTHONPATH"; export PYTHONUNBUFFERED=1

OUT_DIR=${OUT_DIR:-$REPO/outputs/baseline_forecast_zeroshot}
mkdir -p "$OUT_DIR"

python chronos_llm/eval/baseline_forecast_zeroshot.py \
  ${PARQUET:+--parquet "$PARQUET"} \
  --split test --limit ${LIMIT:-500} \
  --batch_size ${BATCH_SIZE:-64} \
  --device ${DEVICE:-cuda} \
  --output "$OUT_DIR/zeroshot_preds.npz" \
  --output_csv "$OUT_DIR/zeroshot_metrics.csv" \
  2>&1 | tee "$OUT_DIR/baseline_$(date +%Y%m%d-%H%M%S).log"
