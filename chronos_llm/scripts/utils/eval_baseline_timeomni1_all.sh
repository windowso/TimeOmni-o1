#!/bin/bash
# Unified evaluation of the TimeOmni-1 baseline once all inference shards are done (pure CPU):
#   1. eval_forecast.py computes full/roi MAPE/PCC/CRPS
#   2. eval_understanding.py computes classification/detection metrics for the jsonl files under understanding/
#   3. paper_understanding_domains.py aggregates by domain
#   4. paper_forecast_by_domain.py aggregates by the 5 forecasting domains
set -ex

REPO=.
BASE=$REPO/outputs/eval/baseline_timeomni1
PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
PY=${PY:-python}
cd "$REPO"
export PYTHONPATH=$REPO:$REPO/src

echo "===== step 1: forecast metrics ====="
"$PY" chronos_llm/eval/eval_forecast.py --pred "$BASE/forecast_preds.npz" \
  --output_csv "$BASE/forecast_metrics.csv"

echo "===== step 2: understanding metrics ====="
"$PY" -m chronos_llm.eval.eval_understanding "$BASE/understanding" \
  -o "$BASE/understanding_metrics.csv"

echo "===== step 3: domain aggregation ====="
"$PY" chronos_llm/scripts/utils/paper_understanding_domains.py \
  "$BASE/understanding_metrics.csv" --label TimeOmni-1 \
  --out-csv "$BASE/understanding_domains.csv" || true

echo "===== step 4: forecast domain aggregation ====="
"$PY" chronos_llm/scripts/utils/paper_forecast_by_domain.py \
  "TimeOmni-1=$BASE/forecast_preds.npz" --parquet "$PARQUET" \
  --out-csv "$BASE/forecast_domains.csv" || true

echo "===== ALL EVAL DONE ====="
ls -la "$BASE"
