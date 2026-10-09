#!/bin/bash
# Unified evaluation once all TimeOmni-VL baseline inference shards are done (CPU only):
#   1. merge the 8 forecast shard npz files -> a single npz
#   2. eval_forecast.py computes full/roi MAPE/PCC/CRPS
#   3. eval_understanding.py computes classification/detection metrics for the jsonl files under understanding/
#   4. paper_understanding_domains.py aggregates by domain
# Note: eval_* need the repository's Python environment (they depend on the chronos_llm package); the merge step
# works with any python.
set -ex

REPO=.
BASE=$REPO/outputs/eval/baseline_timeomnivl
PY=${PY:-python}
cd "$REPO"
export PYTHONPATH=$REPO:$REPO/src

echo "===== step 1: merge forecast shards ====="
"$PY" chronos_llm/scripts/utils/merge_timeomnivl_forecast_shards.py \
  "$BASE/forecast_preds_full.npz" --num_shards 8 || {
    # The shard files are named forecast_preds.shard{i}.npz by the batch-inference script's OUT_DIR convention;
    # the merge script derives the shard names from the output file name, so align it first:
    echo "trying with the batch-inference output naming (forecast_preds.npz base)"
    "$PY" chronos_llm/scripts/utils/merge_timeomnivl_forecast_shards.py \
      "$BASE/forecast_preds.npz" --num_shards 8
  }
MERGED=$BASE/forecast_preds_full.npz
[ -f "$MERGED" ] || MERGED=$BASE/forecast_preds.npz

echo "===== step 2: forecast metrics ====="
"$PY" -m chronos_llm.eval.eval_forecast --pred "$MERGED" \
  --output_csv "$BASE/forecast_metrics.csv"

echo "===== step 3: understanding metrics ====="
"$PY" -m chronos_llm.eval.eval_understanding "$BASE/understanding" \
  -o "$BASE/understanding_metrics.csv"

echo "===== step 4: domain aggregation ====="
"$PY" chronos_llm/scripts/utils/paper_understanding_domains.py \
  "$BASE/understanding_metrics.csv" --label TimeOmni-VL \
  --out-csv "$BASE/understanding_domains.csv" || true

echo "===== ALL EVAL DONE ====="
ls -la "$BASE"
