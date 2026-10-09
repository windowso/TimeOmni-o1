#!/bin/bash
# Launcher for the Time-MQA forecasting-side baseline.
# Inference uses third_party/.venv_timeomnivl (transformers+peft+bitsandbytes, the same environment
# as the understanding side); evaluation (eval_forecast, CPU only) runs in the main project environment.
set -ex

REPO=.
CK=checkpoints
VENV_PY=third_party/.venv_timeomnivl/bin/python
PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_timemqa}
OUTPUT=${OUTPUT:-$OUT_DIR/forecast_preds.npz}
LIMIT=${LIMIT:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2560}

cd "$REPO"
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$VENV_PY" -u chronos_llm/eval/baseline_timemqa_forecast.py \
  --adapter_dir "$CK/Time-MQA-Qwen2.5-7B" \
  --base_dir_model "$CK/qwen2.5-7b-unsloth-bnb-4bit" \
  --parquet "$PARQUET" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --output "$OUTPUT" \
  2>&1 | tee "$(dirname "$OUTPUT")/infer_forecast_$(date +%Y%m%d-%H%M%S).log"

# Metrics are computed only after a full run (skipped for smoke runs with LIMIT>0, so that
# small-sample metrics are never mistaken for results)
if [ "$LIMIT" = "0" ]; then
    export PYTHONPATH="$REPO:$PYTHONPATH"
  python -m chronos_llm.eval.eval_forecast --pred "$OUTPUT" \
    --output_csv "$(dirname "$OUTPUT")/forecast_metrics.csv"
  cat "$(dirname "$OUTPUT")/forecast_metrics.csv"
fi
