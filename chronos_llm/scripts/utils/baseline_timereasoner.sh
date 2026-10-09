#!/bin/bash
# TimeReasoner-style baseline (WSDM 2025, pure prompting; does not execute its official repository
# code -- see the header of chronos_llm/eval/baseline_timereasoner.py): only calls the LLM API
# (default deepseek-reasoner -- the same model identifier literal as in the original paper's code;
# see the "model choice" section of the .py header).
# Pure API network calls, no local model is loaded:
#   bash chronos_llm/scripts/utils/baseline_timereasoner.sh
# Resumable: the script itself recovers already successful samples from raw_log.jsonl (see
# --no_resume); if interrupted, simply rerun the same command and successful samples are not
# called again.
set -e -o pipefail
set -a
if [ -f .env ]; then source .env; fi   # optional: API keys (OPENAI_API_KEY ...)
set +a
REPO=.
cd "$REPO"; export PYTHONPATH="$REPO:$PYTHONPATH"; export PYTHONUNBUFFERED=1

OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_timereasoner}
mkdir -p "$OUT_DIR"

python chronos_llm/eval/baseline_timereasoner.py \
  --parquet ${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet} \
  --model ${MODEL:-deepseek-reasoner} \
  --hist_len ${HIST_LEN:-600} \
  --max_workers ${MAX_WORKERS:-5} \
  --timeout ${API_TIMEOUT:-180} \
  --checkpoint_every ${CHECKPOINT_EVERY:-20} \
  ${LIMIT:+--limit $LIMIT} \
  --output "$OUT_DIR/forecast_preds.npz" \
  --log_jsonl "$OUT_DIR/raw_log.jsonl" \
  --failures_jsonl "$OUT_DIR/failures.jsonl" \
  2>&1 | tee -a "$OUT_DIR/infer_$(date +%Y%m%d-%H%M%S).log"

python -m chronos_llm.eval.eval_forecast --pred "$OUT_DIR/forecast_preds.npz" \
  --output_csv "$OUT_DIR/forecast_metrics.csv"
cat "$OUT_DIR/forecast_metrics.csv"
