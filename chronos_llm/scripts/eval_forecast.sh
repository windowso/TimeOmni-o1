#!/bin/bash
# Forecasting evaluation: torchrun inference on the parquet test split (sharded over ranks, merged
# by rank 0 into an npz + jsonl) followed by CPU metrics (CRPS / PCC / MAPE over the full horizon
# and over the ROI). MODE=both evaluates two settings with one model load:
#   generate        the model writes its own reasoning + forecast instruction (the reported number)
#   teacher_forced  the annotated trace is fed back instead (upper bound / feedback-pathway diagnostic)
#
#   bash chronos_llm/scripts/eval_forecast.sh <MODEL_PATH> [OUT_DIR]
# Environment: PARQUET (default: MMTR corpus), MODE=generate|teacher_forced|both, BATCH_SIZE,
#   MAX_NEW_TOKENS, ENSEMBLE_K (average K sampled reasonings), FB_MASK_GEN=1 (mask the generated
#   span from the feedback), MERGE_LORA=1 (merge the adapter before inference; default keeps it
#   unmerged, which reproduces the training-time numerics exactly), LLM_ONLY_ZEROSHOT=1.
# Per-domain aggregation (equal-weight mean over the domains):
#   python chronos_llm/scripts/utils/paper_forecast_by_domain.py run=<OUT_DIR>/forecast_preds.npz --parquet $PARQUET
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_PATH=${1:?usage: eval_forecast.sh <MODEL_PATH> [OUT_DIR]}
OUT_DIR=${2:-outputs/eval/forecast}
mkdir -p "$OUT_DIR"
MODE=${MODE:-both}
PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
OUT_NPZ="$OUT_DIR/forecast_preds.npz"
[ "$MODE" = "teacher_forced" ] && OUT_NPZ="$OUT_DIR/forecast_preds_tf.npz"
MERGE_LORA=${MERGE_LORA:-0}
if [ "${LLM_ONLY_ZEROSHOT:-0}" = "1" ]; then
  LOZS_ARG=(--llm_only_zeroshot); MODE=generate
  MAX_USER_TOKENS=${MAX_USER_TOKENS:-17000}; MAX_TOKENS=${MAX_TOKENS:-20480}
  MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}; BATCH_SIZE=${BATCH_SIZE:-32}
else
  LOZS_ARG=()
fi

torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK --nproc_per_node=$PROC_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))} \
  chronos_llm/eval/infer_forecast.py \
  --model_path "$MODEL_PATH" \
  --parquet "$PARQUET" \
  --split test --limit ${LIMIT:-500} \
  --batch_size ${BATCH_SIZE:-64} --max_new_tokens ${MAX_NEW_TOKENS:-320} \
  --max_user_tokens ${MAX_USER_TOKENS:-1500} --max_tokens ${MAX_TOKENS:-4096} \
  --num_workers ${NUM_WORKERS:-4} --num_beams ${NUM_BEAMS:-1} --ensemble_k ${ENSEMBLE_K:-1} \
  ${FB_MASK_GEN:+--fb_mask_generated} \
  $([ "$MERGE_LORA" != "1" ] && echo --no_merge) \
  "${LOZS_ARG[@]}" \
  --mode "$MODE" \
  --output "$OUT_NPZ" \
  --tf_output "$OUT_DIR/forecast_preds_tf.npz"

if [ "$NODE_RANK" = "0" ]; then
  if [ "$MODE" != "teacher_forced" ]; then
    python chronos_llm/eval/eval_forecast.py --pred "$OUT_DIR/forecast_preds.npz" --output_csv "$OUT_DIR/forecast_metrics.csv"
    python chronos_llm/scripts/utils/paper_forecast_by_domain.py "generate=$OUT_DIR/forecast_preds.npz" --parquet "$PARQUET" \
      --out-csv "$OUT_DIR/forecast_by_domain.csv" || true
  fi
  if [ "$MODE" != "generate" ]; then
    python chronos_llm/eval/eval_forecast.py --pred "$OUT_DIR/forecast_preds_tf.npz" --output_csv "$OUT_DIR/forecast_metrics_tf.csv"
  fi
fi
