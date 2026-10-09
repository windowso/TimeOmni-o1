#!/bin/bash
# Forecasting-only training: parquet -> soft prompt + (teacher-forced or self-generated) reasoning
# -> feedback Q-Former -> gated cross-attention into every Chronos-2 encoder block -> quantile
# forecast. Loss = text CE + masked pinball loss (full horizon) + ROI-weighted pinball loss.
#
# Usage:
#   bash chronos_llm/scripts/train_forecast.sh
#
# Training recipe for forecasting (2 GPUs, 30 epochs, scheduled sampling of the fed-back text 1.0 -> 0.7):
#   PARQUET=data/forecast/mmtr_forecast_corpus.parquet GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 \
#   SS_END_RATIO=0.7 PROC_PER_NODE=2 bash chronos_llm/scripts/train_forecast.sh
# ST-Bench T4 fine-tuning (same recipe, small corpus -> fixed batch size):
#   PARQUET=data/external/stbench_t4_forecast.parquet GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 \
#   SS_END_RATIO=0.7 PROC_PER_NODE=2 FORECAST_BS=32 F_TOKEN_BUDGET=0 EVAL_F_BS=32 bash chronos_llm/scripts/train_forecast.sh
# Ablation arms (mutually exclusive unless noted):
#   CHRONOS_ONLY=1              fine-tune Chronos-2 alone (no LLM, no feedback)
#   FORECAST_PROMPT_ONLY=1      no generated reasoning; the prompt's hidden states are fed back ("PO")
#   FEEDBACK_SCOPE=conclusion   feed back only the conclusion span instead of the whole trace
#   CHRONOS_RANDOM_INIT=1       randomly initialised Chronos-2 (can be combined with the others)
#   LLM_ONLY=1                  LLM-only backbone: history serialized as text, forecast written as text
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
DATA_TAG="_$(basename "$PARQUET" .parquet)"

EPOCHS=${EPOCHS:-30}
LR=${LR:-1e-5}; LR_LORA=${LR_LORA:-1e-4}; LR_QFORMER=${LR_QFORMER:-1e-4}; LR_GATE=${LR_GATE:-1e-3}
FORECAST_BS=${FORECAST_BS:-150}
F_TOKEN_BUDGET=${F_TOKEN_BUDGET:-155000}
PATCH_BUDGET=${PATCH_BUDGET:-0}; MAX_DYN_BS=${MAX_DYN_BS:-1000}; ENCODE_MAX_ROWS=${ENCODE_MAX_ROWS:-58}
BS_LADDER=${BS_LADDER:-"1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,300,400,500,600,700,800,900,1000"}
GATE_INIT=${GATE_INIT:-0}                  # 1.0 in the released recipe (gate starts open)
SS_START_RATIO=${SS_START_RATIO:-1.0}; SS_END_RATIO=${SS_END_RATIO:-1.0}  # scheduled sampling of the fed-back text
MAX_USER_TOKENS=${MAX_USER_TOKENS:-1500}; MAX_TOKENS=${MAX_TOKENS:-4096}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0}    # 0 = keep every epoch (adapter-only checkpoints)
LOGGING_STEPS=${LOGGING_STEPS:-10}; NUM_WORKERS=${NUM_WORKERS:-8}
RUN_TS=${RUN_TS:-$(date +%Y%m%d-%H%M%S)}

EXTRA=(); TAG=""
[ -n "${ZERO_CROSS_OUT_PROJ:-}" ] && { EXTRA+=(--zero_cross_out_proj); TAG="${TAG}_zop"; }
if [ "${LLM_ONLY:-0}" = "1" ]; then
  EXTRA+=(--llm_only); TAG="${TAG}_llmonly"
  # Serialized histories are long (up to ~16k tokens); widen the budgets and the generation length.
  [ "$MAX_USER_TOKENS" = "1500" ] && MAX_USER_TOKENS=17000
  [ "$MAX_TOKENS" = "4096" ] && MAX_TOKENS=20480
  EVAL_F_MAX_NEW_TOKENS=${EVAL_F_MAX_NEW_TOKENS:-2560}; EVAL_F_BS=${EVAL_F_BS:-32}
fi
[ "${CHRONOS_ONLY:-0}" = "1" ]         && { EXTRA+=(--chronos_only); TAG="${TAG}_chronosonly"; }
[ "${FORECAST_PROMPT_ONLY:-0}" = "1" ] && { EXTRA+=(--forecast_prompt_only); TAG="${TAG}_promptonly"; }
[ "${FEEDBACK_SCOPE:-all}" != "all" ]  && { EXTRA+=(--feedback_scope "$FEEDBACK_SCOPE"); TAG="${TAG}_fb${FEEDBACK_SCOPE}"; }
[ "${CHRONOS_RANDOM_INIT:-0}" = "1" ]  && { EXTRA+=(--chronos_random_init); TAG="${TAG}_randinit"; }
if [ "$SS_END_RATIO" != "1.0" ] || [ "$SS_START_RATIO" != "1.0" ]; then
  EXTRA+=(--ss_start_ratio "$SS_START_RATIO" --ss_end_ratio "$SS_END_RATIO" --ss_max_new_tokens "${SS_MAX_NEW_TOKENS:-288}")
  TAG="${TAG}_ss${SS_START_RATIO}-${SS_END_RATIO}"
fi
[ -n "${INIT_FROM_CHECKPOINT:-}" ] && { EXTRA+=(--init_from_checkpoint "$INIT_FROM_CHECKPOINT"); TAG="${TAG}_initckpt"; }
[ "${INIT_MERGE_REOPEN:-0}" = "1" ]  && { EXTRA+=(--init_merge_reopen); TAG="${TAG}merge"; }
[ "${FREEZE_LLM:-0}" = "1" ]         && { EXTRA+=(--freeze_llm); TAG="${TAG}_freezellm"; }
# Architecture ablations (see train_understanding.sh for the meaning of each switch).
[ -n "${HISTORY_COMPRESSOR:-}" ]  && { EXTRA+=(--history_compressor "$HISTORY_COMPRESSOR"); TAG="${TAG}_hc-${HISTORY_COMPRESSOR}"; }
[ -n "${FEEDBACK_COMPRESSOR:-}" ] && { EXTRA+=(--feedback_compressor "$FEEDBACK_COMPRESSOR"); TAG="${TAG}_fc-${FEEDBACK_COMPRESSOR}"; }
[ -n "${HISTORY_ENCODE_LAYER:-}" ] && { EXTRA+=(--history_encode_layer "$HISTORY_ENCODE_LAYER"); TAG="${TAG}_hel${HISTORY_ENCODE_LAYER}"; }
[ -n "${FEEDBACK_LLM_LAYER:-}" ]  && { EXTRA+=(--feedback_llm_layer "$FEEDBACK_LLM_LAYER"); TAG="${TAG}_fll${FEEDBACK_LLM_LAYER}"; }
[ -n "${CROSS_ATTN_LAYERS:-}" ]   && { EXTRA+=(--cross_attn_layers "$CROSS_ATTN_LAYERS"); TAG="${TAG}_cal-${CROSS_ATTN_LAYERS}"; }
[ -n "${SW_GLOBAL_QUERIES:-}" ]   && { EXTRA+=(--sw_global_queries "$SW_GLOBAL_QUERIES"); TAG="${TAG}_gq${SW_GLOBAL_QUERIES}"; }
[ -n "${SW_TOKEN_MAX:-}" ]        && { EXTRA+=(--sw_token_max "$SW_TOKEN_MAX"); TAG="${TAG}_tmax${SW_TOKEN_MAX}"; }
[ -n "${SEED:-}" ]                && { EXTRA+=(--seed "$SEED"); TAG="${TAG}_s${SEED}"; }
if [ -n "${MAX_STEPS:-}" ]; then STEP_ARG=(--max_steps "$MAX_STEPS"); else STEP_ARG=(--num_train_epochs "$EPOCHS"); fi

RUN_NAME="ep${EPOCHS}_lr${LR}_lora${LR_LORA}_qf${LR_QFORMER}_gate${LR_GATE}_g${GATE_INIT}_fbs${FORECAST_BS}_ftb${F_TOKEN_BUDGET}${TAG}${BACKBONE_TAG}${DATA_TAG}_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/forecast_run}/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"; archive_script "$OUTPUT_DIR" "${BASH_SOURCE[0]}"
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d-%H%M%S)_node${NODE_RANK}.log"
echo "[train_forecast] OUTPUT_DIR=$OUTPUT_DIR"

torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK --nproc_per_node=$PROC_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT:-29500} chronos_llm/train.py \
  --chronos_ckpt "$CHRONOS" --llm_path "$LLM" \
  --tsfm_backbone "$TSFM_BACKBONE" \
  --forecast_parquet "$PARQUET" \
  --output_dir "$OUTPUT_DIR" \
  --bf16 --gradient_checkpointing --deepspeed chronos_llm/configs/ds_zero2.json \
  --learning_rate $LR --lr_lora $LR_LORA --lr_qformer $LR_QFORMER --lr_gate $LR_GATE \
  --gate_init $GATE_INIT \
  "${STEP_ARG[@]}" --warmup_ratio 0.03 \
  --forecast_bs $FORECAST_BS --forecast_token_budget $F_TOKEN_BUDGET \
  --patch_budget $PATCH_BUDGET --max_dynamic_bs $MAX_DYN_BS --dynamic_bs_ladder "$BS_LADDER" \
  --encode_max_rows $ENCODE_MAX_ROWS --forecast_max_context 8192 \
  --max_user_tokens $MAX_USER_TOKENS --max_tokens $MAX_TOKENS \
  --text_loss_weight 1.0 --pred_loss_weight 1.0 --roi_loss_weight ${ROI_LOSS_WEIGHT:-1.0} \
  --eval_every_epochs ${EVAL_EVERY_EPOCHS:-1} --eval_forecast_limit ${EVAL_F_LIMIT:-0} \
  --eval_forecast_bs ${EVAL_F_BS:-64} --eval_forecast_max_new_tokens ${EVAL_F_MAX_NEW_TOKENS:-320} \
  "${EXTRA[@]}" \
  --logging_steps $LOGGING_STEPS --save_total_limit $SAVE_TOTAL_LIMIT --save_only_model \
  --dataloader_num_workers $NUM_WORKERS \
  2>&1 | tee -a "$LOG_FILE"
