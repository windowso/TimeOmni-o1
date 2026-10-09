#!/bin/bash
# Understanding-only SFT: jsonl -> Chronos-2 encoder -> sliding-window Q-Former soft prompt -> LLM,
# supervised with text cross-entropy on `<think>reasoning</think>` + answer.
# The feedback Q-Former and the gated cross-attention are not exercised by this branch
# (they belong to the forecasting pathway), so `gate_tanh_mean/max` stays at its initial value.
#
# Usage (single node, all GPUs; every knob below can be overridden from the environment):
#   bash chronos_llm/scripts/train_understanding.sh
#
# Training recipe for understanding (two stages):
#   Stage 1, SFT on the MMTR understanding task (10 epochs, 8 GPUs) -- this script:
#     UNDERSTANDING_LIST=chronos_llm/configs/understanding_train_mmtr.txt \
#     EVAL_LIST=chronos_llm/configs/understanding_test_mmtr.txt \
#     bash chronos_llm/scripts/train_understanding.sh
#   Stage 2, GRPO of the LLM LoRA starting from the SFT checkpoint: see train_grpo_understanding.sh.
#   No-reasoning control arm: add NO_REASONING=1 (answers only, no `<think>` segment).
#   LLM-only backbone baseline (series serialized as text): LLM_ONLY=1 PROC_PER_NODE=2 U_TOKEN_BUDGET=80000
#   SciTS retraining: UNDERSTANDING_LIST=chronos_llm/configs/scits_train.txt \
#     EVAL_LIST=chronos_llm/configs/scits_test.txt EVAL_BASE_DIR=data/raw/scits/Release_v1 SAMPLE_CHUNKS=8
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

UNDERSTANDING_LIST=${UNDERSTANDING_LIST:-chronos_llm/configs/understanding_train_mmtr.txt}
read_manifest UNDERSTANDING_JSONL "$UNDERSTANDING_LIST"
# In-training evaluation every EVAL_EVERY_EPOCHS epochs (0 = off). Test jsonl files whose
# `ori_path` is relative need EVAL_BASE_DIR (the shipped MMTR files are relative to the repository root,
# the SciTS files to data/raw/scits/Release_v1; absolute paths are unaffected).
EVAL_LIST=${EVAL_LIST:-chronos_llm/configs/understanding_test_mmtr.txt}
read_manifest EVAL_JSONL "$EVAL_LIST"
EVAL_BASE_DIR=${EVAL_BASE_DIR:-.}

# ---- hyper-parameters (also encoded in the run name) ----
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-5}; LR_LORA=${LR_LORA:-1e-4}; LR_QFORMER=${LR_QFORMER:-1e-4}; LR_GATE=${LR_GATE:-1e-3}
UNDERSTANDING_BS=${UNDERSTANDING_BS:-70}          # pool anchor for dynamic batching
U_TOKEN_BUDGET=${U_TOKEN_BUDGET:-20000}           # padded-token budget per rank (0 = fixed batch size)
PATCH_BUDGET=${PATCH_BUDGET:-0}; MAX_DYN_BS=${MAX_DYN_BS:-150}
LENGTH_POOL_FACTOR=${LENGTH_POOL_FACTOR:-0}
ENCODE_MAX_ROWS=${ENCODE_MAX_ROWS:-58}            # rows per Chronos-2 encode bucket
# Short-window enhancement: statistics token (loc/scale/range of the window, which instance
# normalization removes) and minimum patch count for very short single-chunk histories.
HISTORY_STATS_TOKEN=${HISTORY_STATS_TOKEN:-1}; SHORT_MIN_PATCHES=${SHORT_MIN_PATCHES:-4}
# Batch-size ladder: the LLM's linear-attention kernels are specialized per batch size, so
# dynamic batch sizes are quantized to these values to bound JIT compilations.
BS_LADDER=${BS_LADDER:-"1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150"}
SAMPLE_CHUNKS=${SAMPLE_CHUNKS:-15}; OVERVIEW_CHUNK=${OVERVIEW_CHUNK:-1}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
LOGGING_STEPS=${LOGGING_STEPS:-10}; NUM_WORKERS=${NUM_WORKERS:-8}
RUN_TS=${RUN_TS:-$(date +%Y%m%d-%H%M%S)}

# ---- optional switches ----
EXTRA=(); TAG=""
[ "${NO_REASONING:-0}" = "1" ]  && { EXTRA+=(--understanding_no_reasoning); TAG="${TAG}_nocot"; }
[ "${ANSWER_FIRST:-0}" = "1" ]  && { EXTRA+=(--understanding_answer_first); TAG="${TAG}_af"; }
[ "${LLM_ONLY:-0}" = "1" ]      && { EXTRA+=(--llm_only); TAG="${TAG}_llmonly"; }
[ "${FREEZE_TS:-0}" = "1" ]     && { EXTRA+=(--freeze_ts); TAG="${TAG}_freezets"; }
[ -n "${INIT_FROM_CHECKPOINT:-}" ] && { EXTRA+=(--init_from_checkpoint "$INIT_FROM_CHECKPOINT"); TAG="${TAG}_initckpt"; }
[ "${INIT_MERGE_REOPEN:-0}" = "1" ] && { EXTRA+=(--init_merge_reopen); TAG="${TAG}merge"; }
[ "$HISTORY_STATS_TOKEN" = "1" ] && EXTRA+=(--history_stats_token)
[ "${GRAD_CKPT:-1}" = "1" ]     && EXTRA+=(--gradient_checkpointing)
# Architecture ablations (all empty by default = the released architecture):
#   HISTORY_COMPRESSOR=pool  segment mean-pooling instead of the sliding-window Q-Former
#   FEEDBACK_COMPRESSOR=pool mean-pooled feedback instead of the feedback Q-Former
#   HISTORY_ENCODE_LAYER=k   take the soft prompt from Chronos-2 encoder layer k (early exit)
#   FEEDBACK_LLM_LAYER=k     take the feedback from LLM layer k
#   CROSS_ATTN_LAYERS=last6  inject feedback only into some encoder blocks (all|lastK|firstK|i,j,k)
#   SW_GLOBAL_QUERIES / SW_TOKEN_MAX   global-query count / soft-token budget cap
[ -n "${HISTORY_COMPRESSOR:-}" ]  && { EXTRA+=(--history_compressor "$HISTORY_COMPRESSOR"); TAG="${TAG}_hc-${HISTORY_COMPRESSOR}"; }
[ -n "${FEEDBACK_COMPRESSOR:-}" ] && { EXTRA+=(--feedback_compressor "$FEEDBACK_COMPRESSOR"); TAG="${TAG}_fc-${FEEDBACK_COMPRESSOR}"; }
[ -n "${HISTORY_ENCODE_LAYER:-}" ] && { EXTRA+=(--history_encode_layer "$HISTORY_ENCODE_LAYER"); TAG="${TAG}_hel${HISTORY_ENCODE_LAYER}"; }
[ -n "${FEEDBACK_LLM_LAYER:-}" ]  && { EXTRA+=(--feedback_llm_layer "$FEEDBACK_LLM_LAYER"); TAG="${TAG}_fll${FEEDBACK_LLM_LAYER}"; }
[ -n "${CROSS_ATTN_LAYERS:-}" ]   && { EXTRA+=(--cross_attn_layers "$CROSS_ATTN_LAYERS"); TAG="${TAG}_cal-${CROSS_ATTN_LAYERS}"; }
[ -n "${SW_GLOBAL_QUERIES:-}" ]   && { EXTRA+=(--sw_global_queries "$SW_GLOBAL_QUERIES"); TAG="${TAG}_gq${SW_GLOBAL_QUERIES}"; }
[ -n "${SW_TOKEN_MAX:-}" ]        && { EXTRA+=(--sw_token_max "$SW_TOKEN_MAX"); TAG="${TAG}_tmax${SW_TOKEN_MAX}"; }
[ -n "${SEED:-}" ]                && { EXTRA+=(--seed "$SEED"); TAG="${TAG}_s${SEED}"; }
if [ -n "${MAX_STEPS:-}" ]; then STEP_ARG=(--max_steps "$MAX_STEPS"); else STEP_ARG=(--num_train_epochs "$EPOCHS"); fi

RUN_NAME="ep${EPOCHS}_lr${LR}_lora${LR_LORA}_qf${LR_QFORMER}_gate${LR_GATE}_ubs${UNDERSTANDING_BS}_utb${U_TOKEN_BUDGET}${TAG}${BACKBONE_TAG}_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/understanding_run}/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"; archive_script "$OUTPUT_DIR" "${BASH_SOURCE[0]}"
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d-%H%M%S)_node${NODE_RANK}.log"
echo "[train_understanding] OUTPUT_DIR=$OUTPUT_DIR"

torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK --nproc_per_node=$PROC_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT:-29500} chronos_llm/train.py \
  --chronos_ckpt "$CHRONOS" --llm_path "$LLM" \
  --tsfm_backbone "$TSFM_BACKBONE" \
  --understanding_jsonl "${UNDERSTANDING_JSONL[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --bf16 --deepspeed chronos_llm/configs/ds_zero2.json \
  --learning_rate $LR --lr_lora $LR_LORA --lr_qformer $LR_QFORMER --lr_gate $LR_GATE \
  "${STEP_ARG[@]}" --warmup_ratio 0.03 \
  --understanding_bs $UNDERSTANDING_BS --understanding_token_budget $U_TOKEN_BUDGET \
  --patch_budget $PATCH_BUDGET --max_dynamic_bs $MAX_DYN_BS --dynamic_bs_ladder "$BS_LADDER" \
  --length_pool_factor $LENGTH_POOL_FACTOR --encode_max_rows $ENCODE_MAX_ROWS \
  --short_min_patches $SHORT_MIN_PATCHES \
  --encode_sample_chunks $SAMPLE_CHUNKS --encode_overview_chunk $OVERVIEW_CHUNK \
  --max_user_tokens ${MAX_USER_TOKENS:-1500} --max_tokens ${MAX_TOKENS:-4096} \
  --eval_every_epochs ${EVAL_EVERY_EPOCHS:-1} \
  --eval_understanding_jsonl "${EVAL_JSONL[@]}" --eval_understanding_base_dir "$EVAL_BASE_DIR" \
  --eval_understanding_bs ${EVAL_U_BS:-400} --eval_understanding_limit ${EVAL_U_LIMIT:-0} \
  --eval_understanding_max_new_tokens ${EVAL_U_MAX_NEW_TOKENS:-200} \
  "${EXTRA[@]}" \
  --logging_steps $LOGGING_STEPS --save_total_limit $SAVE_TOTAL_LIMIT --save_only_model \
  --dataloader_num_workers $NUM_WORKERS \
  2>&1 | tee -a "$LOG_FILE"
