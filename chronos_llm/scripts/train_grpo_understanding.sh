#!/bin/bash
# Second training stage for understanding: GRPO on the LLM LoRA, starting from the SFT checkpoint
# (stage 1 = train_understanding.sh).
# For every question a group of G reasoning traces is sampled at temperature T; the reward is
# final-answer correctness only, advantages are group-standardized, and only the LoRA is updated.
# Data parallel over GPUs: every rank rolls out and back-propagates independently and the LoRA
# gradients are all-reduced before each optimizer step (effective batch = BS x #GPUs).
#
# Training recipe (8 GPUs x BS=8 = 64 questions per step, rollouts up to 2,048 tokens):
#   INIT_CKPT=<SFT checkpoint> FOCUS_LIST=chronos_llm/configs/grpo_focus_pools.txt \
#   REST_LIST=chronos_llm/configs/grpo_rest_pools.txt FOCUS_RATIO=0.7 POOL_BALANCE=1 \
#   G=8 TEMP=1.0 LR=1e-5 BS=8 MAX_STEPS=1000 bash chronos_llm/scripts/train_grpo_understanding.sh
# Pools are one jsonl per discipline (see make_grpo_domain_pools_v10.py); with POOL_BALANCE=1 the
# sampler draws uniformly over files, so "focus" disciplines share FOCUS_RATIO of the steps equally
# and "rest" disciplines share the remainder equally (smaller domains are drawn at a higher rate
# than their share of the rows).
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INIT_CKPT=${INIT_CKPT:?set INIT_CKPT to the SFT checkpoint directory (adapter + config)}
FOCUS_LIST=${FOCUS_LIST:-chronos_llm/configs/grpo_focus_pools.txt}
REST_LIST=${REST_LIST:-chronos_llm/configs/grpo_rest_pools.txt}
read_manifest FOCUS_FILES "$FOCUS_LIST"
read_manifest REST_FILES "$REST_LIST"
BASE_DIR=${BASE_DIR:-.}
FOCUS_RATIO=${FOCUS_RATIO:-0.7}
BS=${BS:-8}; G=${G:-8}; TEMP=${TEMP:-1.0}; LR=${LR:-1e-5}
MNT=${MAX_NEW_TOKENS:-2048}
LOGP_MB=${LOGP_MB:-2}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-50}
SAMPLE_CHUNKS=${SAMPLE_CHUNKS:-15}      # must match the SFT setting
INIT_MERGE_REOPEN=${INIT_MERGE_REOPEN:-0}   # 1 = merge the SFT LoRA into the LLM and open a fresh LoRA
ROLLOUT_MERGE=${ROLLOUT_MERGE:-1}
ALGO=${ALGO:-grpo}                      # grpo | drgrpo | rft
POOL_BALANCE=${POOL_BALANCE:-1}
DIRECT_LIST=${DIRECT_LIST:-}            # optional pool rolled out without a reasoning trace
DIRECT_RATIO=${DIRECT_RATIO:-0}
RUN_TS=${RUN_TS:-$(date +%Y%m%d-%H%M%S)}

DIRECT_ARGS=()
if [ -n "$DIRECT_LIST" ]; then
  read_manifest DIRECT_FILES "$DIRECT_LIST"
  DIRECT_ARGS=(--direct_jsonl "${DIRECT_FILES[@]}" --direct_ratio "$DIRECT_RATIO")
fi
TAG=""; [ "$INIT_MERGE_REOPEN" = "1" ] && TAG="${TAG}_mr"; [ "$ALGO" != "grpo" ] && TAG="${TAG}_${ALGO}"
[ "$POOL_BALANCE" = "1" ] && TAG="${TAG}_bal"
RUN_NAME="grpo_g${G}_t${TEMP}_lr${LR}_bs${BS}x${PROC_PER_NODE}_fr${FOCUS_RATIO}${TAG}_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_understanding_run}/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"; archive_script "$OUTPUT_DIR" "${BASH_SOURCE[0]}"
echo "[train_grpo_understanding] OUTPUT_DIR=$OUTPUT_DIR"

torchrun --nnodes=1 --nproc_per_node="$PROC_PER_NODE" --master_port="${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))}" \
  -m chronos_llm.rl.train_grpo_understanding \
  --init_ckpt "$INIT_CKPT" \
  --focus_jsonl "${FOCUS_FILES[@]}" --rest_jsonl "${REST_FILES[@]}" --focus_ratio "$FOCUS_RATIO" \
  --base_dir "$BASE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BS" --group_size "$G" --temperature "$TEMP" --lr "$LR" \
  --max_new_tokens "$MNT" --max_steps "$MAX_STEPS" --save_every_steps "$SAVE_EVERY" \
  --sample_chunks "$SAMPLE_CHUNKS" --logp_micro_bs "$LOGP_MB" \
  --init_merge_reopen "$INIT_MERGE_REOPEN" --rollout_merge "$ROLLOUT_MERGE" \
  --algo "$ALGO" --pool_balance "$POOL_BALANCE" "${DIRECT_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
