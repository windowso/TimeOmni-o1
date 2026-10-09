#!/bin/bash
# Fine-tuning of the TimeOmni (SciTS, TimeOmni-v2 code base) forecasting baseline -- TimeOmni and
# UniTS are the two baselines that are fine-tuned rather than evaluated zero-shot on forecasting.
#
# Recipe:
# - Start = the released unified checkpoint epoch_10 (qwen3 router, frozen LLM, full4; trained on the
#   full SciTS dual-task data), loaded weights-only (--ckpt_path without --load_training_states),
#   then a fresh 10-epoch cosine fine-tune.
# - Data = our forecast train split converted to its JSONL format (produced by
#   convert_forecast_to_timeomni.py; the event text goes into input_text[0] -> the analyze_with_ts
#   path injects it into TimesFM via LLM + Q-former cross-attn, i.e. the same textual information our
#   model gets). valid = a 5% hold-out inside train (the real test split is untouched).
# - forecast-only (no --add_analysis); the architecture flags match the checkpoint's config.json
#   exactly (loading uses strict=False, mismatched flags would silently leave random modules).
# - Losses follow the checkpoint's protocol: quantile=1 / point=0 / pcc=0 + forecast_text CE 1.0 +
#   horizon 1.0.
#
# Usage (GPU; run from the repository root with the TimeOmni environment activated):
#   SMOKE=1 bash timeomni_ft_forecast.sh   # smoke test: valid used as train, 1 epoch
#   PROC_PER_NODE=4 bash timeomni_ft_forecast.sh
set -e
export TOKENIZERS_PARALLELISM=false

REPO=$(pwd)
TOMNI=$REPO/third_party/TimeOmni
DATA=$REPO/outputs/eval/baseline_timeomni_forecast/data
EXP_DIR=${EXP_DIR:-$REPO/outputs/eval/baseline_timeomni_forecast/exp}
INIT_CKPT=${INIT_CKPT:-$TOMNI/exp/timeomni_v2_unified_qformer_crossattn_qwen3_router_freezellm_full4/lr2e-06_COSINE_WARMUP_bs32_epochs10/exp0/epoch_10}

cd "$TOMNI"   # its code uses relative paths such as ./checkpoints and ./ds_config_zero2.json

train_epochs=${EPOCHS:-10}
learning_rate=${LR:-2e-6}
forecast_batch_size=${F_BS:-4}
comment=${COMMENT:-timeomni_ft_eventcap}
TRAIN_JSONL=$DATA/eventcap_forecast_train.jsonl
if [ "${SMOKE:-0}" = "1" ]; then
  train_epochs=1
  comment=${COMMENT:-timeomni_ft_eventcap_smoke}_smoke
  TRAIN_JSONL=$DATA/eventcap_forecast_valid.jsonl
fi

NODE_COUNT=${NODE_COUNT:-1}
PROC_PER_NODE=${PROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))}
WORLD_SIZE=$((NODE_COUNT * PROC_PER_NODE))
timestamp=$(date +"%Y%m%d_%H%M%S")

# With a single process accelerate does not inject the distributed env variables and DeepSpeed falls
# back to mpi4py discovery (crashes if not installed); export them explicitly so DeepSpeed uses env://
# initialisation (consistent with its official scripts).
if [ "$WORLD_SIZE" -eq 1 ]; then
  export RANK=${RANK:-0}
  export WORLD_SIZE=${WORLD_SIZE}
  export LOCAL_RANK=${LOCAL_RANK:-0}
  export MASTER_ADDR
  export MASTER_PORT
fi

accelerate launch \
  --mixed_precision bf16 \
  --num_processes $WORLD_SIZE \
  --num_machines $NODE_COUNT \
  --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  run_main_refactored_unified.py \
  --model TimeOmni_v2 \
  --llm_model qwen3 \
  --llm_model_path ./checkpoints/Qwen3-4B-Instruct-2507 \
  --timesfm_model_path ./checkpoints/TimesFM2.5 \
  --ts_encoder_path "$TOMNI/ts_encoder" \
  --itr 1 \
  --exp_dir "$EXP_DIR" \
  --timestamp $timestamp \
  --batch_size 32 \
  --forecast_batch_size $forecast_batch_size \
  --analysis_batch_size 4 \
  --eval_batch_size ${EVAL_BS:-16} \
  --learning_rate $learning_rate \
  --lradj COSINE_WARMUP \
  --train_epochs $train_epochs \
  --test_epoch_interval 1 \
  --max_keep_epochs ${MAX_KEEP:-10} \
  --model_comment $comment \
  --ckpt_path "$INIT_CKPT" \
  --forecast_jsonl_file_path "$TRAIN_JSONL" \
  --forecast_val_jsonl_file_path $DATA/eventcap_forecast_valid.jsonl \
  --forecast_test_jsonl_file_path $DATA/eventcap_forecast_valid.jsonl \
  --llm_low_cpu_mem_usage \
  --llm_freeze \
  --v2_point_loss_weight 0.0 \
  --v2_quantile_loss_weight 1.0 \
  --v2_pcc_loss_weight 0.0 \
  --forecast_text_loss_weight 1.0 \
  --horizon_loss_weight 1.0 \
  --use_horizon_head \
  --use_token_routing \
  --use_llm_embedding \
  --timesfm_llm_embedding_source analyze_with_ts \
  --use_ts_encoder_embedding \
  --timesfm_embedding_compressor qformer \
  --timesfm_embedding_injection cross_attn \
  --timesfm_cross_attn_gate tanh_zero \
  --qformer_num_query_tokens 16 \
  --qformer_num_layers 2 \
  --qformer_num_heads 8 \
  --qformer_dropout 0.0 \
  --timesfm_max_horizon 1024 \
  --timesfm_max_context 2048 \
  --gradient_accumulation_steps 1
