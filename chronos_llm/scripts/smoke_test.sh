#!/bin/bash
# GPU smoke test: real Chronos-2 + Qwen3.5-9B, six optimizer steps on a handful of samples,
# to verify that the full model trains end to end (finite losses, adapter saved). Single GPU.
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
JSONL=${JSONL:-data/understanding/veritime/veritime_test.jsonl}

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT:-29503} chronos_llm/train.py \
  --chronos_ckpt "$CHRONOS" --llm_path "$LLM" \
  --tsfm_backbone "$TSFM_BACKBONE" \
  --forecast_parquet "$PARQUET" \
  --understanding_jsonl "$JSONL" \
  --output_dir "${OUTPUT_DIR:-outputs/smoke}" \
  --bf16 --gradient_checkpointing \
  --learning_rate 1e-4 --max_steps 6 \
  --understanding_bs 2 --forecast_bs 1 \
  --understanding_max_context 2048 --forecast_max_context 8192 \
  --max_user_tokens 512 --max_tokens 1536 \
  --logging_steps 1 --dataloader_num_workers 2
