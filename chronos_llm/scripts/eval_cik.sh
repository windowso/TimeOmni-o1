#!/bin/bash
# Zero-shot evaluation on CiK (Context is Key). Only the `generate` mode applies (CiK has no
# annotated traces to teacher-force). The official RCRPS is computed afterwards on CPU by
# scripts/utils/eval_cik_rcrps.py, which imports the official benchmark code.
#
#   python chronos_llm/scripts/utils/build_cik_eval_parquet.py --cik_parquet data/raw/CiK/data/test-00000-of-00001.parquet --out data/external/cik_eval.parquet
#   bash chronos_llm/scripts/eval_cik.sh <MODEL_PATH> [PARQUET] [OUT_DIR]
#   python chronos_llm/scripts/utils/eval_cik_rcrps.py --pred <OUT_DIR>/cik_preds.npz \
#       --cik_parquet data/raw/CiK/data/test-00000-of-00001.parquet --cik_code_dir third_party/context-is-key-forecasting \
#       --out <OUT_DIR>/rcrps_summary.json
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_PATH=${1:?usage: eval_cik.sh <MODEL_PATH> [PARQUET] [OUT_DIR]}
PARQUET=${2:-data/external/cik_eval.parquet}
OUT_DIR=${3:-outputs/eval/cik}
mkdir -p "$OUT_DIR"

torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK --nproc_per_node=$PROC_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))} \
  chronos_llm/eval/infer_forecast.py \
  --model_path "$MODEL_PATH" --parquet "$PARQUET" \
  --split test --limit ${LIMIT:-400} \
  --batch_size ${BATCH_SIZE:-32} --max_new_tokens ${MAX_NEW_TOKENS:-320} \
  --num_workers ${NUM_WORKERS:-4} --no_merge --mode generate \
  --output "$OUT_DIR/cik_preds.npz"
echo "CiK inference done -> $OUT_DIR/cik_preds.npz"
