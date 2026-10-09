#!/bin/bash
# TSFM-only baseline for understanding: Chronos-2 encoder + one classification head per
# fixed-label subtask (the subtasks with free-text answers or per-question options cannot be
# attempted by this baseline). Single GPU.
#   bash chronos_llm/scripts/train_headcls.sh
#   MAX_STEPS=20 LIMIT_TEST=200 EPOCHS=1 bash chronos_llm/scripts/train_headcls.sh   # smoke test
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TRAIN_LIST=${TRAIN_LIST:-chronos_llm/configs/understanding_train_mmtr.txt}
TEST_LIST=${TEST_LIST:-chronos_llm/configs/understanding_test_mmtr.txt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/baseline_headcls}
EPOCHS=${EPOCHS:-10}
BS=${BS:-96}
INFER_BS=${INFER_BS:-256}
LR_CHRONOS=${LR_CHRONOS:-2e-5}
LR_HEAD=${LR_HEAD:-1e-4}
MAX_STEPS=${MAX_STEPS:-0}
LIMIT_TRAIN=${LIMIT_TRAIN:-0}
LIMIT_TEST=${LIMIT_TEST:-0}
NUM_WORKERS=${NUM_WORKERS:-8}

python -m chronos_llm.train_headcls \
  --chronos_path "$CHRONOS" \
  --train_list "$TRAIN_LIST" --test_list "$TEST_LIST" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" --bs "$BS" --infer_bs "$INFER_BS" \
  --lr_chronos "$LR_CHRONOS" --lr_head "$LR_HEAD" \
  --max_steps "$MAX_STEPS" --limit_train "$LIMIT_TRAIN" --limit_test "$LIMIT_TEST" \
  --num_workers "$NUM_WORKERS"
