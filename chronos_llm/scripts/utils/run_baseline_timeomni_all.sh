#!/bin/bash
# End-to-end driver for the TimeOmni forecasting baseline (GPU; run from the repository root with the
# TimeOmni environment activated):
#   STAGE=train  full fine-tuning (4 processes by default; override with PROC_PER_NODE)
#   STAGE=infer  run the checkpoint of a given epoch over the test split and write an npz (single GPU)
# Usage:
#   STAGE=train bash chronos_llm/scripts/utils/run_baseline_timeomni_all.sh
#   STAGE=infer EPOCH=10 bash chronos_llm/scripts/utils/run_baseline_timeomni_all.sh
set -e
REPO=.
OUT=$REPO/outputs/eval/baseline_timeomni_forecast
STAGE=${STAGE:-train}

if [ "$STAGE" = "train" ]; then
  PROC_PER_NODE=${PROC_PER_NODE:-4} bash $REPO/chronos_llm/scripts/utils/timeomni_ft_forecast.sh
elif [ "$STAGE" = "infer" ]; then
  EPOCH=${EPOCH:?need EPOCH}
  # The run directory is named after the fine-tuning job's --model_comment (see the `comment` default
  # in timeomni_ft_forecast.sh); set RUN_DIR explicitly to evaluate a different run.
  RUN_DIR=${RUN_DIR:-$(ls -d $OUT/exp/timeomni_ft_eventcap/lr*/exp0 | head -1)}
  python $REPO/chronos_llm/eval/baseline_timeomni.py \
    --model_path "$RUN_DIR/epoch_$EPOCH/pytorch_model/mp_rank_00_model_states.pt" \
    --output $OUT/preds_epoch$EPOCH.npz --batch_size ${BS:-8} ${LIMIT:+--limit $LIMIT}
else
  echo "unknown STAGE=$STAGE" >&2; exit 1
fi
