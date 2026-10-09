#!/bin/bash
# UniTS fine-tuning baseline: the official repository + the timm shim are put on PYTHONPATH and run
# in the project environment (UniTS only needs torch/pandas/numpy + three small timm pieces, no
# extra packages -- see the comments in third_party/units_timm_shim/timm/__init__.py).
# The model has only ~8M parameters => a CPU machine is enough:
#   bash chronos_llm/scripts/utils/baseline_units.sh
# Smoke test: LIMIT=64 EPOCHS=2 with the same command.
set -e -o pipefail
REPO=.
BL=third_party
cd "$REPO"
export PYTHONPATH="$BL/UniTS:$BL/units_timm_shim:$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1

OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_units}
mkdir -p "$OUT_DIR"

python chronos_llm/eval/baseline_units.py \
  --mode ${MODE:-both} \
  --units_repo "$BL/UniTS" \
  --ckpt "$BL/UniTS/checkpoints_release/units_x128_pretrain_checkpoint.pth" \
  --parquet ${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet} \
  --output_dir "$OUT_DIR" \
  --epochs ${EPOCHS:-30} --bs ${BS:-64} --lr ${LR:-1e-4} \
  ${LIMIT:+--limit $LIMIT} \
  --device ${DEVICE:-cpu} \
  2>&1 | tee "$OUT_DIR/run_$(date +%Y%m%d-%H%M%S).log"

# metrics + per-domain breakdown (same environment, pure CPU)
python -m chronos_llm.eval.eval_forecast --pred "$OUT_DIR/forecast_preds.npz" \
  --output_csv "$OUT_DIR/forecast_metrics.csv"
cat "$OUT_DIR/forecast_metrics.csv"
if [ -z "$LIMIT" ]; then
  python chronos_llm/scripts/utils/paper_forecast_by_domain.py \
    UniTS="$OUT_DIR/forecast_preds.npz" \
    --out-csv "$REPO/outputs/eval/paper_tables/forecast_units_by_domain.csv"
fi
