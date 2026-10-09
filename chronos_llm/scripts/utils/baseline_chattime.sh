#!/bin/bash
# ChatTime baseline: .venv_doublecast + the ChatTime repository on PYTHONPATH.
# Two sides: STAGE=forecast|understanding|both (default both) --
#   forecast: produces the npz on the test split; understanding: per-sample jsonl + metrics CSV on the understanding test sets.
set -e -o pipefail
REPO=.
BL=third_party
CK=checkpoints
cd "$REPO"
source "$BL/.venv_doublecast/bin/activate"
export PYTHONPATH="$BL/ChatTime:$PYTHONPATH"; export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_chattime}
STAGE=${STAGE:-both}
mkdir -p "$OUT_DIR"

if [ "$STAGE" = "forecast" ] || [ "$STAGE" = "both" ]; then
  python chronos_llm/eval/baseline_chattime.py \
    --ckpt "$CK/ChatTime-1-7B-Chat" \
    --parquet ${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet} \
    ${LIMIT:+--limit $LIMIT} \
    --output "$OUT_DIR/forecast_preds.npz" \
    2>&1 | tee "$OUT_DIR/infer_$(date +%Y%m%d-%H%M%S).log"
fi

if [ "$STAGE" = "understanding" ] || [ "$STAGE" = "both" ]; then
  python chronos_llm/eval/baseline_chattime_understanding.py \
    --ckpt "$CK/ChatTime-1-7B-Chat" \
    --jsonl_list chronos_llm/configs/understanding_test_jsonl.txt \
    --base_dir data/raw/scits/Release_v1 \
    --out_dir "$OUT_DIR/understanding" \
    ${U_LIMIT:+--limit $U_LIMIT} \
    ${U_BS:+--batch_size $U_BS} \
    2>&1 | tee "$OUT_DIR/infer_understanding_$(date +%Y%m%d-%H%M%S).log"
fi

export PYTHONPATH="$REPO:$PYTHONPATH"
if [ -f "$OUT_DIR/forecast_preds.npz" ]; then
  python -m chronos_llm.eval.eval_forecast --pred "$OUT_DIR/forecast_preds.npz" \
    --output_csv "$OUT_DIR/forecast_metrics.csv"
  cat "$OUT_DIR/forecast_metrics.csv"
fi
if [ -d "$OUT_DIR/understanding" ]; then
  python -m chronos_llm.eval.eval_understanding "$OUT_DIR/understanding" \
    -o "$OUT_DIR/understanding_metrics.csv"
  cat "$OUT_DIR/understanding_metrics.csv"
fi
