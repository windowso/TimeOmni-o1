#!/bin/bash
# DoubleCast baseline: .venv_doublecast + the DoubleCast repository on PYTHONPATH; produces the npz on the test split.
#   Run on a GPU machine: bash chronos_llm/scripts/utils/baseline_doublecast.sh
#
# Loading protocol (see the detailed header comment of baseline_doublecast.py):
#   DC_CKPT must be the HF repo id string "ServiceNow/DoubleCast", not a local directory -- a local
#   directory triggers strict full-checkpoint key matching, and the public checkpoint only contains
#   DualT5 itself (not the 14.8B text_encoder), so it always fails with Missing critical weights. The
#   paths of the text_encoder (Qwen/Qwen3-14B) / chronos base (amazon/chronos-t5-large) cannot be
#   overridden via CLI kwargs either -- DoubleCastModel.from_pretrained unconditionally drops such
#   kwargs and always reads the HF repo ids recorded in the checkpoint's own config.json. The three
#   repos must be downloaded into the same HF_HOME cache beforehand (standard
#   huggingface-cli download <repo_id>, without --local-dir); example download commands are in the
#   header of chronos_llm/eval/baseline_doublecast.py.
set -e -o pipefail
REPO=.
BL=third_party
cd "$REPO"
source "$BL/.venv_doublecast/bin/activate"
export PYTHONPATH="$BL/DoubleCast:$PYTHONPATH"; export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$BL/.hf_cache_doublecast}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUT_DIR=${OUT_DIR:-$REPO/outputs/eval/baseline_doublecast}
mkdir -p "$OUT_DIR"

python chronos_llm/eval/baseline_doublecast.py \
  --dc_ckpt "${DC_CKPT:-ServiceNow/DoubleCast}" \
  --parquet ${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet} \
  --num_samples ${NUM_SAMPLES:-20} \
  ${LIMIT:+--limit $LIMIT} \
  --device ${DEVICE:-cuda} \
  --output "$OUT_DIR/forecast_preds.npz" \
  2>&1 | tee "$OUT_DIR/infer_$(date +%Y%m%d-%H%M%S).log"

# Metrics are computed with the main environment (eval_forecast depends on chronos_llm/transformers 5 -- isolated from the 4.51 venv)
export PYTHONPATH="$REPO:$PYTHONPATH"
python -m chronos_llm.eval.eval_forecast --pred "$OUT_DIR/forecast_preds.npz" \
  --output_csv "$OUT_DIR/forecast_metrics.csv"
cat "$OUT_DIR/forecast_metrics.csv"
