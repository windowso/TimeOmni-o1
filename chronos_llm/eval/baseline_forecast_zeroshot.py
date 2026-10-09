"""Chronos-2 zero-shot forecasting baseline: original Chronos-2 weights + ``cross_states=None`` (no LLM
feedback), computing MAPE/PCC/CRPS (full horizon + ROI) on ``split=test`` of the forecasting parquet.

Inside the forecasting branch of the full model the final quantile forecast of Chronos-2 is influenced by
the LLM **only** through the gated cross-attention;
with ``cross_states=None`` (equivalent to gate=0) the Chronos-2 output == original Chronos-2 zero-shot
(``test_cross_attn_identity`` guards the gate=0 identity).

Reuses ``ForecastParquetDataset`` + ``ChronosLLMCollator`` + ``ChronosLLM._forecast_layout`` + the npz
collection/saving of ``run_forecast_infer`` + the ``eval_forecast`` metrics, so the **sample set, context
construction, target-row expansion and metric protocol are exactly those of mid_eval / eval_forecast.sh** --
the only difference is that no LLM feedback is fed in. Only Chronos-2 is loaded (not the 9B LLM); runs on a
single GPU or on CPU.

See chronos_llm/scripts/utils/baseline_forecast_zeroshot.sh for usage.
"""
import argparse
import math
import os

import torch
from torch.utils.data import Subset
from transformers import AutoTokenizer

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.eval import eval_forecast
from chronos_llm.eval.infer_forecast import run_forecast_infer
from chronos_llm.models.chronos_llm_model import ChronosLLM, _add_ts_special_tokens
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn
from chronos_llm.models.timesfm_backbone import load_timesfm3_backbone

PARQUET = "data/forecast/mmtr_forecast_corpus.parquet"
CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
TIMESFM3 = os.environ.get("TIMESFM3_PATH", "checkpoints/TimesFM3.0")
LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")


class _ZeroShotForecaster:
    """Impersonates a ChronosLLM for ``run_forecast_infer``: exposes only ``.chronos`` and
    ``generate_forecast``; the latter runs a chronos-only forward (``cross_states=None``) without the LLM."""

    def __init__(self, chronos):
        self.chronos = chronos

    @torch.no_grad()
    def generate_forecast(self, batch, horizon, max_new_tokens=0, do_sample=False, **_):
        context = batch["context"]
        # _forecast_layout is a pure function (reads no attribute of self), so the training-side logic can be
        # reused with None as a placeholder self, guaranteeing that group_ids / target-row expansion /
        # known-future covariate scatter are exactly those of training / mid_eval.
        group_ids, target_idx, fcov = ChronosLLM._forecast_layout(None, batch, context)
        ops = int(self.chronos.chronos_config.output_patch_size)
        nop = math.ceil(horizon / ops)
        qp = self.chronos(
            context, num_output_patches=nop, group_ids=group_ids,
            context_mask=batch.get("context_mask"), future_covariates=fcov,
            cross_states=None,  # <- zero-shot: no LLM feedback, numerically == original Chronos-2
        ).quantile_preds[target_idx]  # (sum of n_targets, Q, H)
        # text placeholder (run_forecast_infer does not use the text of generate mode, only quantile_preds)
        return {"quantile_preds": qp, "text": [""] * int(qp.shape[0])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chronos_ckpt", default=None,
                    help="leave empty to take the default weight path of --tsfm_backbone")
    # Swapping the backbone touches this one place only: TimesFM3Backbone exposes the same forward
    # signature and .chronos_config as Chronos2WithCrossAttn (cross_states=None == zero-shot), so
    # none of the downstream logic changes.
    ap.add_argument("--tsfm_backbone", default="chronos2", choices=["chronos2", "timesfm3"])
    ap.add_argument("--llm_path", default=LLM, help="tokenizer only (the 9B weights are not loaded)")
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_user_tokens", type=int, default=1500)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True, help="output path of the forecast npz")
    ap.add_argument("--output_csv", required=True, help="output path of the metrics csv")
    args = ap.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[baseline] no CUDA, falling back to CPU (float32)")
        device = "cpu"
    # bf16 matmul on CPU is very slow and some ops are unsupported -> float32; on GPU use bf16 (same precision as mid_eval training).
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    ckpt = args.chronos_ckpt or (TIMESFM3 if args.tsfm_backbone == "timesfm3" else CHRONOS)
    print(f"[baseline] loading {args.tsfm_backbone} {ckpt} (dtype={dtype}, device={device}) ...")
    if args.tsfm_backbone == "timesfm3":
        chronos = load_timesfm3_backbone(ckpt, dtype=dtype).to(device).eval()
    else:
        chronos = load_chronos2_with_cross_attn(ckpt, dtype=dtype).to(device).eval()

    tok = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    _add_ts_special_tokens(tok)

    ds = ForecastParquetDataset(args.parquet, tok, split=args.split, inference_mode=True,
                                max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens)
    if args.limit and args.limit < len(ds):
        ds = Subset(ds, range(args.limit))
    print(f"[baseline] {args.split} split size = {len(ds)}")

    model = _ZeroShotForecaster(chronos)
    run_forecast_infer(model, tok, ds, args.output, batch_size=args.batch_size,
                       max_new_tokens=0, device=device, num_workers=args.num_workers)
    eval_forecast.main(["--pred", args.output, "--output_csv", args.output_csv])
    print(f"[baseline] {args.tsfm_backbone} zero-shot metrics -> {args.output_csv}")


if __name__ == "__main__":
    main()
