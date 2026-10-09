"""DoubleCast (ServiceNow, arXiv 2603.12451) zero-shot text-conditioned forecasting baseline.

NOTE: standalone script: it **must not import chronos_llm** (DoubleCast hard-pins
transformers==4.51.3, which conflicts with the 5.x used by this repository); run it inside
baselines/.venv_doublecast with PYTHONPATH pointing at the DoubleCast repository. It reads the test
split of the corpus parquet directly (same order as ForecastParquetDataset: rows with split=='test'
in their original order, row-aligned with every existing npz) and writes an npz in exactly our
format (21 quantiles, gt / roi / valid masks, ids / dataset_names); metrics are then computed with
the in-repo eval_forecast.py.

Protocol:
- univariate: target = first-channel history (DoubleCast has no multivariate / covariate
  interface);
- text = plain_prompt (the same text the other text-conditioned rows see);
- probabilistic output: num_samples sampled trajectories -> np.quantile at our 21 levels;
- when 288>64 the pipeline continues autoregressively internally; format_prompt is an idempotent
  closure that ignores the already-formatted text of the previous round (the official AR loop
  would otherwise nest-reformat the prompt);
- the prompt follows the official structured layout but omits forecast_start_date (we have no
  real timestamps, and a fake date would conflict with the real dates in plain_prompt); the
  frequency and scale_factor lines are kept.

Loading: `--dc_ckpt` must be the **HF repo id string** `"ServiceNow/DoubleCast"` (resolved from the
local cache with `HF_HUB_OFFLINE=1`); it **must not point at a local directory, and no
`text_encoder_path` / `chronos_path` override kwargs may be passed**. In
`doublecast/models/doublecast.py` (`DoubleCastPipeline.from_pretrained` /
`DoubleCastModel.from_pretrained`) there are two independent traps:
  1. `DoubleCastPipeline.from_pretrained(dc_ckpt, text_encoder_path=..., chronos_path=...)` --
     as soon as a kwarg such as `text_encoder_path` is non-empty, the "build a brand-new model
     from scratch" branch is taken (the branch the training script uses for cold starts), which
     **never calls `DoubleCastModel.from_pretrained` to load the trained parameters of the
     checkpoint**, and additionally sets `chronos_path` wrongly to `dc_ckpt` itself.
  2. Even without those kwargs, `DoubleCastModel.from_pretrained` **unconditionally discards** the
     `text_encoder_path` / `chronos_path` kwargs ("Consume build-from-scratch kwargs -- not used
     when loading a checkpoint") -- `self.text_encoder` / the chronos backbone are always rebuilt
     from **the values recorded in the checkpoint's own config.json** (HF repo ids such as
     `"Qwen/Qwen3-14B"`); local paths passed on the CLI have no effect. Moreover,
     `is_hf_hub = not Path(pretrained_model_name_or_path).is_dir()` -- a local directory is
     classified as "not HF Hub" and triggers **strict full-checkpoint key matching**; the public
     `ServiceNow/DoubleCast` on the Hub (503.4M parameters, safetensors) clearly contains only the
     DualT5 weights (chronos backbone + cross-modal injection layers) and not the 14.8B
     text_encoder, so strict loading from a local directory inevitably fails on `text_encoder.*`
     with `RuntimeError: Missing critical weights`. The only usage in the official README is
     precisely `DoubleCastPipeline.from_pretrained("ServiceNow/DoubleCast")` (pure repo id, no
     override kwargs), which corresponds to the lenient incremental-loading branch with
     `is_hf_hub=True` (missing keys only warn), the path that loads the trained weights.
  => This script therefore does not expose `--text_encoder_path` / `--chronos_path` CLI
  arguments: offline resolution relies entirely on the caller having downloaded the three repos
  `ServiceNow/DoubleCast`, `Qwen/Qwen3-14B` and `amazon/chronos-t5-large` into the same `HF_HOME`
  cache beforehand (standard `huggingface-cli download <repo_id>`, without `--local-dir`) and
  having set `HF_HOME` + `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before running.

Usage (GPU machine, .venv_doublecast, HF_HOME pointing at the pre-populated cache):
  python chronos_llm/eval/baseline_doublecast.py \
    --dc_ckpt ServiceNow/DoubleCast --parquet data/forecast/mmtr_forecast_corpus.parquet \
    --output outputs/eval/baseline_doublecast/forecast_preds.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

QUANTILE_LEVELS = np.array([
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
], dtype=np.float32)   # the same 21 levels as chronos-2 / every existing npz


def _first_channel(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    return np.asarray(v, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc_ckpt", default="ServiceNow/DoubleCast",
                    help="HF repo id (NOT a local directory! see the module docstring -- a local "
                         "directory triggers strict full-checkpoint loading, which always fails with "
                         "Missing critical weights on the public checkpoint). Download this repo "
                         "together with the text_encoder_path/chronos_path recorded in its config.json "
                         "into the $HF_HOME cache with huggingface-cli download beforehand, and set "
                         "HF_HUB_OFFLINE=1 so that it is resolved offline here.")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from doublecast.models.doublecast import DoubleCastPipeline

    # Do not pass text_encoder_path/chronos_path kwargs -- doing so triggers the "build from
    # scratch" branch and the trained checkpoint weights are skipped entirely (see module
    # docstring). The text_encoder / chronos backbone paths are fully determined by the config.json
    # shipped with dc_ckpt (HF repo ids), resolved from the offline HF_HOME cache.
    pipe = DoubleCastPipeline.from_pretrained(
        args.dc_ckpt, device_map=args.device, torch_dtype=torch.bfloat16)

    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit]
    H = int(test["future_len"].max())
    N = len(test)
    print(f"[doublecast] test N={N}, H={H}, num_samples={args.num_samples}")

    P = np.full((N, len(QUANTILE_LEVELS), H), np.nan, dtype=np.float32)
    G = np.full((N, H), np.nan, dtype=np.float32)
    R = np.zeros((N, H), dtype=np.float32)
    V = np.zeros((N, H), dtype=bool)
    ids, ds_names = [], []

    for i, row in test.iterrows():
        hist = _first_channel(row["history_values"])
        fut = _first_channel(row["future_values"])
        fl = int(row["future_len"])
        text = str(row.get("plain_prompt") or row.get("prompt") or "")
        freq = str(row.get("freq") or "")

        def fmt(context, past_target, scale, _text=text, _freq=freq):
            # Idempotent: ignore the incoming context (from the second AR window on it is the
            # previous round's formatted output) and always use the original text
            return (f"<info>\nfrequency={_freq}\nscale_factor={float(scale.item()):.4f}\n"
                    f"</info>\n\n<context>\n{_text}\n</context>")

        ctx = torch.tensor(hist, dtype=torch.bfloat16).flatten()  # NaN is treated as missing by the chronos tokenizer
        with torch.no_grad():
            samples = pipe.predict(
                context=ctx, text_context=text, prediction_length=fl,
                num_samples=args.num_samples, limit_prediction_length=False,
                format_prompt=fmt,
            )                                        # (1, S, fl) float32 cpu
        s = samples.squeeze(0).numpy()               # (S, fl)
        q = np.quantile(s, QUANTILE_LEVELS.astype(np.float64), axis=0)  # (21, fl)
        P[i, :, :fl] = q
        G[i, :fl] = fut[:fl]
        V[i, :fl] = np.isfinite(fut[:fl])
        rs, re = row.get("roi_start_idx"), row.get("roi_end_idx")
        if rs is not None and re is not None and not (pd.isna(rs) or pd.isna(re)):
            a = max(0, int(rs) - int(row["past_len"]))
            b = min(fl, int(re) - int(row["past_len"]))
            if b > a:
                R[i, a:b] = 1.0
        ids.append(str(row["id"]))
        ds_names.append(str(row["dataset_name"]))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
             quantile_levels=QUANTILE_LEVELS, ids=np.array(ids, dtype=object),
             dataset_names=np.array(ds_names, dtype=object))
    print(f"[doublecast] npz -> {args.output}")


if __name__ == "__main__":
    main()
