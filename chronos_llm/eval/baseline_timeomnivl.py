"""TimeOmni-VL (ICML 2026, arXiv 2602.17149, "Unified Models for Time Series Understanding
and Generation") zero-shot forecasting baseline -- an image-editing paradigm, not text generation.

Standalone script (run inside `.venv_timeomnivl`; the script inserts the TimeOmni-VL repository
path into sys.path itself, so no external PYTHONPATH is needed, and it does not import the
chronos_llm package).

## Protocol / key adaptation points

TimeOmni-VL is a **vision-centric** unified model: it does not accept numeric time-series input.
Instead the series is rendered into a "banded structured image" (one horizontal colour band per
variable; inside a band the values are laid out as a 2D grid of [number of cycles x cycle length]
and resized to a square image; values are normalised to [-1,1] with robust-MAD+tanh and mapped into
a single colour channel). Forecasting = an **image-editing (inpainting) task**: the history segment
is rendered normally, the future segment is covered by a patch-aligned black mask, and the model
(Bagel architecture: Qwen2 LLM + SigLIP ViT + VAE diffusion head) "paints" the masked region with
50 conditional diffusion steps; the official `metrics/reconstruct.py` then inverts the edited image
plus the metadata kept from rendering (means/stdev/periodicity/pad, ...) back into a numeric series.

The official code is reused instead of hand-drawn matplotlib plots (the model was trained on this
specific rendering convention; an ad-hoc line plot would be badly OOD and the result meaningless):
- `data_pipeline/bi_tsi/forecasting_adapter.py::VisionTSImageConverter`: rendering + inversion
  (this script inverts with the more compact `metrics/reconstruct.py::reconstruct_timeseries`;
  the two are algorithmically equivalent -- the latter is an independent rewrite of the former's
  inversion logic and is what the official evaluate_forecast_edits_mase.py uses).
- the 4 pure functions from `data_pipeline/gen_test_data/gen_gifteval_forecasting_test.py`
  (`random_masking`/`unpatchify`/an equivalent of `show_image`/`build_instruction_and_thinking`)
  are copied verbatim into this file (the file itself is not imported -- it hard-imports
  `from gift_eval.data import Dataset` at the top, which would drag in a GIFT-Eval evaluation
  dependency we never use; these 4 functions are self-contained and independent of the GIFT-Eval
  data set, copied without changing a character).
- `eval/generation_inference.py::load_model`/`_resolve_ckpt`: model loading (Bagel + VAE +
  tokenizer + InterleaveInferencer), reused directly rather than reimplemented.

## Mapping decisions from our data to its interface

- `past_len`/`future_len` map directly to `context_len`/`pred_len` -- we do not force their own
  GIFT-Eval short/medium/long bucketing (that is only their evaluation convention, not a hard model
  constraint).
- **Text conditioning**: `InterleaveInferencer.__call__(image=, text=, ...)` has a single text slot
  and no separate "auxiliary text" field; the structured instruction (the training-time template text
  with image size / number of cycles etc.) is concatenated with `plain_prompt` into one text and
  passed in together -- this still injects plain_prompt as a text condition; it is not a real
  "no text slot" limitation (`--use_plain_prompt 0` disables it).
- **Periodicity**: when the `freq` field ("15 min"/"1 hour"/... human-readable descriptions) can be
  mapped to a pandas frequency alias, the official `freq_to_seasonality_list` automatic period
  detection is used; when the mapping fails or the freq is unreliable (e.g. "intraday"), we fall back
  to `period=1` (no periodicity assumption; the whole segment is linearly resampled into the image)
  -- this does not affect correctness, only the "grid look" of the rendering; any exception in the
  mapping / automatic detection also falls back to period=1 so that a single sample's freq parsing
  problem never interrupts the batch.
- **Point forecasts only**: the image-editing pipeline inverts a single value per pixel and has no
  sampling / quantile uncertainty output interface (unlike Chronos-2's 21 quantile heads); the
  21-quantile array is filled by broadcasting the point trajectory, as in baseline_chattime.py
  (CRPS then reduces to a weighted absolute error).

## Usage (GPU node, .venv_timeomnivl, see chronos_llm/scripts/utils/run_baseline_timeomnivl_forecast.sh)

  .venv_timeomnivl/bin/python chronos_llm/eval/baseline_timeomnivl.py \\
    --model_dir checkpoints/TimeOmni-VL \\
    --parquet data/forecast/mmtr_forecast_corpus.parquet \\
    --output outputs/eval/baseline_timeomnivl/forecast_preds.npz \\
    [--limit 5] [--save_images_dir outputs/eval/baseline_timeomnivl/images] \\
    [--shard_idx 0 --num_shards 1]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TIMEOMNIVL_ROOT = Path(os.environ.get(
    "TIMEOMNIVL_ROOT",
    "third_party/TimeOmni-VL",
))
for _p in (TIMEOMNIVL_ROOT, TIMEOMNIVL_ROOT / "eval", TIMEOMNIVL_ROOT / "metrics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

QUANTILE_LEVELS = np.array([
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
], dtype=np.float32)

# The "freq" column is a human-readable description (the annotation convention of the MMTR
# forecasting corpus); it is mapped to a pandas frequency alias and fed to VisionTSImageConverter's
# official period detection (freq_to_seasonality_list internally calls
# pd.tseries.frequencies.to_offset). Unmappable values (e.g. "intraday") fall back to period=1.
_FREQ_MAP = {
    "5 min": "5min",
    "15 min": "15min",
    "30 min": "30min",
    "1 hour": "h",
    "1 day": "D",
    "1 week": "W",
}


def _first_channel(v):
    """Identical implementation to baseline_chattime.py: robust to nested lists (multi-channel rows use the first channel)."""
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    return np.asarray(v, dtype=np.float32)


def resolve_freq_and_period(freq_str: str):
    key = (freq_str or "").strip().lower()
    if key in _FREQ_MAP:
        return _FREQ_MAP[key], None  # period=None -> convert() detects the period automatically
    return "D", 1  # unmappable: assume no periodicity; convert() skips the freq parsing internally


# =============================================================================
# The following 3 functions are copied verbatim from data_pipeline/gen_test_data/gen_gifteval_forecasting_test.py
# (pure functions, independent of the GIFT-Eval data set / gift_eval package; the file itself is not
#  imported so as not to drag in the gift_eval dependency -- see the module docstring).
# =============================================================================
def random_masking(x, mask_ratio, noise):
    n_batch, n_tokens, n_dim = x.shape
    len_keep = int(round(n_tokens * (1 - mask_ratio)))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, n_dim))
    mask = torch.ones([n_batch, n_tokens], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_masked, mask, ids_restore


def unpatchify(x, patch_size=16):
    batch, length, dim = x.shape
    p = patch_size
    h = w = int(length ** 0.5)
    x = x.reshape(batch, h, w, p, p, 3)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(batch, 3, h * p, w * p)


def build_instruction_and_thinking(meta):
    context_len = int(meta.get("context_len", 0))
    pred_len = int(meta.get("pred_len", 0))
    periodicity = int(meta.get("periodicity", 1))
    image_size = int(meta.get("image_size", 0))
    n_vars = int(meta.get("nvars", 1))
    image_size_per_var = int(meta.get("image_size_per_var", image_size // max(1, n_vars)))
    exact_width = int(meta.get("exact_width", image_size))

    context_cycles = context_len // periodicity if periodicity > 0 else 0
    pred_cycles = pred_len // periodicity if periodicity > 0 else 0
    cycles_total = context_cycles + pred_cycles

    pred_bbox_lines = []
    pred_start_cycle = context_cycles + 1
    pred_end_cycle = context_cycles + pred_cycles

    for var_idx in range(n_vars):
        y0 = var_idx * image_size_per_var
        y1 = y0 + image_size_per_var - 1
        if cycles_total > 0:
            x0 = int((pred_start_cycle - 1) / cycles_total * exact_width)
            x1 = int(pred_end_cycle / cycles_total * exact_width) - 1
        else:
            x0, x1 = 0, exact_width - 1
        pred_bbox_lines.append(f"Var{var_idx+1}: pred bbox=[({x0}, {y0}), ({x1}, {y1})].")
    pred_bbox_text = " ".join(pred_bbox_lines)

    instruction = (
        f"<image>You are given a {image_size}x{image_size} image that encodes {n_vars} time-series variable(s) "
        f"as horizontal bands stacked from top to bottom. "
        f"Each series contains {context_cycles} cycles, "
        f"where each cycle has {periodicity} time steps (totaling {context_cycles}*{periodicity}={context_len} observations). "
        f"The right side is masked and needs to be predicted for the next {pred_cycles} cycles "
        f"(totaling {pred_cycles}*{periodicity}={pred_len} observations). "
        f"Based on the observable parts in the time-series image, please restore the masked right side."
    )
    thinking = (
        f"This image contains {n_vars} independent time-series encoded as horizontal bands, "
        f"where brighter pixels indicate larger values and darker pixels indicate smaller values. "
        f"Each series spans {cycles_total} cycles with {periodicity} time steps per cycle. "
        f"Each variable band has height image_size_per_var = image_size / n_vars = {image_size}/{n_vars} = {image_size_per_var} pixels. "
        f"Cycle width is computed as exact_width divided by the total number of cycles. "
        f"Each cycle occupies approximately image_size/cycles_total = {exact_width}/{cycles_total} pixels in width. "
        f"Prediction-region bounding-box per variable: {pred_bbox_text} "
        f"First, analyze each variable's historical pattern in the left {context_len} time steps "
        f"({context_cycles} cycles): identify trend, seasonality, value range, and anomalies. "
        f"Then, for each variable independently, extrapolate the future {pred_len} time steps ({pred_cycles} cycles) "
        f"by continuing its specific patterns and maintaining consistent brightness encoding. "
        f"Finally, output the complete full image with all series restored."
    )
    return instruction, thinking


def show_image_array(image_hwc, cur_nvars, cur_color_list):
    """Equivalent rewrite of show_image() (the original saves to disk directly; this returns a uint8 HWC
    array and lets the caller decide whether to save). Key semantics: the output starts from a pure black
    (all 0) background and each variable's band writes (v*0.5+0.5)*255 only into its "own" colour
    channel; all other channels/regions stay pure black -- it is not a single affine map over the whole
    image (that would turn blank regions into 127 grey instead of black, inconsistent with the
    inversion assumptions of metrics/reconstruct.py).
    """
    imagenet_mean = np.array([0.5, 0.5, 0.5])
    imagenet_std = np.array([0.5, 0.5, 0.5])
    cur_image = torch.zeros_like(image_hwc).cpu()
    height_per_var = image_hwc.shape[0] // max(cur_nvars, 1)
    for i in range(cur_nvars):
        cur_color = cur_color_list[i]
        cur_image[i * height_per_var: (i + 1) * height_per_var, :, cur_color] = (
            image_hwc[i * height_per_var: (i + 1) * height_per_var, :, cur_color].cpu()
            * imagenet_std[cur_color] + imagenet_mean[cur_color]
        ) * 255
    cur_image = torch.clip(cur_image, 0, 255).to(torch.uint8).numpy()
    return cur_image


def build_masked_source_image(converted_image, metadata, patch_embed, patch_size):
    """Turn the output of VisionTSImageConverter.convert() (whose image tensor already has value 0 in
    the future segment) into the "masked image" the model actually consumes: the future-segment patches
    are overwritten with -2 (pure black after mapping), with patch boundaries aligned to the model's ViT
    patches (PatchEmbed+random_masking+unpatchify pipeline, following
    gen_gifteval_forecasting_test.py::_process_window verbatim, rather than using the built-in 0 values
    of converted.image -- those map to grey, not black, inconsistent with the mask convention the model
    saw during training).
    """
    image_tensor = torch.as_tensor(converted_image).float()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    patches = patch_embed(image_tensor)
    if "mask" in metadata:
        mask_val = metadata["mask"]
        mask_src = mask_val.float() if isinstance(mask_val, torch.Tensor) else torch.as_tensor(
            np.asarray(mask_val, dtype=np.float32))
        noise = mask_src.reshape(1, -1)
    else:
        noise = torch.rand(1, patches.shape[1])
    mask_ratio = float(metadata.get("mask_ratio", 0.0))
    _, mask, _ = random_masking(patches, mask_ratio, noise)
    mask_expand = mask.unsqueeze(-1).repeat(1, 1, patch_size * patch_size * 3)
    mask_img = unpatchify(mask_expand, patch_size)
    green_bg = -torch.ones_like(image_tensor) * 2
    image_vis = image_tensor * (1 - mask_img) + green_bg * mask_img
    hwc = image_vis[0].permute(1, 2, 0)

    nvars = int(metadata.get("nvars", 1))
    color_list = metadata.get("color_list")
    if color_list is None:
        color_list = [i % 3 for i in range(nvars)]
    else:
        color_list = list(np.asarray(color_list, dtype=int))

    from PIL import Image
    arr = show_image_array(hwc, nvars, color_list)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="TimeOmni-VL checkpoint directory")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=896)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--norm_const", type=float, default=0.4)
    ap.add_argument("--align_const", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device_ids", default="0")
    ap.add_argument("--save_images_dir", default="")
    ap.add_argument("--save_images_limit", type=int, default=20,
                     help="only save the source/edit images of the first N samples for manual inspection; saving all of them is unnecessary")
    ap.add_argument("--num_timesteps", type=int, default=50)
    ap.add_argument("--cfg_text_scale", type=float, default=4.0)
    ap.add_argument("--cfg_img_scale", type=float, default=2.0)
    ap.add_argument("--cfg_interval", type=float, default=0.0)
    ap.add_argument("--timestep_shift", type=float, default=3.0)
    ap.add_argument("--cfg_renorm_min", type=float, default=0.0)
    ap.add_argument("--cfg_renorm_type", default="text_channel")
    ap.add_argument("--use_plain_prompt", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from data_pipeline.bi_tsi.forecasting_adapter import VisionTSImageConverter
    from reconstruct import reconstruct_timeseries
    import generation_inference as gi
    from timm.models.vision_transformer import PatchEmbed

    class _MArgs:
        pass

    margs = _MArgs()
    margs.base_model = args.model_dir
    margs.ckpt = gi._resolve_ckpt(args.model_dir, None)
    margs.device_ids = args.device_ids
    margs.mode = 1
    print(f"[timeomnivl] loading model from {args.model_dir} (ckpt={margs.ckpt}) ...")
    t0 = time.time()
    inferencer = gi.load_model(margs)
    print(f"[timeomnivl] model loaded in {time.time()-t0:.1f}s")

    converter = VisionTSImageConverter(
        image_size=args.image_size, patch_size=args.patch_size, num_patch_input=None,
        norm_const=args.norm_const, align_const=args.align_const, color=True,
    )
    patch_embed = PatchEmbed(args.image_size, args.patch_size, 3, embed_dim=1024)

    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[: args.limit]
    # H must be computed from the full test set before sharding, not per shard from its own max --
    # iloc[shard_idx::num_shards] is a strided slice, different shards may draw different longest
    # samples, and different H values would make P/G/R/V differ in width along axis=2 so the shard
    # npz files could not be merged.
    H = int(test["future_len"].max()) if len(test) else 0
    if args.num_shards > 1:
        test = test.iloc[args.shard_idx:: args.num_shards].reset_index(drop=True)
    N = len(test)
    print(f"[timeomnivl] test N={N} (shard {args.shard_idx}/{args.num_shards}), H={H}")

    P = np.full((N, len(QUANTILE_LEVELS), H), np.nan, dtype=np.float32)
    G = np.full((N, H), np.nan, dtype=np.float32)
    R = np.zeros((N, H), dtype=np.float32)
    V = np.zeros((N, H), dtype=bool)
    ids, ds_names = [], []

    if args.save_images_dir:
        os.makedirs(args.save_images_dir, exist_ok=True)

    n_fail = 0
    for i, row in test.iterrows():
        hist = _first_channel(row["history_values"])
        hist = hist[np.isfinite(hist)]
        fut = _first_channel(row["future_values"])
        fl = int(row["future_len"])
        cl = int(len(hist))
        ids.append(str(row["id"]))
        ds_names.append(str(row["dataset_name"]))

        t_sample = time.time()
        try:
            if cl < 2:
                raise ValueError(f"history too short (cl={cl}); cannot render")
            freq_alias, forced_period = resolve_freq_and_period(str(row.get("freq") or ""))
            series = torch.from_numpy(hist).float().view(-1, 1)  # (T, 1) single channel
            try:
                converted = converter.convert(
                    context_len=cl, pred_len=fl, series=series,
                    freq=freq_alias, period=forced_period,
                )
            except Exception as e:  # period detection / pandas parsing error: retry with period=1
                print(f"  [warn] sample {i} ({row['id']}) period detection failed ({type(e).__name__}: {e}); retrying with period=1")
                converted = converter.convert(
                    context_len=cl, pred_len=fl, series=series, freq=freq_alias, period=1,
                )

            meta = dict(converted.metadata)
            meta["exact_width"] = args.image_size  # aligned with the show_edit_data compatibility shim in gen_gifteval_forecasting_test.py

            instruction, _thinking = build_instruction_and_thinking(meta)
            text = instruction
            if args.use_plain_prompt:
                pp = str(row.get("plain_prompt") or "").strip()
                if pp:
                    text = instruction + "\n\nAdditional context you may consider: " + pp

            src_img = build_masked_source_image(converted.image, meta, patch_embed, args.patch_size)

            result = inferencer(
                image=src_img, text=text, think=False,
                cfg_text_scale=args.cfg_text_scale, cfg_img_scale=args.cfg_img_scale,
                cfg_interval=[args.cfg_interval, 1.0], timestep_shift=args.timestep_shift,
                num_timesteps=args.num_timesteps, cfg_renorm_min=args.cfg_renorm_min,
                cfg_renorm_type=args.cfg_renorm_type,
            )
            out_img = result["image"]
            if out_img is None:
                raise RuntimeError("inferencer returned no image (result['image'] is None)")

            if args.save_images_dir and i < args.save_images_limit:
                src_img.save(os.path.join(args.save_images_dir, f"{i:04d}_source.png"))
                out_img.save(os.path.join(args.save_images_dir, f"{i:04d}_edit.png"))

            _, pred_hat, _ = reconstruct_timeseries(out_img, meta, denormalize=True)
            pred = np.asarray(pred_hat, dtype=np.float32).reshape(pred_hat.shape[0], -1)[:, 0][:fl]
        except Exception as e:  # noqa: BLE001 -- a single failed sample must not take down the batch; leave NaN and count it as a failure
            n_fail += 1
            print(f"  [warn] sample {i} ({row['id']}) failed: {type(e).__name__}: {e}")
            pred = np.full(fl, np.nan, dtype=np.float32)

        P[i, :, : len(pred)] = pred[None, :]
        G[i, :fl] = fut[:fl]
        V[i, :fl] = np.isfinite(fut[:fl])
        rs, re_ = row.get("roi_start_idx"), row.get("roi_end_idx")
        if rs is not None and re_ is not None and not (pd.isna(rs) or pd.isna(re_)):
            a = max(0, int(rs) - int(row["past_len"]))
            b = min(fl, int(re_) - int(row["past_len"]))
            if b > a:
                R[i, a:b] = 1.0

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1}/{N} (last sample {time.time()-t_sample:.1f}s, failures so far {n_fail})")

    out_path = args.output
    if args.num_shards > 1:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}.shard{args.shard_idx}{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
              quantile_levels=QUANTILE_LEVELS, ids=np.array(ids, dtype=object),
              dataset_names=np.array(ds_names, dtype=object))
    print(f"[timeomnivl] npz -> {out_path} (fail {n_fail}/{N})")


if __name__ == "__main__":
    main()
