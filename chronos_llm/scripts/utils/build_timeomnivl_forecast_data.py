"""Convert our forecasting test set (corpus parquet, split=test) into the jsonl + TS-image +
metadata format required by TimeOmni-VL, so that its eval/generation_inference.py can consume it
directly.

The TimeOmni-VL module gen_gifteval_forecasting_test.py, which depends on the gift_eval package,
is not imported as a whole (an unused top-level `from gift_eval.data import Dataset` would create a
brittle coupling; gift_eval is not installed here); only its VisionTSImageConverter (the actual
TS<->image encoder -- the pixel-encoding logic is not re-invented) is reused, and the remaining pure
functions around the pixel encoding (period selection / capacity check / instruction template /
image rendering) are copied into this file verbatim, with logic identical to
data_pipeline/gen_test_data/gen_gifteval_forecasting_test.py of the TimeOmni-VL repository
(transcribed and adapted from that file; see the source header).

Usage (pure CPU image encoding, no GPU needed; use the TimeOmni-VL specific venv with timm/einops installed):
  python \
    chronos_llm/scripts/utils/build_timeomnivl_forecast_data.py \
    --parquet data/forecast/mmtr_forecast_corpus.parquet \
    --output-root outputs/eval/baseline_timeomnivl/forecast_data \
    --limit 5   # smoke-test on a few samples first, then drop this argument for the full test set
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
from PIL import Image
from timm.models.vision_transformer import PatchEmbed

TIMEOMNIVL_REPO = "third_party/TimeOmni-VL"
if TIMEOMNIVL_REPO not in sys.path:
    sys.path.insert(0, TIMEOMNIVL_REPO)
from data_pipeline.bi_tsi.forecasting_adapter import VisionTSImageConverter  # noqa: E402

# ---- The following is copied from TimeOmni-VL/data_pipeline/gen_test_data/gen_gifteval_forecasting_test.py ----
# (only the pure functions unrelated to GIFT-Eval Dataset loading are kept: period selection /
# capacity check / instruction template / rendering)

POSSIBLE_SEASONALITIES = {
    "S": [3600], "T": [1440], "H": [24, 168], "D": [7], "B": [5], "W": [52], "M": [12], "Q": [4],
}

FREQ_NAME_ALIASES = {
    "s": "S", "sec": "S", "second": "S", "seconds": "S", "secondly": "S",
    "t": "T", "min": "T", "mins": "T", "minute": "T", "minutes": "T", "minutely": "T",
    "h": "H", "hr": "H", "hrs": "H", "hour": "H", "hours": "H", "hourly": "H",
    "d": "D", "day": "D", "days": "D", "daily": "D",
    "b": "B", "business": "B", "businessday": "B", "businessdays": "B",
    "w": "W", "week": "W", "weeks": "W", "weekly": "W",
    "m": "M", "month": "M", "months": "M", "monthly": "M",
    "q": "Q", "quarter": "Q", "quarters": "Q", "quarterly": "Q",
}

# Supplementary mapping from our corpus's own freq wording ("15 min" / "intraday" ...) to the
# standard "<n><unit>"; "intraday" rows have a modal history-timestamp interval of 5 minutes.
CORPUS_FREQ_ALIASES = {"intraday": "5min"}


def _norm_freq_str(freq_str: str) -> str:
    base_freq = str(freq_str).split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def _normalize_freq_for_period(freq_str: str) -> str:
    raw = CORPUS_FREQ_ALIASES.get(str(freq_str).strip().lower(), str(freq_str).strip())
    if not raw:
        raise ValueError("Empty freq string.")
    try:
        offset = pd.tseries.frequencies.to_offset(raw)
        base = _norm_freq_str(offset.name)
        if base.lower() == "min":
            base = "T"
        if offset.n and offset.n != 1:
            return f"{offset.n}{base}"
        return base
    except Exception:
        pass
    lower = raw.lower()
    if lower in FREQ_NAME_ALIASES:
        return FREQ_NAME_ALIASES[lower]
    match = re.match(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$", raw)
    if match:
        multiplier = match.group(1)
        unit = match.group(2).lower()
        base = FREQ_NAME_ALIASES.get(unit)
        if base:
            return f"{multiplier}{base}"
    raise ValueError(f"Unrecognized frequency string: {freq_str}")


def _split_freq(freq_str: str):
    base_freq_char = "".join([c for c in str(freq_str) if c.isalpha()]).upper()
    if base_freq_char == "MIN":
        base_freq_char = "T"
    if base_freq_char == "Y":
        base_freq_char = "A"
    match = re.match(r"(\d+)", str(freq_str))
    multiplier = int(match.group(1)) if match else 1
    return base_freq_char, multiplier


def _candidate_periods(freq_str: str, total_len: int):
    if not freq_str or total_len <= 0:
        return []
    base_freq_char, multiplier = _split_freq(freq_str)
    suggestions = POSSIBLE_SEASONALITIES.get(base_freq_char)
    if not suggestions:
        return []
    periods = []
    for base_period_steps in suggestions:
        if multiplier <= 0:
            continue
        period = int(base_period_steps // multiplier)
        if period > 0:
            periods.append(period)
    return periods


def analyze_capacity_constraints(context_len, pred_len, image_size, patch_size, period,
                                  align_const, nvars=None, verbose=False):
    total_len = context_len + pred_len
    if total_len <= 0 or period <= 0:
        return False
    height_ok = True
    if nvars is not None and nvars > 0:
        height_per_var = image_size / nvars
        if height_per_var < period:
            height_ok = False
    input_ratio = context_len / total_len
    num_total_patch = image_size // patch_size
    num_patch_in = max(1, int(input_ratio * num_total_patch * align_const))
    num_patch_out = max(1, num_total_patch - num_patch_in)
    global_capacity = image_size * period
    capacity_ctx = num_patch_in * patch_size * period
    capacity_pred = num_patch_out * patch_size * period
    is_all_safe = (total_len <= global_capacity and context_len <= capacity_ctx
                   and pred_len <= capacity_pred and height_ok)
    if verbose:
        print(f"[capacity] {'PASS' if is_all_safe else 'FAIL'} total={total_len} "
              f"ctx={context_len} pred={pred_len} period={period}")
    return is_all_safe


def _select_period(freq, context_len, pred_len, image_size, patch_size, align_const, nvars):
    """Difference from the original TimeOmni-VL gen_gifteval_forecasting_test.py: the original draws
    one period with random.choice and skips the whole row if it does not fit; here all candidate
    periods are traversed from smallest to largest and the first one satisfying the capacity
    constraint is taken. GIFT-Eval series are generally long enough that a randomly drawn period
    usually fits, whereas many samples in our corpus have a shorter context (e.g. 120 points at 1H
    hold only 5 daily cycles; drawing 168 (weekly) would discard the row because
    max_context_cycles=0<pred_cycles). Traversing stops an unlucky random draw from discarding a
    sample that would fit."""
    total_len = context_len + pred_len
    period_candidates = _candidate_periods(freq, total_len)
    if not period_candidates:
        return None, None
    for period in sorted(period_candidates):
        if analyze_capacity_constraints(context_len, pred_len, image_size, patch_size, period,
                                         align_const, nvars=nvars):
            return period, period_candidates
    return None, None


def _as_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _pad_series_to_length(series: torch.Tensor, target_len: int, pad_left: bool) -> torch.Tensor:
    if series.shape[0] >= target_len:
        return series
    pad_len = target_len - series.shape[0]
    if pad_left:
        pad_value = series[:1].repeat(pad_len, 1)
        return torch.cat([pad_value, series], dim=0)
    pad_value = series[-1:].repeat(pad_len, 1)
    return torch.cat([series, pad_value], dim=0)


def random_masking(x, mask_ratio, noise):
    n_batch, n_tokens, n_dim = x.shape
    len_keep = int(round(n_tokens * (1 - mask_ratio)))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones([n_batch, n_tokens])
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return mask


def unpatchify(x, patch_size=16):
    batch, length, dim = x.shape
    p = patch_size
    h = w = int(length ** 0.5)
    x = x.reshape(batch, h, w, p, p, 3)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(batch, 3, h * p, w * p)


def show_image(image, cur_nvars, cur_color_list, save_path):
    imagenet_mean = np.array([0.5, 0.5, 0.5])
    imagenet_std = np.array([0.5, 0.5, 0.5])
    cur_image = torch.zeros_like(image).cpu()
    height_per_var = image.shape[0] // max(cur_nvars, 1)
    for i in range(cur_nvars):
        cur_color = cur_color_list[i]
        cur_image[i * height_per_var:(i + 1) * height_per_var, :, cur_color] = (
            image[i * height_per_var:(i + 1) * height_per_var, :, cur_color].cpu()
            * imagenet_std[cur_color] + imagenet_mean[cur_color]
        ) * 255
    cur_image = torch.clip(cur_image, 0, 255).to(torch.uint8).numpy()
    Image.fromarray(cur_image).save(save_path)


def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


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
        pred_bbox_lines.append(f"Var{var_idx + 1}: pred bbox=[({x0}, {y0}), ({x1}, {y1})].")
    pred_bbox_text = " ".join(pred_bbox_lines)

    instruction = (
        f"<image>You are given a {image_size}x{image_size} image that encodes {n_vars} time-series "
        f"variable(s) as horizontal bands stacked from top to bottom. "
        f"Each series contains {context_cycles} cycles, where each cycle has {periodicity} time steps "
        f"(totaling {context_cycles}*{periodicity}={context_len} observations). "
        f"The right side is masked and needs to be predicted for the next {pred_cycles} cycles "
        f"(totaling {pred_cycles}*{periodicity}={pred_len} observations). "
        f"Based on the observable parts in the time-series image, please restore the masked right side."
    )
    thinking = (
        f"This image contains {n_vars} independent time-series encoded as horizontal bands, "
        f"where brighter pixels indicate larger values and darker pixels indicate smaller values. "
        f"Each series spans {cycles_total} cycles with {periodicity} time steps per cycle. "
        f"Each variable band has height image_size_per_var = image_size / n_vars = "
        f"{image_size}/{n_vars} = {image_size_per_var} pixels. "
        f"Cycle width is computed as exact_width divided by the total number of cycles. "
        f"Each cycle occupies approximately image_size/cycles_total = {exact_width}/{cycles_total} "
        f"pixels in width. Prediction-region bounding-box per variable: {pred_bbox_text} "
        f"First, analyze each variable's historical pattern in the left {context_len} time steps "
        f"({context_cycles} cycles): identify trend, seasonality, value range, and anomalies. "
        f"Then, for each variable independently, extrapolate the future {pred_len} time steps "
        f"({pred_cycles} cycles) by continuing its specific patterns and maintaining consistent "
        f"brightness encoding. Finally, output the complete full image with all series restored."
    )
    return instruction, thinking


# ---- Above: copied from the TimeOmni-VL repository; below: our own adaptation logic (data source replaced by this project's parquet) ----


def process_row(row, row_id, output_root: Path, converter, patch_embed_layer,
                 image_size, patch_size, align_const):
    freq_norm = _normalize_freq_for_period(row["freq"])
    context_arr = _as_2d(np.asarray(row["history_values"], dtype=np.float32))
    pred_arr = _as_2d(np.asarray(row["future_values"], dtype=np.float32))
    context_len, pred_len = context_arr.shape[-1], pred_arr.shape[-1]
    nvars = context_arr.shape[0]

    period, period_candidates = _select_period(
        freq_norm, context_len, pred_len, image_size, patch_size, align_const, nvars)
    if period is None:
        return None, f"no valid period (freq={row['freq']}->{freq_norm}, ctx={context_len}, pred={pred_len})"

    pred_len_rounded = max(int(period), int(math.ceil(pred_len / period) * period))
    pred_cycles = pred_len_rounded // period
    max_context_cycles = context_len // period
    if max_context_cycles < pred_cycles:
        return None, f"context shorter than pred cycles ({max_context_cycles} < {pred_cycles})"
    context_cycles = 2 * pred_cycles if max_context_cycles >= 2 * pred_cycles else max_context_cycles
    context_len_real = context_cycles * period
    context_arr = context_arr[:, -context_len_real:]

    if not analyze_capacity_constraints(context_len_real, pred_len_rounded, image_size, patch_size,
                                         period, align_const, nvars=nvars):
        return None, "capacity fail"

    context_series = torch.from_numpy(context_arr.T).float()
    pred_series_raw = torch.from_numpy(pred_arr.T).float()
    pred_series = _pad_series_to_length(pred_series_raw, pred_len_rounded, pad_left=False)

    converted = converter.convert(context_len=context_len_real, pred_len=pred_len_rounded,
                                   series=context_series, freq=freq_norm, period=period)
    full_series = torch.cat([context_series, pred_series], dim=0)
    full_image = converter.render_full_groundtruth_image(full_series, converted)

    sample_dir = output_root / f"sample_{row_id:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_tensor = torch.as_tensor(converted.image).float()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    patches = patch_embed_layer(image_tensor)
    if "mask" in converted.metadata:
        noise = einops.repeat(torch.as_tensor(converted.metadata["mask"]), "1 l -> n l", n=1)
    else:
        noise = torch.rand(1, patches.shape[1])
    mask_ratio = float(converted.metadata.get("mask_ratio", 0.0))
    mask = random_masking(patches, mask_ratio, noise)
    mask_expand = mask.unsqueeze(-1).repeat(1, 1, patch_size * patch_size * 3)
    mask_img = unpatchify(mask_expand, patch_size)
    green_bg = -torch.ones_like(image_tensor) * 2
    image_vis = image_tensor * (1 - mask_img) + green_bg * mask_img
    image_vis_hwc = image_vis[0].permute(1, 2, 0)

    nvars_meta = int(converted.metadata.get("nvars", 1))
    color_list = converted.metadata.get("color_list") or [0] * nvars_meta
    color_list = list(np.asarray(color_list, dtype=int))

    show_image(image_vis_hwc, nvars_meta, color_list, sample_dir / "image_mask.png")
    full_tensor = torch.as_tensor(full_image).float()
    if full_tensor.dim() == 3:
        full_tensor = full_tensor.unsqueeze(0)
    show_image(full_tensor[0].permute(1, 2, 0), nvars_meta, color_list, sample_dir / "image_full.png")

    meta = dict(converted.metadata)
    meta["freq"] = freq_norm
    meta["exact_width"] = image_size
    meta["generated_period"] = period
    meta["period_candidates"] = period_candidates
    meta["source_id"] = str(row["id"])
    meta["source_dataset"] = str(row["dataset_name"])
    meta["pred_len_real"] = pred_len
    meta["pred_len_padded"] = pred_len_rounded
    meta["context_len_real"] = context_len_real
    meta["nvars_total"] = nvars

    instruction, thinking = build_instruction_and_thinking(meta)
    meta["instruction"] = instruction
    meta["thinking"] = thinking
    (sample_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=_to_json_safe), encoding="utf-8")
    np.savez(sample_dir / "series.npz", context=context_series.numpy(),
             pred=pred_series_raw.numpy(), series=torch.cat([context_series, pred_series_raw], dim=0).numpy())

    row_out = {
        "dataset": str(row["dataset_name"]),
        "source_id": str(row["id"]),
        "instruction": instruction,
        "thinking": thinking,
        "source_image": str(sample_dir / "image_mask.png"),
        "target_image": str(sample_dir / "image_full.png"),
        "metadata": str(sample_dir / "metadata.json"),
    }
    return row_out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--norm-const", type=float, default=0.4)
    ap.add_argument("--align-const", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="0=full test set; >0 processes only the first N rows (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit]
    print(f"[build_timeomnivl_data] {len(test)} test rows (limit={args.limit or 'ALL'})")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    converter = VisionTSImageConverter(image_size=args.image_size, patch_size=args.patch_size,
                                        num_patch_input=None, norm_const=args.norm_const,
                                        align_const=args.align_const, color=True)
    patch_embed_layer = PatchEmbed(args.image_size, args.patch_size, 3, embed_dim=1024)

    rows, n_skip = [], 0
    for i, row in test.iterrows():
        row_out, err = process_row(row, i, output_root, converter, patch_embed_layer,
                                    args.image_size, args.patch_size, args.align_const)
        if row_out is None:
            n_skip += 1
            print(f"  [skip] row {i} ({row['id']}): {err}")
        else:
            rows.append(row_out)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test)} done, {n_skip} skipped so far")

    jsonl_path = output_root / "forecast_samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build_timeomnivl_data] wrote {len(rows)} rows ({n_skip} skipped) -> {jsonl_path}")


if __name__ == "__main__":
    main()
