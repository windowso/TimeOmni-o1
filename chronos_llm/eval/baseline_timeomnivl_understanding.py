"""TimeOmni-VL zero-shot understanding baseline: produce per-sample jsonl in the same format as
infer_understanding on the understanding test sets (metrics are then computed with
eval_understanding.py).

Standalone script (run in a separate environment carrying TimeOmni-VL's dependencies; it does not
import the chronos_llm package).

## Protocol / key adaptations

- The official TimeOmni-VL `data_pipeline/bi_tsi/understanding_adapter.py::
  VisionTSFullSequenceConverter` renders the whole series into a structured image (the same
  family of rendering conventions as the forecasting-side VisionTSImageConverter -- striped colour
  encoding, but **no context/future split and no masking**; it purely "draws this series as one
  picture" for the model).
- **Important boundary**: the understanding QA TimeOmni-VL was trained on consists of templated
  structural questions about the rendering convention itself ("how many variables does this image
  encode", "what is the y range of variable 1", "which is brighter, period 34 or period 8" --
  essentially arithmetic on pixel coordinates/brightness, with answers generated from metadata
  formulas and no time-series content semantics); our understanding test sets are **domain
  content QA** (is this ECG atrial fibrillation / where is the bearing fault in this vibration
  signal / does this gravitational-wave segment contain an event ...). The interfaces are
  identical (image + arbitrary text question -> text answer); only the question distribution
  differs from the model's training distribution; it is run through the same interface without
  rephrasing the questions into its own templates.
- Series loading (`ori_path` relative to base_dir, npy/csv/wav loading, first non-all-NaN
  channel) is copied verbatim from `baseline_chattime_understanding.py::load_ts_first_channel`.
- **No hard length cap**: the image rendering pipeline compresses a series of any length into a
  fixed resolution with a bilinear resize (unlike ChatTime's text serialisation with its hard
  512-point truncation) -- this script does not actively downsample and leaves it to the rendering
  pipeline; very long series are drawn more compressed, not truncated.
- **No reliable freq metadata** (these test sets are sensor/audio/physiological signals without a
  pandas frequency notion) => uniform `period=1` (no periodicity assumption; convert() then skips
  the freq-string parsing entirely, see its `if not period: ...(use freq)... else:
  periodicity=period` branch).
- **The model interface does not support batched inference** (InterleaveInferencer hard-codes a
  single sample in `prepare_prompts(prompts=[text])`), so samples are run one by one; parallelism
  is obtained with **file-level** sharding via `--file_shard_idx/--file_num_shards` (the files
  are naturally independent, coarse sharding is balanced enough, and the shards' output files
  never overwrite each other, so no merge step is needed) -- simpler and more robust than row
  sharding.
- Resumable: an existing output jsonl is skipped (same convention as
  baseline_chattime_understanding.py).
- The output schema strictly matches infer_understanding.py:
  {"id","uid","dataset_name","task","scene","input_text","generated_text",
   "ground_truth","gt_result"}.

## Usage (GPU, environment with TimeOmni-VL dependencies)

  python chronos_llm/eval/baseline_timeomnivl_understanding.py \\
    --model_dir checkpoints/TimeOmni-VL \\
    --jsonl_list chronos_llm/configs/understanding_test_jsonl.txt \\
    --base_dir data/raw/scits/Release_v1 \\
    --out_dir outputs/eval/baseline_timeomnivl/understanding \\
    [--limit 5] [--file_shard_idx 0 --file_num_shards 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

TIMEOMNIVL_ROOT = Path(os.environ.get(
    "TIMEOMNIVL_ROOT",
    "third_party/TimeOmni-VL",
))
for _p in (TIMEOMNIVL_ROOT, TIMEOMNIVL_ROOT / "eval", TIMEOMNIVL_ROOT / "metrics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# =============================================================================
# Copied verbatim from baseline_chattime_understanding.py (same loading convention: npy/csv via
# numpy/pandas, wav/flac via soundfile; first non-all-NaN channel). Unlike the ChatTime baseline
# there is no max_len downsampling here -- TimeOmni-VL goes through an image resize and has no
# hard length cap.
# =============================================================================
def load_ts_first_channel(path, max_len=0):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            arr = np.load(path)
        elif ext == ".csv":
            import pandas as pd
            arr = pd.read_csv(path, header=None).apply(
                lambda c: pd.to_numeric(c, errors="coerce")).values
        elif ext in (".wav", ".flac", ".mp3", ".m4a"):
            import soundfile as sf
            arr, _ = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        else:
            raise ValueError(f"unsupported ext {ext}")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"load {path} failed: {e}")
        return np.zeros(16, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.T  # (T,C)->(C,T)
    if arr.ndim == 1:
        arr = arr[None, :]
    for row in arr:
        if np.isfinite(row).any():
            s = row
            break
    else:
        return np.zeros(16, dtype=np.float32)
    if max_len and len(s) > max_len:
        idx = np.linspace(0, len(s) - 1, max_len).round().astype(int)
        s = s[idx]
    return np.nan_to_num(s, nan=0.0)


def render_understanding_image(converter, series_1d, image_size, patch_size):
    """series_1d: (T,) single channel. Returns a PIL Image (whole-series rendering, no masking)."""
    from PIL import Image
    series = torch.from_numpy(series_1d.astype(np.float32)).view(-1, 1)  # (T,1)
    converted = converter.convert(series=series, freq="D", period=1)
    image_tensor = torch.as_tensor(converted.image).float()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    hwc = image_tensor[0].permute(1, 2, 0)

    meta = converted.metadata
    nvars = int(meta.get("nvars", 1))
    color_list = meta.get("color_list")
    if color_list is None:
        color_list = [i % 3 for i in range(nvars)]
    else:
        color_list = list(np.asarray(color_list, dtype=int))

    imagenet_mean = np.array([0.5, 0.5, 0.5])
    imagenet_std = np.array([0.5, 0.5, 0.5])
    cur_image = torch.zeros_like(hwc).cpu()
    height_per_var = hwc.shape[0] // max(nvars, 1)
    for i in range(nvars):
        c = color_list[i]
        cur_image[i * height_per_var:(i + 1) * height_per_var, :, c] = (
            hwc[i * height_per_var:(i + 1) * height_per_var, :, c].cpu()
            * imagenet_std[c] + imagenet_mean[c]
        ) * 255
    arr = torch.clip(cur_image, 0, 255).to(torch.uint8).numpy()
    return Image.fromarray(arr), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--jsonl_list", required=True)
    ap.add_argument("--base_dir", required=True, help="prefix for relative ori_path values (Release_v1 root)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--image_size", type=int, default=896)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--norm_const", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="MIMII")
    ap.add_argument("--device_ids", default="0")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--file_shard_idx", type=int, default=0)
    ap.add_argument("--file_num_shards", type=int, default=1)
    args = ap.parse_args()

    from data_pipeline.bi_tsi.understanding_adapter import VisionTSFullSequenceConverter
    import generation_inference as gi

    class _MArgs:
        pass

    margs = _MArgs()
    margs.base_model = args.model_dir
    margs.ckpt = gi._resolve_ckpt(args.model_dir, None)
    margs.device_ids = args.device_ids
    margs.mode = 1
    print(f"[timeomnivl-u] loading model from {args.model_dir} ...")
    t0 = time.time()
    inferencer = gi.load_model(margs)
    print(f"[timeomnivl-u] model loaded in {time.time()-t0:.1f}s")

    converter = VisionTSFullSequenceConverter(
        image_size=args.image_size, patch_size=args.patch_size,
        norm_const=args.norm_const, color=True,
    )

    files = [l.strip() for l in open(args.jsonl_list)
             if l.strip() and not l.strip().startswith("#")]
    files = [f for f in files if not any(e in f for e in args.exclude.split(",") if e)]
    if args.file_num_shards > 1:
        files = files[args.file_shard_idx:: args.file_num_shards]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[timeomnivl-u] shard {args.file_shard_idx}/{args.file_num_shards}: {len(files)} files")

    for jf in files:
        rows = [json.loads(l) for l in open(jf)]
        if args.limit:
            rows = rows[: args.limit]
        name = os.path.basename(jf)
        out_path = os.path.join(args.out_dir, name)
        if os.path.exists(out_path):
            print(f"[skip] {name} already exists")
            continue
        print(f"[timeomnivl-u] {name}: {len(rows)} rows")
        results = []
        n_fail = 0
        for ridx, r in enumerate(rows):
            t_row = time.time()
            ori = ((r.get("input_ts") or {}).get("original") or {}).get("ori_path") or ""
            path = ori if os.path.isabs(ori) else os.path.join(args.base_dir, ori)
            q = (r.get("input_text") or [""])[0]
            gen_text = ""
            try:
                s = load_ts_first_channel(path)
                img, _meta = render_understanding_image(converter, s, args.image_size, args.patch_size)
                result = inferencer(
                    image=img, text=q, think=args.think, understanding_output=True,
                    max_think_token_n=args.max_new_tokens, do_sample=False, text_temperature=0.3,
                )
                gen_text = (result.get("text") or "").strip()
            except Exception as e:  # noqa: BLE001 -- one failing sample must not take down the whole file; empty text is counted as a failure
                n_fail += 1
                gen_text = ""
                print(f"  [warn] {name} row {ridx} failed: {type(e).__name__}: {e}")

            results.append({
                "id": r.get("id"), "uid": r.get("uid"),
                "dataset_name": r.get("dataset_name"), "task": r.get("task"),
                "scene": r.get("scene"), "input_text": r.get("input_text"),
                "generated_text": gen_text,
                "ground_truth": r.get("gt_text"), "gt_result": r.get("gt_result"),
            })
            if (ridx + 1) % 20 == 0 or ridx == 0:
                print(f"  {ridx+1}/{len(rows)} (last row {time.time()-t_row:.1f}s, failures so far {n_fail})")

        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out_path} (fail {n_fail}/{len(rows)})")


if __name__ == "__main__":
    main()
