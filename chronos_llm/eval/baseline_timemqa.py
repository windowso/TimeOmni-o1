"""Time-MQA (ACL 2025, arXiv 2503.01875) zero-shot understanding baseline.

Model = hf.co/Time-MQA/Qwen-2.5-7B: a **LoRA adapter** (r16, base =
unsloth/qwen2.5-7b-unsloth-bnb-4bit, a 4bit bnb-quantised base), further trained on the 200k
QA pairs of TSQA. Loaded exactly as released (4bit base + adapter, precision unchanged, not
merged -- under QLoRA semantics the adapter was trained against the 4bit base, so this is the
most faithful way to load it).

## Protocol (official code not released -- the code page says "available soon" and the TSQA
dataset is gated; the format follows the paper)

- Paper training format: `<QUE> {Question} <ANS> {Answer} </END>` (the Qwen variant ends with
  <|endoftext|>) => inference prompt = `<QUE> {question} <ANS>`, let the model continue the
  answer, cut at `</END>`/eos.
- The series is embedded in the question as **comma-separated numeric text** (paper example
  "The input Time Series are [Time Series Data Points]."). Our template:
  `The input Time Series are [v1, v2, ...]. {original question text}`.
- Values keep 4 significant digits (text serialisation costs ~4-6 tokens/point); the series is
  uniformly downsampled to `--max_points` (default 512, the same downsampling as the ChatTime
  baseline).
- Same loading convention as the other baselines (first non-all-NaN channel), same output
  schema, same test sets.

## Usage (GPU, environment with transformers 4.49 + peft)

  python chronos_llm/eval/baseline_timemqa.py \\
    --adapter_dir checkpoints/Time-MQA-Qwen2.5-7B \\
    --base_dir_model checkpoints/qwen2.5-7b-unsloth-bnb-4bit \\
    --jsonl_list chronos_llm/configs/understanding_test_jsonl.txt \\
    --base_dir data/raw/scits/Release_v1 \\
    --out_dir outputs/eval/baseline_timemqa/understanding [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch


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
            arr, _ = sf.read(path, dtype="float32", always_2d=True)
        else:
            raise ValueError(f"unsupported ext {ext}")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"load {path} failed: {e}")
        return np.zeros(16, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.T
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


def serialize(series: np.ndarray) -> str:
    return ", ".join(f"{v:.4g}" for v in series.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--base_dir_model", required=True, help="local path of the unsloth qwen2.5-7b bnb-4bit base model")
    ap.add_argument("--jsonl_list", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--max_points", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="MIMII")
    ap.add_argument("--file_shard_idx", type=int, default=0)
    ap.add_argument("--file_num_shards", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[timemqa] loading base {args.base_dir_model} + adapter {args.adapter_dir} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)  # the adapter repo ships the full tokenizer
    base = AutoModelForCausalLM.from_pretrained(args.base_dir_model, device_map="cuda:0")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    print(f"[timemqa] loaded in {time.time()-t0:.1f}s")

    files = [l.strip() for l in open(args.jsonl_list)
             if l.strip() and not l.strip().startswith("#")]
    files = [f for f in files if not any(e in f for e in args.exclude.split(",") if e)]
    if args.file_num_shards > 1:
        files = files[args.file_shard_idx:: args.file_num_shards]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[timemqa] shard {args.file_shard_idx}/{args.file_num_shards}: {len(files)} files")

    for jf in files:
        rows = [json.loads(l) for l in open(jf)]
        if args.limit:
            rows = rows[: args.limit]
        name = os.path.basename(jf)
        out_path = os.path.join(args.out_dir, name)
        if os.path.exists(out_path):
            print(f"[skip] {name} already exists")
            continue
        print(f"[timemqa] {name}: {len(rows)} rows")
        results = []
        n_fail = 0
        for ridx, r in enumerate(rows):
            t_row = time.time()
            ori = ((r.get("input_ts") or {}).get("original") or {}).get("ori_path") or ""
            path = ori if os.path.isabs(ori) else os.path.join(args.base_dir, ori)
            q = (r.get("input_text") or [""])[0]
            gen_text = ""
            try:
                s = load_ts_first_channel(path, max_len=args.max_points)
                prompt = f"<QUE> The input Time Series are [{serialize(s)}]. {q} <ANS>"
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
                with torch.no_grad():
                    out = model.generate(
                        ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id)
                gen_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                gen_text = gen_text.split("</END>")[0].strip()
            except Exception as e:  # noqa: BLE001
                n_fail += 1
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
