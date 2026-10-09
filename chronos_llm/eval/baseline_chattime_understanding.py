"""ChatTime-1-7B understanding-side zero-shot baseline: produce per-sample jsonl in the same
format as infer_understanding on the understanding test sets (metrics are then computed with
eval_understanding.py).

Standalone script (run with the ChatTime repository on PYTHONPATH; it does not import chronos_llm).
Protocol = ChatTime's official analysis pipeline (Discretizer 10K-bin discretisation ->
###x.xxxx### serialisation -> getPrompt(flag="analysis")), with two engineering substitutions:
- the official analyze() only parses MCQ "(a)(b)(c)" while our questions are open-ended =>
  batched greedy generation keeping the free-text answer (keyword matching is left to
  eval_understanding);
- its context limit is 512 points and univariate only => take the first channel and uniformly
  downsample longer series to 512 points.

Usage (GPU):
  python chronos_llm/eval/baseline_chattime_understanding.py --ckpt .../ChatTime-1-7B-Chat \
    --jsonl_list chronos_llm/configs/understanding_test_jsonl.txt \
    --out_dir outputs/eval/baseline_chattime/understanding
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch


def load_ts_first_channel(path, max_len=512):
    """Read the series -> first channel -> uniform downsampling to max_len. npy/csv via
    numpy/pandas, wav/flac via soundfile."""
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
            arr, _ = sf.read(path, dtype="float32", always_2d=True)   # (T, C)
        else:
            raise ValueError(f"unsupported ext {ext}")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"load {path} failed: {e}")
        return np.zeros(16, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.T   # (T,C)->(C,T) (same convention as the main package's _load_ts_2d)
    if arr.ndim == 1:
        arr = arr[None, :]
    # Take the first non-all-NaN channel
    for row in arr:
        if np.isfinite(row).any():
            s = row
            break
    else:
        return np.zeros(16, dtype=np.float32)
    if len(s) > max_len:
        idx = np.linspace(0, len(s) - 1, max_len).round().astype(int)
        s = s[idx]
    return np.nan_to_num(s, nan=np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl_list", required=True)
    ap.add_argument("--base_dir", required=True, help="prefix for relative ori_path values (Release_v1 root)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="MIMII")
    ap.add_argument("--file_shard_idx", type=int, default=0)
    ap.add_argument("--file_num_shards", type=int, default=1)
    args = ap.parse_args()

    from transformers import LlamaForCausalLM, LlamaTokenizer
    from utils.prompt import getPrompt          # ChatTime repository
    from utils.tools import Discretizer, Serializer

    model = LlamaForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True).eval()
    tok = LlamaTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"                    # left pad for batched generation
    disc, ser = Discretizer(), Serializer()

    files = [l.strip() for l in open(args.jsonl_list)
             if l.strip() and not l.strip().startswith("#")]
    files = [f for f in files if not any(e in f for e in args.exclude.split(",") if e)]
    if args.file_num_shards > 1:
        files = files[args.file_shard_idx:: args.file_num_shards]
    os.makedirs(args.out_dir, exist_ok=True)

    for jf in files:
        rows = [json.loads(l) for l in open(jf)]
        if args.limit:
            rows = rows[:args.limit]
        name = os.path.basename(jf)
        out_path = os.path.join(args.out_dir, name)
        if os.path.exists(out_path):
            print(f"[skip] {name} already exists")
            continue
        print(f"[chattime-u] {name}: {len(rows)} rows")
        results = []
        for b0 in range(0, len(rows), args.batch_size):
            chunk = rows[b0:b0 + args.batch_size]
            prompts = []
            for r in chunk:
                ori = ((r.get("input_ts") or {}).get("original") or {}).get("ori_path") or ""
                path = ori if os.path.isabs(ori) else os.path.join(args.base_dir, ori)
                s = load_ts_first_channel(path, args.max_len)
                serialized = ser.serialize(disc.discretize(s))
                q = (r.get("input_text") or [""])[0]
                prompts.append(getPrompt(flag="analysis", instruction=q, input=serialized))
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=3800).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.eos_token_id)
            gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, g in zip(chunk, gen):
                results.append({
                    "id": r.get("id"), "uid": r.get("uid"),
                    "dataset_name": r.get("dataset_name"), "task": r.get("task"),
                    "scene": r.get("scene"), "input_text": r.get("input_text"),
                    "generated_text": (g or "").strip(),
                    "ground_truth": r.get("gt_text"), "gt_result": r.get("gt_result"),
                })
            if (b0 // args.batch_size) % 10 == 0:
                print(f"  {b0 + len(chunk)}/{len(rows)}")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
