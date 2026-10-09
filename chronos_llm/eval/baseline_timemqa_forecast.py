"""Time-MQA (ACL 2025, arXiv 2503.01875) zero-shot **forecasting** baseline: produce an npz with
the same protocol as eval_forecast.py on the MMTR test split.

Time-MQA is not understanding-only: forecasting is one of the five tasks of its TSQA training set
(42,557 instances, 64-256 input points, 8-32 forecast points), hence this forecasting-side run.
Model loading is identical to `baseline_timemqa.py` (understanding side): the
hf.co/Time-MQA/Qwen-2.5-7B LoRA adapter on the unsloth qwen2.5-7b bnb-4bit base, loaded in 4bit
as released and not merged.

## Protocol (aligned with the native format of the paper's Appendix A.1 Forecasting)

- The paper's forecasting training format (Appendix A.1 verbatim):
  Q = "{background text}. The input Time Series are [Time Series Data Points]. Please predict
  the next nine time series points given the information above."
  A = "Based on the given information, the predictions are [-11.69, -10.72, ...]."
  wrapped as `<QUE> {Q} <ANS> {A} </END>` (the Qwen variant ends with <|endoftext|>).
- Our prompt = `<QUE> {plain_prompt} The input Time Series are [v1, ...]. Please
  predict the next {fl} time series points given the information above. <ANS>` --
  the background text is plain_prompt (the same conditioning text as the other baselines), the
  question wording replicates its training distribution verbatim.
- History values as a `.6g` bracketed list (same serialisation as baseline_timeomni1.py; the
  `.4g` downsampling used by the understanding-side baseline does not apply to forecasting --
  forecasting needs the contiguous most recent history, uniform downsampling would change the
  frequency). History is cut to the most recent `--hist_len` (default 600).
- Answer parsing: cut at `</END>`/eos, then take the bracketed numeric list -- exact length
  preferred, longer lists truncated to the first fl values, shorter ones count as failure (the
  whole row stays NaN and counts against the parse success rate), plus an
  "unclosed list" rescue -- the same principle as baseline_timeomni1.py::extract_array (not
  imported, re-implemented independently).
- **Point forecast copied to all 21 quantiles** (same protocol as ChatTime/TimeReasoner/
  TimeOmni-1: non-probabilistic forecast, all 21 levels filled with the point value, CRPS
  reduces to a weighted absolute error).
- Its training forecast length is capped at 32 while many of our horizons are longer; the
  model is run without accommodation (same principle as the other baselines).

## Usage (GPU, environment with the Time-MQA dependencies):
  python chronos_llm/eval/baseline_timemqa_forecast.py \\
    --adapter_dir checkpoints/Time-MQA-Qwen2.5-7B \\
    --base_dir_model checkpoints/qwen2.5-7b-unsloth-bnb-4bit \\
    --parquet data/forecast/mmtr_forecast_corpus.parquet \\
    --output outputs/eval/baseline_timemqa/forecast_preds.npz [--limit 8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

QUANTILE_LEVELS = np.array([
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
], dtype=np.float32)


def _first_channel(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    return np.asarray(v, dtype=np.float32)


def series_to_text(vals):
    return "[" + ", ".join(f"{v:.6g}" for v in vals) + "]"


def build_prompt(hist, text, future_len):
    text = (text or "").strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return (
        f"<QUE> {text} The input Time Series are {series_to_text(hist)}. "
        f"Please predict the next {future_len} time series points given the "
        f"information above. <ANS>"
    )


def _trailing_open_list(text):
    """Rescue the case "the array hit max_new_tokens before its closing bracket was written"
    (same principle as baseline_timeomni1.py): take the numbers parseable after the last
    unclosed `[`, split by commas."""
    last_open = text.rfind("[")
    if last_open == -1 or "]" in text[last_open:]:
        return None
    tail = text[last_open + 1:]
    nums = []
    for tok in tail.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            nums.append(float(tok))
        except ValueError:
            break
    return nums or None


def extract_array(text, expect_len):
    """Prefer an exact length match; a longer list is truncated to the first expect_len values;
    a shorter one counts as failure. Same principle as baseline_timeomni1.py::extract_array (not
    imported, re-implemented independently)."""
    matches = re.findall(r"\[[^\[\]]*\]", text)
    best = None
    for m in reversed(matches):
        try:
            arr = json.loads(m)
        except (ValueError, TypeError):
            try:
                import ast
                arr = ast.literal_eval(m)
            except (ValueError, SyntaxError, TypeError):
                continue
        if not isinstance(arr, (list, tuple)) or len(arr) < expect_len:
            continue
        try:
            nums = [float(x) for x in arr]
        except (ValueError, TypeError):
            continue
        if len(nums) == expect_len:
            return np.asarray(nums, dtype=np.float32)
        if best is None:
            best = nums
    if best is not None:
        return np.asarray(best[:expect_len], dtype=np.float32)
    trailing = _trailing_open_list(text)
    if trailing is not None and len(trailing) >= expect_len:
        return np.asarray(trailing[:expect_len], dtype=np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--base_dir_model", required=True, help="local path of the unsloth qwen2.5-7b bnb-4bit base model")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--hist_len", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=2560)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[timemqa-fc] loading base {args.base_dir_model} + adapter {args.adapter_dir} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.adapter_dir)  # the adapter repo ships the full tokenizer
    base = AutoModelForCausalLM.from_pretrained(args.base_dir_model, device_map="cuda:0")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # batched generation must left-pad (decoder-only)
    print(f"[timemqa-fc] loaded in {time.time()-t0:.1f}s")

    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit]
    H = int(test["future_len"].max())
    N = len(test)
    print(f"[timemqa-fc] test N={N}, H={H}, hist_len={args.hist_len}, batch_size={args.batch_size}")

    P = np.full((N, len(QUANTILE_LEVELS), H), np.nan, dtype=np.float32)
    G = np.full((N, H), np.nan, dtype=np.float32)
    R = np.zeros((N, H), dtype=np.float32)
    V = np.zeros((N, H), dtype=bool)
    ids, ds_names = [None] * N, [None] * N
    n_fail = 0

    def _save():
        np.savez(args.output, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
                 quantile_levels=QUANTILE_LEVELS,
                 ids=np.array([i if i is not None else "" for i in ids], dtype=object),
                 dataset_names=np.array([d if d is not None else "" for d in ds_names], dtype=object))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fail_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), "failures.jsonl")
    fail_f = open(fail_path, "w", encoding="utf-8")

    for b0 in range(0, N, args.batch_size):
        idxs = list(range(b0, min(b0 + args.batch_size, N)))
        rows = [test.iloc[i] for i in idxs]
        prompts, fls = [], []
        for row in rows:
            hist = _first_channel(row["history_values"])
            hist = hist[np.isfinite(hist)][-args.hist_len:]
            fl = int(row["future_len"])
            fls.append(fl)
            text = str(row.get("plain_prompt") or "")
            prompts.append(build_prompt(hist, text, fl))

        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
        t_b = time.time()
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.pad_token_id)
            gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:  # noqa: BLE001
            gens = [""] * len(rows)
            print(f"  [warn] batch {b0//args.batch_size} generation failed: {type(e).__name__}: {e}")

        for i, row, fl, g in zip(idxs, rows, fls, gens):
            rid, dsname = str(row["id"]), str(row["dataset_name"])
            ids[i], ds_names[i] = rid, dsname
            fut_vals = _first_channel(row["future_values"])
            G[i, :fl] = fut_vals[:fl]
            V[i, :fl] = np.isfinite(fut_vals[:fl])
            rs, re_ = row.get("roi_start_idx"), row.get("roi_end_idx")
            if rs is not None and re_ is not None and not (pd.isna(rs) or pd.isna(re_)):
                a = max(0, int(rs) - int(row["past_len"]))
                b = min(fl, int(re_) - int(row["past_len"]))
                if b > a:
                    R[i, a:b] = 1.0

            ans = (g or "").split("</END>")[0]
            arr = extract_array(ans, fl)
            if arr is not None:
                P[i, :, :fl] = arr[None, :]
            else:
                n_fail += 1
                fail_f.write(json.dumps({"id": rid, "dataset_name": dsname,
                                          "answer_preview": (g or "")[-500:]}, ensure_ascii=False) + "\n")
                fail_f.flush()
                print(f"  [warn] sample {i} ({rid}) parse failed")

        print(f"  {min(b0 + args.batch_size, N)}/{N} (batch {time.time()-t_b:.1f}s, failures {n_fail})")
        if (b0 // args.batch_size) % 5 == 0:
            _save()

    fail_f.close()
    _save()
    print(f"[timemqa-fc] done {N} rows, {n_fail} failures -> {args.output}")


if __name__ == "__main__":
    main()
