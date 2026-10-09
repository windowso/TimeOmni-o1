"""TimeOmni-1 (ICLR'26, github.com/AntonGuan/TimeOmni-1, arXiv 2509.24803) zero-shot understanding
baseline: produce per-sample jsonl in the same format as infer_understanding on the understanding
test sets (metrics are then computed with eval_understanding.py).

Note: this is a completely different work from the existing "TimeOmni" baseline in this repository
(SciTS/TimeOmni, ICLR 2026, arXiv 2510.03255, `baseline_timeomni.py`); do not confuse the two.

## Model and protocol

TimeOmni-1-4B/9B is built on Qwen3.5 (`Qwen3_5ForConditionalGeneration`, same architecture / same
vocab_size=248320 as this project's main LLM Qwen3.5-9B) and is a **pure-text LLM with no separate
time-series encoder** -- confirmed by the official `inference/inference.py`: the time-series values are
spliced into the prompt text as a Python list literal (e.g. `"The time series of J96A is: [3.35,
2.92, ...]"`), via `AutoModelForCausalLM` + chat template, no `trust_remote_code` needed (the
checkpoint ships no custom modeling file). TimeOmni-1-4B is not access-restricted (TimeOmni-1-9B is a
gated repo requiring manual consent to share contact details), so the unrestricted 4B is used.

The output protocol of the official training is reused (`<think>...</think><answer>...</answer>`,
verified verbatim from the `causality_discovery` / `scenario_understanding` tasks of the official test
bed `anton-hugging/timeomni-1-testbed`):
- The system prompt of the three official MCQ-style tasks demands "output a single uppercase letter
  only" -- our understanding test sets are open-ended QA / classification / detection (not MCQ);
  copying it would push the model to emit a bare letter and lose the scorable descriptive text, so
  the system prompt drops the "single uppercase letter" condition and keeps the `<think>/<answer>`
  scaffold (maximal compatibility with the model's instruction-following training distribution).
- Time-series serialisation: the official format is a per-point bracketed list (not ChatTime-style
  discretised tokens, nor TimeReasoner-style "timestamp: value" lines), with `%.6g` adaptive
  significant digits (across datasets of different magnitudes, a fixed number of decimals would
  flatten small-magnitude values to 0.0000).
- Series loading (`ori_path` relative paths joined with base_dir, npy/csv/wav loading, first
  non-all-NaN channel, uniform down-sampling of over-long series to `--max_len`) is reused verbatim
  from `baseline_chattime_understanding.py::load_ts_first_channel`; `max_len` defaults to 512,
  aligned with ChatTime (same protocol for the two pure-text baselines).
- **Batched generation is supported** (unlike the TimeOmni-VL image pipeline, pure text has no batch
  interface limitation): left padding, greedy (`do_sample=False`, the same convention as the other
  local baselines in the repository; the officially recommended temperature=0.1/top_p=0.001 is
  near-deterministic anyway).
- Output `generated_text` = text inside the `<answer>...</answer>` tags; if the answer tag is
  unclosed / missing, fall back to the residual text after `</think>`, then to the whole raw text (a
  sample is never blanked out just because the tags are missing; keyword matching in
  `eval_understanding.py` decides correctness naturally).
- File-level sharding (`--file_shard_idx/--file_num_shards`; the files are naturally independent,
  output files never overwrite each other, no merge needed) + resume (existing output jsonl files are
  skipped) -- same convention as `baseline_timeomnivl_understanding.py`.

## Usage (GPU; TimeOmni-1 shares the Qwen3.5 backbone with this project's main LLM, so the same
environment with transformers 5.9 / fla works, no separate venv needed):

  python chronos_llm/eval/baseline_timeomni1_understanding.py \\
    --model_dir checkpoints/TimeOmni-1-4B \\
    --jsonl_list chronos_llm/configs/understanding_test_jsonl.txt \\
    --base_dir data/raw/scits/Release_v1 \\
    --out_dir outputs/eval/baseline_timeomni1/understanding \\
    [--limit 5] [--file_shard_idx 0 --file_num_shards 1]
"""
import argparse
import json
import os
import re
import time
import warnings

import numpy as np
import torch

SYSTEM_PROMPT = (
    "Output Format:\n"
    "<think>Your step-by-step reasoning process that justifies your answer</think>\n"
    "<answer>Your final answer</answer>"
)


def load_ts_first_channel(path, max_len=512):
    """Load a time series -> first channel -> uniformly down-sample to max_len. npy/csv via
    numpy/pandas, wav/flac via soundfile. Reused verbatim from baseline_chattime_understanding.py."""
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
        arr = arr.T   # (T,C)->(C,T)
    if arr.ndim == 1:
        arr = arr[None, :]
    for row in arr:
        if np.isfinite(row).any():
            s = row
            break
    else:
        return np.zeros(16, dtype=np.float32)
    if len(s) > max_len:
        idx = np.linspace(0, len(s) - 1, max_len).round().astype(int)
        s = s[idx]
    return np.nan_to_num(s, nan=0.0)


def series_to_text(s):
    vals = ", ".join(f"{v:.6g}" for v in s.tolist())
    return f"[{vals}]"


def build_problem(question, series):
    return (
        f"{question}\n\n"
        f"The input time series is: {series_to_text(series)}\n"
        f"The input length is: {len(series)}."
    )


def extract_answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text.strip()


def build_messages(question, series):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_problem(question, series)},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--jsonl_list", required=True)
    ap.add_argument("--base_dir", required=True, help="prefix for relative ori_path values (Release_v1 root)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--file_shard_idx", type=int, default=0)
    ap.add_argument("--file_num_shards", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[timeomni1-u] loading model from {args.model_dir} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # left padding for batched generation
    print(f"[timeomni1-u] model loaded in {time.time()-t0:.1f}s")

    files = [l.strip() for l in open(args.jsonl_list)
             if l.strip() and not l.strip().startswith("#")]
    files = [f for f in files if not any(e in f for e in args.exclude.split(",") if e)]
    if args.file_num_shards > 1:
        files = files[args.file_shard_idx:: args.file_num_shards]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[timeomni1-u] shard {args.file_shard_idx}/{args.file_num_shards}: {len(files)} files")

    for jf in files:
        rows = [json.loads(l) for l in open(jf)]
        if args.limit:
            rows = rows[:args.limit]
        name = os.path.basename(jf)
        out_path = os.path.join(args.out_dir, name)
        if os.path.exists(out_path):
            print(f"[skip] {name} already exists")
            continue
        print(f"[timeomni1-u] {name}: {len(rows)} samples")
        results = []
        n_fail = 0
        for b0 in range(0, len(rows), args.batch_size):
            chunk = rows[b0:b0 + args.batch_size]
            prompts = []
            for r in chunk:
                ori = ((r.get("input_ts") or {}).get("original") or {}).get("ori_path") or ""
                path = ori if os.path.isabs(ori) else os.path.join(args.base_dir, ori)
                s = load_ts_first_channel(path, args.max_len)
                q = (r.get("input_text") or [""])[0]
                messages = build_messages(q, s)
                prompts.append(tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=6144).to("cuda")
            try:
                # Do not override eos_token_id: generation_config.json declares eos as the two-id
                # list [<|im_end|>, <|endoftext|>], and model.generate reads the loaded
                # model.generation_config automatically; passing a single tok.eos_token_id explicitly
                # would drop one of the terminators and make some samples run to max_new_tokens.
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                        repetition_penalty=args.repetition_penalty,
                        pad_token_id=tok.pad_token_id)
                gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            except Exception as e:  # noqa: BLE001 -- a failed batch must not take down the whole file; blank text is counted as a failure
                n_fail += len(chunk)
                gen = [""] * len(chunk)
                print(f"  [warn] {name} batch {b0//args.batch_size} failed: {type(e).__name__}: {e}")

            for r, g in zip(chunk, gen):
                raw = (g or "").strip()
                if not raw:
                    n_fail += 1
                results.append({
                    "id": r.get("id"), "uid": r.get("uid"),
                    "dataset_name": r.get("dataset_name"), "task": r.get("task"),
                    "scene": r.get("scene"), "input_text": r.get("input_text"),
                    "generated_text": extract_answer(raw) if raw else "",
                    "ground_truth": r.get("gt_text"), "gt_result": r.get("gt_result"),
                })
            if (b0 // args.batch_size) % 5 == 0:
                print(f"  {b0 + len(chunk)}/{len(rows)} (cumulative failures {n_fail})")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out_path} (fail {n_fail}/{len(rows)})")


if __name__ == "__main__":
    main()
