"""TimeOmni-1 (ICLR'26, github.com/AntonGuan/TimeOmni-1, arXiv 2509.24803) forecasting-side
zero-shot baseline: produce an npz on our test parquet in the same format as eval_forecast.py.

NOTE: this is a completely different work from the existing "TimeOmni" baseline
(SciTS/TimeOmni, ICLR 2026, arXiv 2510.03255, `baseline_timeomni.py`, which on the forecasting side
is run under the fine-tuning exception) -- do not confuse the two. TimeOmni-1 is run **zero-shot**
here (consistent with our default baseline principle).

## Model and protocol

Architecture / environment as in `baseline_timeomni1_understanding.py` (a text-only LLM on a
Qwen3.5 base; runs in the main environment, no trust_remote_code needed).

The only forecasting task in the official test bed `anton-hugging/timeomni-1-testbed` is
`event_aware_forecasting`; the protocol verified verbatim:
- system prompt: `"You should think the impact of the event first, then output the "
  "predicted sequence.\n\nOutput Format:\n<think>Your step-by-step reasoning process</think>\n"
  "<answer>[Your predicted sequence]</answer>"` (reused as is, not rewritten -- this is the output
  protocol the model was specifically trained on for forecasting; rewriting it would introduce our
  own out-of-distribution instruction).
- problem text template: "Over the past N hours (from T0 to T1), the <desc> was: \n[v1, v2, ...]"
  + "During this time, <event or 'no significant events took place'>." +
  "In the next H hours, <event or 'no significant events are scheduled'>." +
  "Based on the patterns and events provided above, please predict the sequence for the
  next H <unit> from T2 to T3." -- we substitute `plain_prompt` (plain text of background + event,
  the same conditioning text as the other baselines) for the "event" clauses of its template, and
  serialise the history values with the same `[v1, v2, ...]` bracketed list (not the
  TimeReasoner-style "timestamp: value" per-line text -- the serialisation format of TimeOmni-1's
  own training distribution is replicated verbatim, this is its specific protocol-alignment point).
- answer parsing: the JSON / Python literal list inside `<answer>[...]</answer>`, the same logic as
  the official `get_score.py::extract_list` + `ast.literal_eval` (this script does not import that
  repository's code; the same regex + parsing rules are re-implemented here to avoid pulling in vllm
  and other extra dependencies). Exact length match is preferred; when the list is longer, the
  first future_len values are taken (rescuing "a few extra values"); when shorter, the sample is a
  failure -- the same failure-handling principle as the TimeReasoner / ChatTime point-forecast
  baselines.
- **point forecast copied to all 21 quantiles** (same convention as ChatTime / TimeReasoner:
  non-probabilistic forecast, all 21 levels filled with the point value, CRPS reduces to a
  weighted absolute error).
- history truncated to the most recent `--hist_len` (default 600, the same default as
  TimeReasoner); univariate, first channel.
- greedy decoding (`do_sample=False`, the same convention as the other local baselines in the repo).

## Usage (GPU machine):
  python chronos_llm/eval/baseline_timeomni1.py \\
    --model_dir checkpoints/TimeOmni-1-4B \\
    --parquet data/forecast/mmtr_forecast_corpus.parquet \\
    --output outputs/eval/baseline_timeomni1/forecast_preds.npz
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch

QUANTILE_LEVELS = np.array([
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
], dtype=np.float32)

SYSTEM_PROMPT = (
    "You should think the impact of the event first, then output the predicted sequence.\n\n"
    "Output Format:\n"
    "<think>Your step-by-step reasoning process</think>\n"
    "<answer>[Your predicted sequence]</answer>"
)


def _first_channel(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    return np.asarray(v, dtype=np.float32)


def series_to_text(vals):
    return "[" + ", ".join(f"{v:.6g}" for v in vals) + "]"


def build_problem(hist, text, freq, future_len):
    return (
        f"Historical time series (frequency={freq}, {len(hist)} points):\n"
        f"{series_to_text(hist)}\n\n"
        f"Context: {text}\n\n"
        f"Based on the patterns and events described above, please predict the sequence for "
        f"the next {future_len} points continuing this series at the same frequency."
    )


def build_messages(hist, text, freq, future_len):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_problem(hist, text, freq, future_len)},
    ]


def extract_answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _trailing_open_list(text):
    """Rescue the case where generation hit max_new_tokens before the array's closing bracket was
    written: take the content after the last unclosed `[` in the text (no `]` after it) and split
    on commas into as many valid numbers as possible (stop at the first token that does not parse
    -- usually a truncated half number, which must not be used as a real value). Returns None when
    there is no unclosed `[`."""
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
    """Prefer an exact length match; when longer, take the first expect_len values; when shorter,
    treat as failure. Same principle as baseline_timereasoner.py::extract_json_array (not
    imported, re-implemented independently). An extra "unclosed list" rescue layer is added (see
    `_trailing_open_list`) -- when generation is cut off at max_new_tokens this often happens
    halfway through the answer array; the text then contains no valid closed `[...]` match, yet
    the model has already listed all the values and merely failed to write the closing `]`.
    Without the rescue, such samples whose information is actually complete would be recorded as
    total failures."""
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
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--hist_len", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=2560)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[timeomni1] loading model from {args.model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit]
    H = int(test["future_len"].max())
    N = len(test)
    print(f"[timeomni1] test N={N}, H={H}, hist_len={args.hist_len}, batch_size={args.batch_size}")

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
            text = str(row.get("plain_prompt") or row.get("prompt") or "")
            freq = str(row.get("freq") or "")
            messages = build_messages(hist, text, freq, fl)
            prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to("cuda")
        try:
            # Do not override eos_token_id: see the corresponding comment in
            # baseline_timeomni1_understanding.py (the two-eos-id list of generation_config.json
            # only takes effect through the generation_config the model loads automatically).
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     repetition_penalty=args.repetition_penalty,
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

            arr = extract_array(extract_answer(g or ""), fl)
            if arr is not None:
                P[i, :, :fl] = arr[None, :]
            else:
                n_fail += 1
                fail_f.write(json.dumps({"id": rid, "dataset_name": dsname,
                                          "answer_preview": (g or "")[-500:]}, ensure_ascii=False) + "\n")
                fail_f.flush()
                print(f"  [warn] sample {i} ({rid}) parse failed")

        if (b0 // args.batch_size) % 5 == 0:
            print(f"  {min(b0 + args.batch_size, N)}/{N} (failures {n_fail})")
            _save()

    fail_f.close()
    _save()
    print(f"[timeomni1] done: {N} samples, {n_fail} failures -> {args.output}")


if __name__ == "__main__":
    main()
