"""TimeReasoner-style (WSDM 2025, realwangjiahao/TimeReasoner) zero-shot forecasting baseline.

NOTE: this does not import or execute that repository's own code: the official repository is a
thin, few-dozen-line wrapper around the DeepSeek API (`deepseek_api_output` in
`utils/api_ouput.py`), and its core contribution is the **prompt design** (hybrid instructions: raw
values + timestamps + textual background, see the "Hybrid Instructions" section of its README) rather
than an irreplaceable code asset. We therefore re-implement the same prompt paradigm independently in
our own repository (the One-Shot Reasoning strategy -- the simplest of the three paradigms, one API
call performs reasoning + prediction) through a standard OpenAI-compatible LLM API.

NOTE on execution: this script only makes API network calls and loads no local model, so it must be
run from a machine with outbound network access.

NOTE on model choice: `deepseek-reasoner` is the model identifier used in the original TimeReasoner
code (`utils/api_ouput.py`: `model="deepseek-reasoner"`), so it is the default. Only the DeepSeek family passes the
`reasoning_content` (the chain-of-thought text, separate from the final answer) through
`message.model_extra['reasoning_content']`; Gemini/Claude/GPT only expose the amount of thinking via
`usage.completion_tokens_details.reasoning_tokens` without emitting the chain-of-thought itself.

Scope: of the three reasoning paradigms (One-Shot / Decoupled / Rollout) only **One-Shot** is
implemented (a single call performs reasoning + prediction); Decoupled/Rollout are not implemented.

Protocol:
- text = plain_prompt (the same background+event text as the other baselines, without our own
  reasoning/answer);
- historical values and timestamps are given together (the core of the hybrid instruction: raw
  values + timestamp feature + contextual description); over-long histories keep only the most recent
  `--hist_len` points (default 600);
- output protocol: the model must end its reasoning with a ```json array``` of exactly future_len
  numbers; parsing prefers an exact-length match and otherwise accepts "array length >= future_len"
  truncated to the first future_len values (rescuing nearly-correct outputs that "gave a few extra",
  but not "gave too few" -- avoids filling real gaps with fabricated values); parse failures /
  too-short outputs leave NaN and are counted as failures (the same failure-handling
  principle as the ChatTime baseline);
- the point forecast is copied to all 21 quantiles (same protocol as the ChatTime baseline: not a
  probabilistic forecast, all 21 levels hold the point value, CRPS reduces to a weighted absolute
  error);
- concurrency: ThreadPoolExecutor, default 5, 3 retries with exponential backoff, 180 s timeout
  (reasoning models emit tokens slowly);
- **audit trail and resumability**: each sample's prompt/reasoning/answer/parse
  result is appended line by line to `--log_jsonl` (JSON per line, for manual review); on a rerun, if
  that `--log_jsonl` exists and an id previously had parse_ok=True, the cached result is reused and
  the API is not called again; the npz is written every `--checkpoint_every` samples (default 20), so
  an interrupted process does not lose completed calls; `--failures_jsonl` is written once at the end
  with the samples that are still failing in the **current state** (excluding historical failures
  that a rerun has since rescued).

Usage (pure API network calls, no local model is loaded):
  python chronos_llm/eval/baseline_timereasoner.py \
    --parquet data/forecast/mmtr_forecast_corpus.parquet \
    --output outputs/eval/baseline_timereasoner/forecast_preds.npz
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

QUANTILE_LEVELS = np.array([
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
], dtype=np.float32)

API_BASE_URL = "https://api.openai.com/v1"

SYSTEM_PROMPT = (
    "You are a time series forecasting expert. You will be given a historical time series "
    "(raw values with timestamps) and contextual background/event information. Reason step by "
    "step about trend, seasonality, and how the event affects the series, then predict the "
    "future values."
)


def _first_channel(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    return np.asarray(v, dtype=np.float32)


def build_prompt(hist, hist_ts, text, freq, future_len):
    # %.6g (6 significant digits, magnitude-adaptive) rather than fixed decimals: the corpus spans
    # datasets whose scales range from near-zero ratios to large counts, and a fixed 4 decimals would
    # truncate small-scale values to 0.0000 (all information lost).
    lines = [f"{t}: {v:.6g}" for t, v in zip(hist_ts, hist)]
    series_block = "\n".join(lines)
    return (
        f"Context: {text}\n\n"
        f"Historical time series (frequency={freq}):\n{series_block}\n\n"
        f"Task: predict the next {future_len} values continuing this series at the same "
        f"frequency. First reason step by step about the trend/seasonality/event impact, then "
        f"end your answer with a JSON array of exactly {future_len} numbers on its own line, "
        f"formatted like: ```json\n[v1, v2, ..., v{future_len}]\n```"
    )


def extract_json_array(text, expect_len):
    """Prefer an exact-length match; otherwise accept a longer valid array truncated to the first
    expect_len values (rescues outputs that "gave a few extra"). Do not rescue "gave too few" -- there
    is no reliable evidence of which values are missing, so it is better to fail the whole sample."""
    matches = re.findall(r"\[[^\[\]]*\]", text)
    best = None
    for m in reversed(matches):  # search from the end (the reasoning may mention other arrays; the final answer is last)
        try:
            arr = json.loads(m)
        except (ValueError, TypeError):
            continue
        if not isinstance(arr, list) or len(arr) < expect_len:
            continue
        try:
            nums = [float(x) for x in arr]
        except (ValueError, TypeError):
            continue
        if len(nums) == expect_len:
            return np.asarray(nums, dtype=np.float32)
        if best is None:
            best = nums  # candidate longer than expect_len; keep looking for a later exact match, fall back to it otherwise
    if best is not None:
        return np.asarray(best[:expect_len], dtype=np.float32)
    return None


def call_one(client, model, hist, hist_ts, text, freq, future_len, max_retries=3, timeout=180):
    prompt = build_prompt(hist, hist_ts, text, freq, future_len)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                timeout=timeout,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            extra = getattr(msg, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or ""
            arr = extract_json_array(content, future_len)
            if arr is not None:
                return arr, content, reasoning, prompt, None
            last_err = f"parse failed (len mismatch or no JSON array), content_tail={content[-200:]!r}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(2 ** attempt)
    return None, None, None, prompt, last_err


def _load_resume_cache(log_path):
    """Read an existing raw_log.jsonl (if any) and keep the last record per id -- on a rerun, samples
    with parse_ok=True reuse the cached result instead of calling the API again (so another
    interruption cannot waste already-completed calls)."""
    cache = {}
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[entry["id"]] = entry
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model", default="deepseek-reasoner")
    ap.add_argument("--hist_len", type=int, default=600)
    ap.add_argument("--max_workers", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log_jsonl", default=None,
                     help="per-sample prompt/reasoning/answer/parse-result audit log; defaults to "
                          "raw_log.jsonl next to --output")
    ap.add_argument("--failures_jsonl", default=None,
                     help="list of failed samples written once at the end; defaults to failures.jsonl "
                          "next to --output")
    ap.add_argument("--checkpoint_every", type=int, default=20,
                    help="write the npz every N completed samples (0 = only at the end)")
    ap.add_argument("--no_resume", action="store_true",
                    help="ignore the existing log_jsonl cache and call the API for every sample")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    log_path = args.log_jsonl or os.path.join(out_dir, "raw_log.jsonl")
    fail_path = args.failures_jsonl or os.path.join(out_dir, "failures.jsonl")

    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=os.environ["OPENAI_API_KEY"])

    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit]
    H = int(test["future_len"].max())
    N = len(test)
    print(f"[timereasoner] test N={N}, H={H}, model={args.model}, max_workers={args.max_workers}, hist_len={args.hist_len}")

    resume_cache = {} if args.no_resume else _load_resume_cache(log_path)
    n_cached = sum(1 for e in resume_cache.values() if e.get("parse_ok"))
    if n_cached:
        print(f"[timereasoner] resumed {n_cached} previously successful samples from {log_path} (not re-called)")

    P = np.full((N, len(QUANTILE_LEVELS), H), np.nan, dtype=np.float32)
    G = np.full((N, H), np.nan, dtype=np.float32)
    R = np.zeros((N, H), dtype=np.float32)
    V = np.zeros((N, H), dtype=bool)
    ids = [None] * N
    ds_names = [None] * N
    fail_records = [None] * N   # per row: None (success) or {"id":..., "error":..., "answer_preview":...}
    n_fail = 0

    log_f = open(log_path, "a", encoding="utf-8")
    log_lock = threading.Lock()

    def _save_npz():
        np.savez(args.output, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
                 quantile_levels=QUANTILE_LEVELS,
                 ids=np.array([i if i is not None else "" for i in ids], dtype=object),
                 dataset_names=np.array([d if d is not None else "" for d in ds_names], dtype=object))

    def _task(i, row):
        rid = str(row["id"])
        fl = int(row["future_len"])
        cached = resume_cache.get(rid)
        if cached is not None and cached.get("parse_ok") and cached.get("parsed_forecast") is not None:
            arr = np.asarray(cached["parsed_forecast"], dtype=np.float32)
            return i, row, fl, arr, None, None, None, True  # cache hit: no new call, no duplicate log line

        hist = _first_channel(row["history_values"])
        hist_ts = list(row["history_timestamps"])
        finite = np.isfinite(hist)
        hist, hist_ts = hist[finite][-args.hist_len:], [t for t, k in zip(hist_ts, finite) if k][-args.hist_len:]
        text = str(row.get("plain_prompt") or row.get("prompt") or "")
        freq = str(row.get("freq") or "")
        arr, content, reasoning, prompt, err = call_one(
            client, args.model, hist, hist_ts, text, freq, fl,
            max_retries=args.max_retries, timeout=args.timeout)
        return i, row, fl, arr, content, reasoning, err, False

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(_task, i, row) for i, row in test.iterrows()]
        done = 0
        try:
            for fut in as_completed(futs):
                i, row, fl, arr, content, reasoning, err, from_cache = fut.result()
                rid = str(row["id"])
                dsname = str(row["dataset_name"])
                fut_vals = _first_channel(row["future_values"])
                G[i, :fl] = fut_vals[:fl]
                V[i, :fl] = np.isfinite(fut_vals[:fl])
                rs, re_ = row.get("roi_start_idx"), row.get("roi_end_idx")
                if rs is not None and re_ is not None and not (pd.isna(rs) or pd.isna(re_)):
                    a = max(0, int(rs) - int(row["past_len"]))
                    b = min(fl, int(re_) - int(row["past_len"]))
                    if b > a:
                        R[i, a:b] = 1.0
                ids[i] = rid
                ds_names[i] = dsname

                if arr is not None:
                    P[i, :, :fl] = arr[None, :]
                    fail_records[i] = None
                else:
                    n_fail += 1
                    fail_records[i] = {"id": rid, "dataset_name": dsname, "error": err or "parse_failed",
                                        "answer_preview": (content or "")[:500]}
                    print(f"  [warn] sample {i} ({rid}) failed: {err}")

                if not from_cache:
                    entry = {
                        "id": rid, "dataset_name": dsname, "past_len": int(row["past_len"]),
                        "future_len": fl, "model": args.model,
                        "parse_ok": arr is not None,
                        "parsed_forecast": arr.tolist() if arr is not None else None,
                        "reasoning": reasoning, "answer": content, "error": err,
                    }
                    with log_lock:
                        log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        log_f.flush()

                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{N} (failed {n_fail})")
                if args.checkpoint_every and done % args.checkpoint_every == 0:
                    with log_lock:
                        _save_npz()
        finally:
            log_f.close()

    print(f"[timereasoner] finished {N} samples, {n_fail} failed")
    _save_npz()
    print(f"[timereasoner] npz -> {args.output}")

    with open(fail_path, "w", encoding="utf-8") as f:
        for rec in fail_records:
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[timereasoner] failures -> {fail_path} ({n_fail} samples)")


if __name__ == "__main__":
    main()
