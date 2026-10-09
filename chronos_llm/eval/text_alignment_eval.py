"""Forecast-branch "generated text vs. ground-truth text" alignment evaluation (pure text-to-text,
no rendered plots involved).

Background: in ``forecast_preds.jsonl`` the model-generated reasoning+conclusion (``gen_text``)
names a "region of interest" (ROI) and a trajectory "shape"; the ground truth likewise carries an
ROI (the parquet's ``roi_start_idx``/``roi_end_idx``, absolute coordinates) and a standardized shape
description (embedded in ``gt_conclusion``). This module computes two metrics (ROI parsing and
strict IoU reuse ``rl.reward.extract_roi``/``roi_iou``):

1. **Relaxed interval-overlap hit rate**: any overlap between the two intervals counts as a hit (no
   IoU threshold), reported alongside the strict IoU mean; samples where ``extract_roi`` fails are
   counted separately rather than silently skipped (the denominator is always the full sample count,
   shared by all three metrics).
2. **Shape semantic similarity**: an LLM-as-judge scores the trajectory-shape descriptions in
   ``gen_text`` and ``gt_conclusion`` on a 1-5 scale (definition in ``SHAPE_JUDGE_USER_TEMPLATE``),
   ignoring numeric ROI coordinates / timestamps / magnitude wording and judging only whether the
   shape narrative (direction, number/location of turning points, smooth vs. jagged, spikes,
   plateaus, oscillation) is semantically consistent.

Coordinate systems (consistent with the ``_forecast_jsonl_rows`` docstring in
``eval/infer_forecast.py``: "roi is converted to a future-relative interval [start, end)"):
- The ROI the model names in ``gen_text`` (parsed by ``extract_roi``) is already an **absolute** row
  index over the concatenated history+future -- that is the coordinate system used at generation time
  (e.g. "[510, 552)" or "rows 510 through 551"), the same as the parquet's
  ``roi_start_idx``/``roi_end_idx``, so it can be compared **without any conversion**.
- The jsonl ``roi`` field is **relative to the start of the future window**; the absolute ground
  truth = ``roi + parquet.past_len``, which must equal the parquet's own
  ``roi_start_idx``/``roi_end_idx`` exactly (``build_records`` cross-checks this for every sample and
  raises on mismatch, guarding against coordinate-convention drift).

The pure scoring logic (overlap test / IoU / coordinate conversion / domain mapping / judge-response
parsing) consists of pure functions testable on CPU; the only network-dependent pieces are
``call_shape_judge`` (single sample) and ``run_shape_judge_batch`` (orchestration: concurrency +
retry + resumable checkpoints), both of which take an externally supplied ``client`` so a fake client
can be injected in tests without touching the network.
"""
import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import time

import numpy as np

from chronos_llm.rl.reward import extract_roi, roi_iou
from chronos_llm.scripts.utils.paper_forecast_by_domain import DOMAIN_SOURCES

DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ENV_FILE = ".env"
DEFAULT_JUDGE_MODEL = "gpt-5.4"

# Data source -> forecasting domain. Reused directly from paper_forecast_by_domain.DOMAIN_SOURCES
# rather than redefined here, so the two cannot drift apart.
_SOURCE_TO_DOMAIN = {src: dom for dom, srcs in DOMAIN_SOURCES.items() for src in srcs}


# ---------------------------------------------------------------------------
# Coordinate conversion & interval-overlap test (pure functions, CPU-testable, no external deps)
# ---------------------------------------------------------------------------

def absolute_gt_roi(roi_relative, past_len):
    """Relative jsonl ROI ``[a, b)`` (relative to the start of the future window) -> absolute
    coordinates (same system as the ROI named in gen_text and the parquet
    ``roi_start_idx``/``roi_end_idx``)."""
    a, b = roi_relative
    return int(a) + int(past_len), int(b) + int(past_len)


def roi_overlap(a, b):
    """Whether two half-open intervals ``[start, end)`` overlap at all (no IoU threshold);
    False if either a or b is None."""
    if a is None or b is None:
        return False
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return hi > lo


def roi_alignment_stats(gen_text, gt_roi_abs):
    """Per-sample ROI alignment: parse the interval the model names in gen_text and compare it with
    the absolute ground-truth interval.

    Returns a dict:
      - extracted: (start,end) or None (parse failure)
      - extracted_ok: bool
      - overlap_hit: bool (relaxed criterion -- any overlap is a hit; always False on parse failure)
      - iou: float (strict criterion, reuses rl.reward.roi_iou; that function returns 0.0 for None)

    Extraction failures are not removed from the statistics -- the denominator is always the full
    sample count (failed samples are recorded as 0/False for overlap_hit/iou).
    """
    ext = extract_roi(gen_text)
    if ext is None:
        return {"extracted": None, "extracted_ok": False, "overlap_hit": False, "iou": 0.0}
    return {
        "extracted": ext,
        "extracted_ok": True,
        "overlap_hit": roi_overlap(ext, gt_roi_abs),
        "iou": roi_iou(ext, gt_roi_abs),
    }


def summarize_roi(items):
    """Aggregate a list of ``roi_alignment_stats`` results (each item must carry
    extracted_ok/overlap_hit/iou).

    overlap_hit_rate / iou_mean / extract_fail_rate share the denominator n (failed samples are not
    silently skipped); the extra ``overlap_hit_rate_given_extracted`` is only a transparency
    diagnostic (hit rate among successfully parsed samples) and does not affect the other
    denominators.
    """
    n = len(items)
    if n == 0:
        return {"n": 0, "overlap_hit_rate": float("nan"), "iou_mean": float("nan"),
                "extract_fail_rate": float("nan"), "overlap_hit_rate_given_extracted": float("nan")}
    hits = sum(1 for it in items if it["overlap_hit"])
    fails = sum(1 for it in items if not it["extracted_ok"])
    ious = [it["iou"] for it in items]
    extracted = [it for it in items if it["extracted_ok"]]
    hit_given_extracted = (
        sum(1 for it in extracted if it["overlap_hit"]) / len(extracted) if extracted else float("nan")
    )
    return {
        "n": n,
        "overlap_hit_rate": hits / n,
        "iou_mean": float(np.mean(ious)),
        "extract_fail_rate": fails / n,
        "overlap_hit_rate_given_extracted": hit_given_extracted,
    }


# ---------------------------------------------------------------------------
# Domain mapping
# ---------------------------------------------------------------------------

def dataset_name_of(sample_id):
    """Recover dataset_name from a jsonl id of the form ``'<dataset_name>/<idx>'`` -- verified to
    match the parquet ``dataset_name`` column for every sample (see the module's unit test)."""
    return "/".join(str(sample_id).split("/")[:-1])


def domain_of(dataset_name):
    """Data source -> forecasting domain; returns None for sources not listed in
    DOMAIN_SOURCES (callers should place those in an explicit "Unknown" bucket rather than silently
    dropping or misclassifying them)."""
    return _SOURCE_TO_DOMAIN.get(dataset_name)


def group_records_by_domain(records):
    """records (each must have a "domain" key) -> {domain_or_"Unknown": [record, ...]}."""
    groups = {}
    for rec in records:
        dom = rec["domain"] or "Unknown"
        groups.setdefault(dom, []).append(rec)
    return groups


# ---------------------------------------------------------------------------
# Data loading: jsonl predictions + parquet ground-truth reference
# ---------------------------------------------------------------------------

def load_predictions(jsonl_path):
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_roi_reference(parquet_path, split="test"):
    """Read the required parquet columns, filter by split, and index by id for per-sample lookup.
    Requires ids to be unique within the split (true for the MMTR corpus test split; if a parquet
    with non-unique ids is supplied this asserts loudly instead of silently falling back to
    positional alignment and producing wrong results)."""
    import pyarrow.parquet as pq

    cols = ["id", "dataset_name", "split", "past_len", "roi_start_idx", "roi_end_idx"]
    df = pq.read_table(parquet_path, columns=cols).to_pandas()
    df = df[df["split"] == split]
    assert df["id"].is_unique, (
        f"ids are not unique within {parquet_path} split={split!r}; positional alignment would be "
        f"required (not supported by this module)"
    )
    return df.set_index("id")


def build_records(pred_rows, ref_df):
    """Merge jsonl prediction rows with parquet ground-truth rows into one unified record per sample
    (including absolute ROI / domain).

    Data-hygiene guard: the absolute coordinates computed from the jsonl's relative ``roi`` plus the
    parquet's ``past_len`` must equal the parquet's own ``roi_start_idx``/``roi_end_idx`` exactly --
    a mismatch means the two files are from different versions or the coordinate convention has
    drifted, so we raise AssertionError instead of silently evaluating against the wrong ground truth.
    """
    records = []
    for r in pred_rows:
        sid = r["id"]
        if sid not in ref_df.index:
            raise KeyError(f"id {sid!r} not found in the parquet reference table (split/file mismatch?)")
        ref = ref_df.loc[sid]
        gt_abs = (int(ref["roi_start_idx"]), int(ref["roi_end_idx"]))
        rel = r.get("roi")
        if rel is not None:
            computed = absolute_gt_roi(rel, ref["past_len"])
            assert computed == gt_abs, (
                f"id={sid} coordinate conversion inconsistent: jsonl.roi(relative)+parquet.past_len="
                f"{computed} != parquet.roi_start_idx/roi_end_idx={gt_abs}"
            )
        ds_name = str(ref["dataset_name"])
        records.append({
            "id": sid,
            "dataset_name": ds_name,
            "domain": domain_of(ds_name),
            "gen_text": r.get("gen_text") or "",
            "gt_conclusion": r.get("gt_conclusion") or "",
            "gt_reasoning": r.get("gt_reasoning") or "",
            "gt_roi_abs": gt_abs,
        })
    return records


def compute_roi_alignment(records):
    """Compute roi_alignment_stats for each record, attaching id/domain for later per-domain
    aggregation."""
    out = []
    for rec in records:
        s = roi_alignment_stats(rec["gen_text"], rec["gt_roi_abs"])
        s["id"] = rec["id"]
        s["domain"] = rec["domain"] or "Unknown"
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Shape semantic similarity: LLM-as-judge
# ---------------------------------------------------------------------------

SHAPE_JUDGE_SYSTEM = (
    "You are a careful evaluator comparing two textual descriptions of a time series' "
    "future trajectory. Judge ONLY the qualitative SHAPE/trend narrative -- direction of "
    "change, number and location of turning points, smoothness vs. jaggedness, spikes, "
    "plateaus, oscillation. Explicitly IGNORE numeric index ranges, clock times/dates, and "
    "magnitude-level qualifiers (e.g. 'well below the series' typical highs') -- those are "
    "scored by a separate metric, not by you."
)

SHAPE_JUDGE_USER_TEMPLATE = """Compare the SHAPE of the future trajectory described in the two passages below.

[Model-generated text]
{gen_text}

[Reference (ground-truth) shape description]
{gt_conclusion}

Rate how semantically consistent the SHAPE described in the model-generated text is with \
the shape in the reference, on this 1-5 scale:
5 = same shape narrative (same direction(s) of change and turning-point structure)
4 = mostly consistent, minor difference in nuance/degree but same overall narrative
3 = partially consistent -- same broad direction but a clearly different shape detail (e.g. \
smooth vs. jagged, or a different number of turning points)
2 = mostly inconsistent -- different overall shape narrative, only superficial overlap
1 = contradictory / opposite shape (e.g. rises vs. falls, flat vs. spiking)

Respond with ONLY a JSON object of the form {{"score": <int 1-5>, "reason": "<one short sentence>"}}"""


def build_shape_judge_messages(gen_text, gt_conclusion):
    user = SHAPE_JUDGE_USER_TEMPLATE.format(gen_text=gen_text, gt_conclusion=gt_conclusion)
    return [
        {"role": "system", "content": SHAPE_JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_judge_response(raw_text):
    """Parse ``(score:int|None, reason:str)`` from the judge's raw reply. JSON first (tolerating
    surrounding markdown code fences etc.), with a regex fallback that looks for a bare 1-5 digit;
    if both fail, return (None, raw text)."""
    text = "" if raw_text is None else str(raw_text)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = obj.get("score")
            if score is not None:
                score = int(score)
                if 1 <= score <= 5:
                    return score, str(obj.get("reason", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    m2 = re.search(r"\b([1-5])\b", text)
    if m2:
        return int(m2.group(1)), text.strip()
    return None, text.strip()


def _retry_exceptions():
    """Retryable exception types of the openai SDK; when the SDK is unavailable, degrade to catching
    all exceptions (still retried, we merely lose the "permanent error should fail immediately"
    distinction, and a CPU-only environment without the openai package does not fail at import)."""
    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
        return (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError)
    except ImportError:
        return (Exception,)


def _with_retry(fn, max_attempts=3, base_delay=1.0, retry_exceptions=(Exception,)):
    """Exponential-backoff retry; non-retryable exceptions propagate immediately (inlined so this
    module keeps zero cross-package dependencies)."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retry_exceptions as e:
            last_exc = e
            if attempt == max_attempts - 1:
                break
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.5))
    raise last_exc


def call_shape_judge(gen_text, gt_conclusion, client, model=DEFAULT_JUDGE_MODEL,
                      max_attempts=3, base_delay=1.0):
    """Actually call the LLM API to score one sample's shape. ``client`` is an ``openai.OpenAI``
    instance (or any object implementing ``.chat.completions.create(model=, messages=) -> resp`` --
    tests inject a fake client and never touch the network). **No temperature/max_tokens are set**:
    for reasoning models (e.g. gpt-5.4) the thinking tokens count against the max_tokens budget, so a
    small value starves the actual answer; some reasoning models also reject a custom temperature,
    so neither is passed and the backend defaults are used.
    Returns ``(score:int|None, reason:str, raw_text:str)``.
    """
    messages = build_shape_judge_messages(gen_text, gt_conclusion)

    def _call():
        return client.chat.completions.create(model=model, messages=messages)

    resp = _with_retry(_call, max_attempts=max_attempts, base_delay=base_delay,
                        retry_exceptions=_retry_exceptions())
    raw = resp.choices[0].message.content
    score, reason = parse_judge_response(raw)
    return score, reason, raw


def run_shape_judge_batch(records, client, model=DEFAULT_JUDGE_MODEL, max_workers=3,
                           max_attempts=3, base_delay=1.0, progress_every=50,
                           checkpoint_path=None):
    """Call ``call_shape_judge`` for each record (each must carry id/gen_text/gt_conclusion) using a
    thread pool (default concurrency 3).

    A single sample exhausting its retries does not bring down the batch -- its error is recorded,
    score=None, and the next sample proceeds. If ``checkpoint_path`` is given: already-finished ids
    are read at start-up (resumable, so occasional network timeouts/interruptions across hundreds of
    requests do not force a restart from scratch); each finished sample is appended immediately as
    one JSONL line and flushed (incremental persistence, so an interruption mid-run does not lose
    API calls already paid for).

    Returns ``{id: {"id", "score", "reason", "raw", "error"}}``.
    """
    results = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                results[obj["id"]] = obj

    todo = [rec for rec in records if rec["id"] not in results]
    if not todo:
        return results

    ckpt_f = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None

    def _one(rec):
        try:
            score, reason, raw = call_shape_judge(
                rec["gen_text"], rec["gt_conclusion"], client, model=model,
                max_attempts=max_attempts, base_delay=base_delay,
            )
            return {"id": rec["id"], "score": score, "reason": reason, "raw": raw, "error": None}
        except Exception as e:  # noqa: BLE001 one failure must not sink the batch; record and go on
            return {"id": rec["id"], "score": None, "reason": "", "raw": "", "error": str(e)}

    done = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_one, rec) for rec in todo]
            for fut in concurrent.futures.as_completed(futures):
                out = fut.result()
                results[out["id"]] = out
                if ckpt_f:
                    ckpt_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    ckpt_f.flush()
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  shape judge: {done}/{len(todo)} done")
    finally:
        if ckpt_f:
            ckpt_f.close()
    return results


def summarize_shape(items):
    """Aggregate a list of shape-judge results (each item has an optional "score" key; None =
    unscored/failed).

    n_scored/mean/std are computed only over the successfully scored subset; judge_fail_rate keeps
    the full n as denominator (same "never silently skip" principle as the ROI side). distribution is
    the count per 1-5 level, to show the shape of the distribution rather than just the mean.
    """
    n = len(items)
    scores = [it["score"] for it in items if it.get("score") is not None]
    dist = {k: 0 for k in range(1, 6)}
    for s in scores:
        dist[s] = dist.get(s, 0) + 1
    return {
        "n": n,
        "n_scored": len(scores),
        "judge_fail_rate": (n - len(scores)) / n if n else float("nan"),
        "mean": float(np.mean(scores)) if scores else float("nan"),
        "std": float(np.std(scores)) if scores else float("nan"),
        "distribution": dist,
    }


# ---------------------------------------------------------------------------
# API client construction (helper next to the only network entry point -- reads the key from
# .env / environment variables)
# ---------------------------------------------------------------------------

def _load_api_key(env_file=DEFAULT_ENV_FILE):
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        f"OPENAI_API_KEY not found: the environment variable is unset and {env_file} does not exist "
        "or has no such key"
    )


def build_openai_client(api_base_url=DEFAULT_API_BASE_URL, api_key=None):
    from openai import OpenAI
    return OpenAI(base_url=api_base_url, api_key=api_key or _load_api_key())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_roi_table(overall, by_domain):
    print(f"\n{'scope':10} {'n':>4} | {'overlap_hit_rate':>17} | {'iou_mean(strict)':>17} | "
          f"{'extract_fail_rate':>18}")
    print("-" * 78)
    for scope, s in [("ALL", overall)] + sorted(by_domain.items()):
        print(f"{scope:10} {s['n']:>4} | {s['overlap_hit_rate']:17.4f} | "
              f"{s['iou_mean']:17.4f} | {s['extract_fail_rate']:18.4f}")


def _print_shape_table(overall, by_domain):
    print(f"\n{'scope':10} {'n':>4} {'n_scored':>9} | {'mean':>6} | {'std':>6} | "
          f"{'judge_fail_rate':>15}")
    print("-" * 66)
    for scope, s in [("ALL", overall)] + sorted(by_domain.items()):
        print(f"{scope:10} {s['n']:>4} {s['n_scored']:>9} | {s['mean']:6.3f} | "
              f"{s['std']:6.3f} | {s['judge_fail_rate']:15.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", default="outputs/eval/forecast/forecast_preds.jsonl")
    ap.add_argument("--parquet", default=(
        "data/forecast/"
        "mmtr_forecast_corpus.parquet"))
    ap.add_argument("--out_dir", default=None,
                    help="write CSV/JSON details here; if omitted, only print to stdout")
    ap.add_argument("--skip_shape_judge", action="store_true",
                     help="only compute ROI alignment (skip LLM calls); for quick smoke runs / "
                          "saving API budget")
    ap.add_argument("--shape_limit", type=int, default=0,
                    help="run the shape judge only on the first N samples (0 = all)")
    ap.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--api_base_url", default=DEFAULT_API_BASE_URL)
    ap.add_argument("--max_workers", type=int, default=3)
    args = ap.parse_args(argv)

    pred_rows = load_predictions(args.jsonl)
    ref_df = load_roi_reference(args.parquet)
    records = build_records(pred_rows, ref_df)
    print(f"Loaded {len(records)} samples ({args.jsonl}); parquet coordinate-conversion "
          f"consistency check passed.")

    unknown = [r for r in records if r["domain"] is None]
    if unknown:
        print(f"WARNING: {len(unknown)} samples have a dataset_name outside DOMAIN_SOURCES; "
              f"placed in the 'Unknown' bucket: {sorted(set(r['dataset_name'] for r in unknown))}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # ---- ROI alignment (purely local computation, all samples) ----
    per_sample_roi = compute_roi_alignment(records)
    overall_roi = summarize_roi(per_sample_roi)
    roi_by_id = {s["id"]: s for s in per_sample_roi}
    dom_groups = group_records_by_domain(records)
    by_domain_roi = {dom: summarize_roi([roi_by_id[r["id"]] for r in recs])
                      for dom, recs in dom_groups.items()}
    _print_roi_table(overall_roi, by_domain_roi)

    # ---- shape semantic similarity (LLM-as-judge, optionally skipped) ----
    overall_shape, by_domain_shape, judge_results = None, {}, {}
    subset = records if args.shape_limit <= 0 else records[:args.shape_limit]
    if not args.skip_shape_judge:
        client = build_openai_client(api_base_url=args.api_base_url)
        ckpt = os.path.join(args.out_dir, "shape_judge_raw.jsonl") if args.out_dir else None
        print(f"\nCalling {args.judge_model} for shape-similarity scores, n={len(subset)}, "
              f"concurrency={args.max_workers}, checkpoint={ckpt}")
        judge_results = run_shape_judge_batch(
            subset, client, model=args.judge_model, max_workers=args.max_workers,
            checkpoint_path=ckpt)
        shape_items = [{"score": judge_results.get(r["id"], {}).get("score")} for r in subset]
        overall_shape = summarize_shape(shape_items)
        subset_dom_groups = group_records_by_domain(subset)
        by_domain_shape = {
            dom: summarize_shape([{"score": judge_results.get(r["id"], {}).get("score")}
                                   for r in recs])
            for dom, recs in subset_dom_groups.items()
        }
        _print_shape_table(overall_shape, by_domain_shape)
        n_fail = sum(1 for v in judge_results.values() if v.get("error"))
        if n_fail:
            print(f"WARNING: {n_fail}/{len(judge_results)} judge calls failed after exhausting "
                  f"retries (score=None)")

    # ---- write outputs ----
    if args.out_dir:
        roi_csv = os.path.join(args.out_dir, "roi_overlap_summary.csv")
        with open(roi_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scope", "n", "overlap_hit_rate", "iou_mean",
                                               "extract_fail_rate", "overlap_hit_rate_given_extracted"])
            w.writeheader()
            w.writerow({"scope": "ALL", **overall_roi})
            for dom in sorted(by_domain_roi):
                w.writerow({"scope": dom, **by_domain_roi[dom]})
        print(f"\nWrote {roi_csv}")

        if overall_shape is not None:
            shape_csv = os.path.join(args.out_dir, "shape_judge_summary.csv")
            with open(shape_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["scope", "n", "n_scored", "judge_fail_rate",
                                                   "mean", "std", "dist_1", "dist_2", "dist_3",
                                                   "dist_4", "dist_5"])
                w.writeheader()
                for scope, s in [("ALL", overall_shape)] + sorted(by_domain_shape.items()):
                    row = {k: v for k, v in s.items() if k != "distribution"}
                    row["scope"] = scope
                    for k, v in s["distribution"].items():
                        row[f"dist_{k}"] = v
                    w.writerow(row)
            print(f"Wrote {shape_csv}")

        per_sample_csv = os.path.join(args.out_dir, "per_sample.csv")
        with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "id", "dataset_name", "domain", "gt_roi_start", "gt_roi_end",
                "extracted_roi_start", "extracted_roi_end", "overlap_hit", "iou",
                "shape_score", "shape_reason"])
            w.writeheader()
            for rec in records:
                s = roi_by_id[rec["id"]]
                jr = judge_results.get(rec["id"], {})
                ext = s["extracted"]
                w.writerow({
                    "id": rec["id"], "dataset_name": rec["dataset_name"],
                    "domain": rec["domain"] or "Unknown",
                    "gt_roi_start": rec["gt_roi_abs"][0], "gt_roi_end": rec["gt_roi_abs"][1],
                    "extracted_roi_start": ext[0] if ext else "",
                    "extracted_roi_end": ext[1] if ext else "",
                    "overlap_hit": s["overlap_hit"], "iou": f"{s['iou']:.4f}",
                    "shape_score": jr.get("score", ""), "shape_reason": jr.get("reason", ""),
                })
        print(f"Wrote {per_sample_csv}")


if __name__ == "__main__":
    main()
