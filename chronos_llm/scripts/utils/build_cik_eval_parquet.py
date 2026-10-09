"""Convert the official CiK (Context is Key) test split (355 rows) into a parquet that
ForecastParquetDataset can read, so that CiK can be evaluated directly (no training on CiK).

Only background/scenario/constraints/seasonal_period/past_time/future_time are taken from the
official data; the prompt is assembled with exactly the template the evaluated checkpoint was
trained on (the corpus prompt builder `build_downstream_prompt` plus the structured
magnitude-tag closing sentence patched in by chronos_llm/scripts/utils/make_structured_conclusion.py),
so that the text structure fed to the model at inference time is verbatim aligned with the training
distribution. The official RCRPS scoring does not read this script's output -- eval_cik_rcrps.py
re-reads the official CiK parquet itself to compute the metric, so no field can drift apart between
the two paths.

Usage:
  python chronos_llm/scripts/utils/build_cik_eval_parquet.py \\
      --cik_parquet data/raw/CiK/data/test-00000-of-00001.parquet \\
      --out data/external/cik_eval.parquet
"""
import argparse
from io import StringIO

import numpy as np
import pandas as pd

# Verbatim identical to the corpus prompt builder `build_downstream_prompt` (including
# the closing sentence, which is immediately replaced by the structured-tag version _TASK_TAIL_NEW
# below -- this two-step assembly is exactly what make_structured_conclusion.py's
# inject_prompt_instruction does to the training parquet, so the inference prompt stays verbatim
# aligned with the training distribution).
_TASK_TAIL_OLD = (
    "One sentence giving the ROI index range [start_idx, end_idx) and the shape effect "
    "the event has on the future window there."
)
_TASK_TAIL_NEW = (
    "First, prefix the conclusion with a bracketed magnitude tag stating how the ROI peak "
    "compares to this series' typical peak, choosing exactly one of: [Magnitude: TINY] "
    "(near zero) / [Magnitude: WELL_BELOW] (well below typical) / [Magnitude: MOD_BELOW] "
    "(moderately below typical) / [Magnitude: TYPICAL] (around typical) / [Magnitude: ABOVE] "
    "(above typical). Then, in the same sentence, give the ROI index range "
    "[start_idx, end_idx) and the shape effect the event has on the future window there."
)


def build_downstream_prompt(*, background, event, freq, past_len, future_len, total_len,
                             hist_start, hist_end, fut_start, fut_end) -> str:
    """Verbatim replica of the corpus prompt builder `build_downstream_prompt`, with
    the structured magnitude-tag closing sentence swapped in afterwards (equivalent to running
    inject_prompt_instruction once over the old template that was used for training)."""
    body = f"""\
Your task is to infer how the FUTURE window evolves from the history dynamics and an established EVENT. A separate time-series encoder feeds you the HISTORY window values directly; you are also given the dataset background and the event. The event may state what happens within a precise time range -- use it to infer which contiguous index range of the future window that time range corresponds to, and what effect (shape) the event has on the series there.

## Dataset background
{background}

## Event (established fact)
{event}

## Series metadata
- frequency: {freq}
- history window: indices [0, {past_len}), {past_len} points, time {hist_start} -> {hist_end}
- future window:  indices [{past_len}, {total_len}), {future_len} points, time {fut_start} -> {fut_end}

## Task
Treating the event as an established fact, reason forward from the history window and the event, then commit to a conclusion. Output your reasoning wrapped in <think> ... </think>, immediately followed by a single conclusion sentence:
<think>
2-4 sentence forward chain: (i) the ROI clock time lines up with the event time; (ii) event + history -> shape.
</think>
{_TASK_TAIL_OLD}"""
    return body.replace(_TASK_TAIL_OLD, _TASK_TAIL_NEW)


def _freq_str(index: pd.DatetimeIndex) -> str:
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return "unknown"
    td = diffs.mode().iloc[0]
    secs = td.total_seconds()
    if secs % 86400 == 0 and secs >= 86400:
        return f"{int(secs // 86400)} day(s)"
    if secs % 3600 == 0 and secs >= 3600:
        return f"{int(secs // 3600)} hour(s)"
    if secs % 60 == 0:
        return f"{int(secs // 60)} minute(s)"
    return f"{secs:g} second(s)"


def build_event_text(scenario: str, constraints: str) -> str:
    scenario = (scenario or "").strip()
    constraints = (constraints or "").strip()
    parts = [p for p in (scenario, f"Constraints: {constraints}" if constraints else "") if p]
    return "\n\n".join(parts) if parts else "(no additional event information provided)"


def build_row(entry: dict) -> dict:
    """Build the fields ForecastParquetDataset needs from one official CiK record (parquet row -> dict)."""
    past = pd.read_json(StringIO(entry["past_time"]))
    future = pd.read_json(StringIO(entry["future_time"]))
    hist = past[past.columns[-1]].to_numpy(dtype=np.float64)
    fut = future[future.columns[-1]].to_numpy(dtype=np.float64)
    past_len, future_len = len(hist), len(fut)

    event = build_event_text(entry.get("scenario", ""), entry.get("constraints", ""))
    seasonal_period = entry.get("seasonal_period", -1)
    background = str(entry.get("background", "")).strip()
    if seasonal_period and seasonal_period > 0:
        background = f"{background}\n(Suggested seasonal period: {int(seasonal_period)} points.)"

    freq = _freq_str(past.index)
    prompt = build_downstream_prompt(
        background=background, event=event, freq=freq,
        past_len=past_len, future_len=future_len, total_len=past_len + future_len,
        hist_start=str(past.index[0]), hist_end=str(past.index[-1]),
        fut_start=str(future.index[0]), fut_end=str(future.index[-1]),
    )

    roi_raw = entry.get("region_of_interest")
    roi = list(roi_raw) if roi_raw is not None and len(roi_raw) > 0 else []
    roi_start_idx = past_len + min(roi) if roi else None
    roi_end_idx = past_len + max(roi) + 1 if roi else None

    return {
        "id": f"{entry['name']}__{entry['seed']}",
        "dataset_name": "CiK",
        "split": "test",
        "background": background,
        "event": event,
        "prompt": prompt,
        "history_values": hist.tolist(),
        "future_values": fut.tolist(),
        "past_len": past_len,
        "roi_start_idx": roi_start_idx,
        "roi_end_idx": roi_end_idx,
        "reasoning": "",
        "conclusion": "",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cik_parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.cik_parquet)
    rows = [build_row(r) for r in df.to_dict("records")]
    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"CiK eval parquet: {len(out)} rows -> {args.out}")
    print(f"history length range: {out['past_len'].min()}-{out['past_len'].max()}, "
          f"future length range: {out['future_values'].map(len).min()}-{out['future_values'].map(len).max()}")


if __name__ == "__main__":
    main()
