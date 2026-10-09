"""Prepend a structured magnitude tag `[Magnitude: {BAND}]` to the conclusion of every forecasting row,
and rewrite the closing sentence of the prompt so that it asks for the tag.

BAND is the upper-case form of one of the five bands tiny / well_below / mod_below / typical / above
(the same band names as `chronos_llm.eval.causal_faithfulness.BANDS`). ratio = this row's future peak /
the p90 of the future peaks over all rows of the same dataset ("typical"); the band is the interval of
BAND_EDGES that contains the ratio. The existing conclusion text is kept unchanged after the tag.

Skipped: multi-channel future (2D), typical/peak non-finite or <= 1e-9.

The closing sentence of the `prompt` column (the fixed final sentence produced by the corpus prompt
builder `build_downstream_prompt`) is replaced by a version that requires the tag and lists the five
options; if that sentence is not found the script raises, to catch upstream template changes.

`--exclude-datasets` optionally removes the given dataset_name values (exact match) from both train
and test.

Usage:
  python chronos_llm/scripts/utils/make_structured_conclusion.py \\
      --in  data/forecast/<corpus>.parquet \\
      --out data/forecast/<corpus>_structtag.parquet \\
      [--exclude-datasets <dataset_name>,<dataset_name>]
"""
import argparse

import numpy as np
import pandas as pd

# Band taxonomy (lowest to highest). No heavy dependencies are imported, so the script runs without torch.
BAND_ORDER = ["tiny", "well_below", "mod_below", "typical", "above"]
BAND_EDGES = [0.10, 0.40, 0.70, 1.30, float("inf")]
BAND_TAG = {
    "tiny": "TINY", "well_below": "WELL_BELOW", "mod_below": "MOD_BELOW",
    "typical": "TYPICAL", "above": "ABOVE",
}

# Verbatim copy of the closing sentence of the corpus prompt builder `build_downstream_prompt`
# (this sentence is the only fixed closing constant in the template without per-row interpolation,
# so it can safely be replaced as a whole).
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


def _target_series(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        return np.asarray(v[0], dtype=float)
    return np.asarray(v, dtype=float)


def _future_peak(v):
    a = _target_series(v)
    a = a[np.isfinite(a)]
    return float(np.nanmax(a)) if a.size else np.nan


def band_of(ratio: float) -> str:
    for name, hi in zip(BAND_ORDER, BAND_EDGES):
        if ratio < hi:
            return name
    return BAND_ORDER[-1]


def make_tag(row: pd.Series, typical: float):
    """Return (band, tag_str) or None (not injectable: multi-channel / invalid typical or peak)."""
    fv = row["future_values"]
    v = list(fv) if not isinstance(fv, (list, np.ndarray)) else fv
    if len(v) and np.ndim(v[0]) > 0:
        return None  # multi-channel: do not inject (defensive skip)
    if not np.isfinite(typical) or typical <= 1e-9:
        return None
    peak = _future_peak(fv)
    if not np.isfinite(peak) or peak <= 1e-9:
        return None
    band = band_of(peak / typical)
    return band, f"[Magnitude: {BAND_TAG[band]}]"


def inject_prompt_instruction(prompt: str) -> str:
    """Replace the closing sentence of the prompt with the version that "requires [Magnitude: BAND]
    and lists the five options". The closing sentence is a verbatim constant in the upstream template;
    if it cannot be found, raise (so an upstream template change cannot silently disconnect the
    training data from the prompt instruction)."""
    s = str(prompt)
    if _TASK_TAIL_OLD not in s:
        if _TASK_TAIL_NEW in s:
            return s  # already the new version, idempotent
        raise ValueError(
            "prompt closing sentence does not match the build_downstream_prompt template; cannot replace safely -- "
            "the upstream template may have changed; verify manually and update _TASK_TAIL_OLD"
        )
    return s.replace(_TASK_TAIL_OLD, _TASK_TAIL_NEW)


def add_structured_tags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Try to prepend the tag to all rows (including test) and rewrite the prompt instruction at the same
    time; rows that cannot be tagged keep their conclusion unchanged, but the prompt instruction is
    rewritten uniformly (even when a row gets no tag, the instruction describes the task's format
    requirement and does not depend on whether a single row was hit). Returns (new df, stats dict)."""
    typ_map = (df["future_values"].map(_future_peak)
               .groupby(df["dataset_name"]).quantile(0.90).to_dict())
    out = df.copy()
    new_concl = out["conclusion"].astype(object).copy()
    out["prompt"] = out["prompt"].map(inject_prompt_instruction)
    bands = []
    n_touch = 0
    for pos, (_, row) in enumerate(df.iterrows()):
        picked = make_tag(row, typ_map.get(row["dataset_name"], np.nan))
        if picked is None:
            bands.append(None)
            continue
        band, tag = picked
        concl = str(row["conclusion"]).lstrip()
        new_concl.iat[pos] = f"{tag} {concl}"
        bands.append(band)
        n_touch += 1
    out["conclusion"] = new_concl
    out["structtag_band"] = bands  # provenance column, not fed to the model (the dataset ignores unknown columns)
    stats = {
        "n_total": len(df), "n_touch": n_touch, "n_skip": len(df) - n_touch,
        "by_band": pd.Series([b for b in bands if b is not None]).value_counts().to_dict(),
    }
    return out, stats


def filter_excluded_datasets(df: pd.DataFrame, exclude_datasets) -> pd.DataFrame:
    """Exclude by exact dataset_name match (from both train and test; the split is not re-stratified).
    Returns df unchanged when exclude_datasets is empty/None."""
    excl = [x.strip() for x in exclude_datasets if str(x).strip()] if exclude_datasets else []
    if not excl:
        return df
    return df[~df["dataset_name"].isin(excl)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--exclude-datasets", default="", help="comma-separated list of dataset_name values (exact match) to exclude from both train and test")
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    before = len(df)
    df = filter_excluded_datasets(df, args.exclude_datasets.split(","))
    if len(df) != before:
        print(f"excluded {args.exclude_datasets}: {before} -> {len(df)} rows")
    out, stats = add_structured_tags(df)
    out.to_parquet(args.dst)
    print(f"structured tag injection done: {stats['n_touch']}/{stats['n_total']} rows"
          f" (skipped {stats['n_skip']}) -> {args.dst}")
    print("by band:", stats["by_band"])


if __name__ == "__main__":
    main()
