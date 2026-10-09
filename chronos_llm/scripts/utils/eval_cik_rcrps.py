"""CiK (Context is Key) evaluation -- CPU-side metric computation: reads the npz produced
by eval_cik.sh (TimeOmni-o1's 21-quantile forecasts) + the official CiK test parquet (355 tasks with
ground truth/weights/ROI/constraints), scores each task with the official
`compute_rcrps_with_hf_dataset.py::roi_crps`, then aggregates with the official formula (cap=5,
task-weight-weighted mean, with stderr) and additionally splits into the 5 context_sources subsets
(matching the column structure of the official README leaderboard).

The RCRPS math is not re-implemented -- the official script (Apache-2.0) is imported directly.
chronos2 natively outputs only 21 quantiles and has no autoregressive sampling interface, so this
script feeds the 21 quantile values to the official CRPS code as 21 empirical samples.

Usage:
  python chronos_llm/scripts/utils/eval_cik_rcrps.py \\
      --pred outputs/eval/cik/cik_preds.npz \\
      --cik_parquet data/raw/CiK/data/test-00000-of-00001.parquet \\
      --cik_code_dir data/raw/CiK \\
      --out outputs/eval/cik/rcrps_summary.json
"""
import argparse
import importlib.util
import json
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

CAP = 5.0
CONTEXT_CATEGORIES = {
    "Intemporal": "c_i", "Historical": "c_h", "Future": "c_f",
    "Covariate": "c_cov", "Causal": "c_causal",
}


def _load_official_module(cik_code_dir: str):
    """Import the official compute_rcrps_with_hf_dataset.py as a module (self-contained, needs only
    numpy/pandas/datasets/fractions) instead of copying its code."""
    path = f"{cik_code_dir}/compute_rcrps_with_hf_dataset.py"
    spec = importlib.util.spec_from_file_location("cik_official_rcrps", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_LIST_COLUMNS = ("region_of_interest", "constraint_variable_max_index",
                  "constraint_variable_max_values", "context_sources")


def normalize_entry(entry: dict) -> dict:
    """``pd.read_parquet(...).to_dict("records")`` reads list-typed columns as numpy arrays -- the
    official ``roi_crps`` code tests these fields directly with ``if entry["region_of_interest"]:``,
    which raises ``ValueError: truth value ... ambiguous`` for numpy arrays of length > 1. Convert
    them to native Python lists, matching the per-row behaviour of a real HF ``datasets.Dataset``."""
    e = dict(entry)
    for k in _LIST_COLUMNS:
        if k in e and e[k] is not None:
            e[k] = list(e[k])
    return e


def weighted_capped_rcrps(metrics: dict, weights: dict) -> tuple[float, float, int]:
    """The aggregation formula of the official compute_all_rcprs (cap at 5, then weight-weighted mean);
    additionally tolerates ids missing from metrics (counted as CAP, consistent with the
    missing-sample rule in results_complete.README). Returns (RCRPS, n_capped, n_total)."""
    num, den, n_capped = 0.0, 0.0, 0
    for k, w in weights.items():
        m = metrics.get(k, CAP)
        if m >= CAP or not np.isfinite(m):
            m = CAP
            n_capped += 1
        num += float(w) * m
        den += float(w)
    return num / den, n_capped, len(weights)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="npz produced by eval_cik.sh")
    ap.add_argument("--cik_parquet", required=True, help="official CiK test-split parquet")
    ap.add_argument("--cik_code_dir", required=True,
                    help="directory containing compute_rcrps_with_hf_dataset.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    official = _load_official_module(args.cik_code_dir)

    npz = np.load(args.pred, allow_pickle=True)
    ids = [str(x) for x in npz["ids"]]
    pred_by_id = {i: npz["pred_quantiles"][k] for k, i in enumerate(ids)}  # (Q, FL)

    df = pd.read_parquet(args.cik_parquet)
    entries = [normalize_entry(e) for e in df.to_dict("records")]
    for e in entries:
        e["id"] = f"{e['name']}__{e['seed']}"

    per_id_metric, per_id_context = {}, {}
    weights = {}
    for e in entries:
        eid = e["id"]
        weights[eid] = Fraction(e["weight"])
        per_id_context[eid] = list(e.get("context_sources") or [])
        if eid not in pred_by_id:
            continue  # missing sample (e.g. lazily skipped for an over-long rendering) -- counted as CAP=5 at aggregation per the official convention
        fl = len(pd.read_json(pd.io.common.StringIO(e["future_time"])))
        forecast = np.asarray(pred_by_id[eid], dtype=np.float64)[:, :fl]  # (Q, FL) is already (samples, timesteps)
        out = official.roi_crps(entry=e, forecast=forecast)
        per_id_metric[eid] = float(out["metric"])

    overall_rcrps, n_capped, n_total = weighted_capped_rcrps(per_id_metric, weights)
    n_missing = n_total - len(pred_by_id.keys() & weights.keys())

    result = {
        "n_total_tasks": n_total,
        "n_predicted": len(set(pred_by_id) & set(weights)),
        "n_missing_capped_at_5": n_missing,
        "n_capped_overall": n_capped,
        "overall_RCRPS": overall_rcrps,
    }
    for cat_name, code in CONTEXT_CATEGORIES.items():
        sub_ids = {k: w for k, w in weights.items() if code in per_id_context.get(k, [])}
        if not sub_ids:
            result[cat_name] = None
            continue
        cat_rcrps, _, _ = weighted_capped_rcrps(per_id_metric, sub_ids)
        result[cat_name] = cat_rcrps

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
