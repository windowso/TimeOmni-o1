"""Forecast metric computation: read the npz produced by infer_forecast.py and compute MAE/MAPE/PCC/CRPS
over the **whole forecast region** and over the **ROI region**.

- full-region mask = valid (forecast positions whose gt is not NaN).
- ROI-region mask = valid & roi_mask.
Metric definitions are in metrics.py (point forecast = q0.5, computed per sample then averaged;
MAE in raw units without normalisation, for comparison with external benchmarks; CRPS = normalised
WQL, the primary internal metric of this project).
"""
import argparse
import csv
import os

import numpy as np

from chronos_llm.eval.metrics import forecast_metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help=".npz produced by infer_forecast.py")
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args(argv)

    d = np.load(args.pred, allow_pickle=True)
    pred, gt = d["pred_quantiles"], d["gt"]
    roi_mask, valid, levels = d["roi_mask"], d["valid_mask"], d["quantile_levels"]
    valid = valid.astype(bool)
    roi = valid & (roi_mask > 0.5)

    full_m = forecast_metrics(pred, gt, valid, levels)
    roi_m = forecast_metrics(pred, gt, roi, levels)

    rows = [{"region": "full", **full_m}, {"region": "roi", **roi_m}]
    print(f"\nsamples={full_m['n_samples']}  quantiles={len(levels)}  "
          f"(valid samples for PCC: full={full_m['n_valid_pcc']}, roi={roi_m['n_valid_pcc']})")
    print(f"{'region':6s} | {'MAE':>10s} | {'MAPE(%)':>9s} | {'PCC':>7s} | {'CRPS(WQL)':>10s}")
    print("-" * 55)
    for r in rows:
        print(f"{r['region']:6s} | {r['MAE']:10.4f} | {r['MAPE']:9.4f} | {r['PCC']:7.4f} | {r['CRPS']:10.5f}")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        cols = ["region", "MAE", "MAPE", "PCC", "CRPS", "n_samples", "n_valid_pcc"]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
