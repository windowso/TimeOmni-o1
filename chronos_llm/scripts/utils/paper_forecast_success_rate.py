"""Parse success rate of forecasting runs + success-rate-weighted metrics.

For runs that emit numbers as text (e.g. ChatTime/TimeReasoner/TimeOmni-1-4B/TimeOmni-VL-15B),
samples whose output could not be parsed are left entirely NaN in `pred_quantiles` (see the
respective `baseline_*.py`), and `forecast_metrics`, which averages per sample with `np.nanmean`,
skips those samples -- i.e. CRPS/PCC are computed only on the successfully parsed samples and the
failure rate itself does not enter the number.

This script weights explicitly:
    CRPS_adj = CRPS_success_only / success_rate   (more failures => heavier CRPS penalty)
    PCC_adj  = PCC_success_only  * success_rate    (more failures => larger PCC discount)
success_rate = number of samples in the domain for which the model produced all-finite values over
the full forecast window / total samples in the domain (shared by the full and roi metrics --
whether parsing succeeded is a property of the generation stage, not of full/roi).

Usage:
  python paper_forecast_success_rate.py NAME=path/to/preds.npz [NAME2=...] [--out-csv OUT]
"""
import argparse
import csv
import os

import numpy as np

from chronos_llm.eval.metrics import forecast_metrics

DOMAIN_SOURCES = {
    "Solar": ["CGTSF/MSPG", "fidelts/Canada_photovoltaics_plants",
              "fidelts/Germany_Renewable_Power_Grid"],
    "Load": ["fnf/load", "fidelts/California_ISO"],
    "Traffic": ["CGTSF/PTF", "fnf/traffic"],
    "Finance": ["finnews/MTBench_finance", "fnf/bitcoin"],
    "Climate": ["timemmd/Climate"],
}


def load_run(spec):
    name, path = spec.split("=", 1)
    z = np.load(path, allow_pickle=True)
    ds = np.array([str(x) for x in z["dataset_names"]])
    pred, gt = z["pred_quantiles"], z["gt"]
    valid = z["valid_mask"].astype(bool)
    roi = valid & (z["roi_mask"] > 0.5)
    # Success = all predicted values of the sample over the full valid window (valid_mask) are
    # finite (unparsed samples are left entirely NaN, consistent with the failure handling of
    # every baseline_*.py).
    success = np.array([
        np.isfinite(pred[i][:, valid[i]]).all() if valid[i].any() else True
        for i in range(pred.shape[0])
    ])
    out = {}
    for dom, srcs in DOMAIN_SOURCES.items():
        idx = np.where(np.isin(ds, srcs))[0]
        n = len(idx)
        n_succ = int(success[idx].sum())
        sr = n_succ / n if n else float("nan")
        mf = forecast_metrics(pred[idx], gt[idx], valid[idx], z["quantile_levels"])
        mr = forecast_metrics(pred[idx], gt[idx], roi[idx], z["quantile_levels"])
        out[dom] = {"n": n, "n_succ": n_succ, "success_rate": sr, "full": mf, "roi": mr}
    return name, out


def adj(m, sr):
    crps = m["CRPS"] / sr if sr > 0 else float("nan")
    pcc = m["PCC"] * sr if not np.isnan(m["PCC"]) else float("nan")
    return crps, pcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=npz_path")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    runs = dict(load_run(s) for s in args.runs)
    names = [s.split("=", 1)[0] for s in args.runs]

    for n in names:
        print(f"\n=== {n} ===")
        print(f"{'domain':10} {'n':>4} {'succ':>5} {'sr':>6} | "
              f"{'CRPS_full':>10} {'->adj':>10} | {'CRPS_roi':>10} {'->adj':>10} | "
              f"{'PCC_full':>9} {'->adj':>9} | {'PCC_roi':>9} {'->adj':>9}")
        overall_full, overall_roi, overall_full_adj, overall_roi_adj = [], [], [], []
        overall_pf, overall_pr, overall_pf_adj, overall_pr_adj = [], [], [], []
        for dom in DOMAIN_SOURCES:
            r = runs[n][dom]
            sr = r["success_rate"]
            cf_adj, pf_adj = adj(r["full"], sr)
            cr_adj, pr_adj = adj(r["roi"], sr)
            print(f"{dom:10} {r['n']:>4} {r['n_succ']:>5} {sr:>6.3f} | "
                  f"{r['full']['CRPS']:>10.4f} {cf_adj:>10.4f} | "
                  f"{r['roi']['CRPS']:>10.4f} {cr_adj:>10.4f} | "
                  f"{r['full']['PCC']:>9.4f} {pf_adj:>9.4f} | "
                  f"{r['roi']['PCC']:>9.4f} {pr_adj:>9.4f}")
            overall_full.append(r["full"]["CRPS"]); overall_full_adj.append(cf_adj)
            overall_roi.append(r["roi"]["CRPS"]); overall_roi_adj.append(cr_adj)
            overall_pf.append(r["full"]["PCC"]); overall_pf_adj.append(pf_adj)
            overall_pr.append(r["roi"]["PCC"]); overall_pr_adj.append(pr_adj)
        print(f"{'Overall(5-dom mean)':10} {'':>4} {'':>5} {'':>6} | "
              f"{np.mean(overall_full):>10.4f} {np.mean(overall_full_adj):>10.4f} | "
              f"{np.mean(overall_roi):>10.4f} {np.mean(overall_roi_adj):>10.4f} | "
              f"{np.nanmean(overall_pf):>9.4f} {np.nanmean(overall_pf_adj):>9.4f} | "
              f"{np.nanmean(overall_pr):>9.4f} {np.nanmean(overall_pr_adj):>9.4f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "domain", "n", "n_succ", "success_rate",
                        "CRPS_full", "CRPS_full_adj", "CRPS_roi", "CRPS_roi_adj",
                        "PCC_full", "PCC_full_adj", "PCC_roi", "PCC_roi_adj"])
            for n in names:
                for dom in DOMAIN_SOURCES:
                    r = runs[n][dom]
                    sr = r["success_rate"]
                    cf_adj, pf_adj = adj(r["full"], sr)
                    cr_adj, pr_adj = adj(r["roi"], sr)
                    w.writerow([n, dom, r["n"], r["n_succ"], f"{sr:.4f}",
                                f"{r['full']['CRPS']:.4f}", f"{cf_adj:.4f}",
                                f"{r['roi']['CRPS']:.4f}", f"{cr_adj:.4f}",
                                f"{r['full']['PCC']:.4f}", f"{pf_adj:.4f}",
                                f"{r['roi']['PCC']:.4f}", f"{pr_adj:.4f}"])
        print(f"\nwrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
