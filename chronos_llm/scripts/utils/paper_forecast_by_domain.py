"""Group the data sources of a forecast npz into the 5 forecasting domains (solar, load, traffic,
finance, climate) and report the metrics of each domain.

A domain row is **not** the weighted average of per-source means; all samples of the domain's
sources are pooled and forecast_metrics is called once on the pooled samples (CRPS computed per
sample, then averaged; samples whose metric is undefined, e.g. all-zero ground truth, are dropped
by nanmean).

DOMAIN_SOURCES holds the source -> domain grouping; changing the grouping only requires editing
this dict.

Usage:
  python paper_forecast_by_domain.py NAME=path/to/preds.npz [NAME2=...] \
      [--parquet corpus.parquet] [--out-csv OUT]
"""
import argparse
import csv
import functools
import os

import numpy as np

from chronos_llm.eval.metrics import forecast_metrics


@functools.lru_cache(maxsize=4)
def test_dataset_order(parquet):
    """(dataset_name array, id list) of the test split, in row order."""
    import pyarrow.parquet as pq
    df = pq.read_table(parquet, columns=["id", "dataset_name", "split"]).to_pandas()
    df = df[df["split"] == "test"].reset_index(drop=True)
    return df["dataset_name"].astype(str).to_numpy(), df["id"].astype(str).tolist()


def resolve_ds_names(npz, parquet):
    """Per-row source label: the npz's own ``dataset_names`` if present, otherwise the test split of
    ``parquet`` aligned by position (ids are not unique, so position is the only valid mapping)."""
    if "dataset_names" in npz.files:
        return np.array([str(x) for x in npz["dataset_names"]])
    if parquet is None:
        raise SystemExit("npz has no dataset_names; pass --parquet to align by position")
    ds_names, ref_ids = test_dataset_order(parquet)
    assert [str(x) for x in npz["ids"]] == ref_ids, "npz row order differs from the test split"
    return ds_names


DOMAIN_SOURCES = {
    "Solar": ["CGTSF/MSPG", "fidelts/Canada_photovoltaics_plants",
              "fidelts/Germany_Renewable_Power_Grid"],
    "Load": ["fnf/load", "fidelts/California_ISO"],
    "Traffic": ["CGTSF/PTF", "fnf/traffic"],
    "Finance": ["finnews/MTBench_finance", "fnf/bitcoin"],
    "Climate": ["timemmd/Climate"],
}


def load_run(spec, parquet):
    name, path = spec.split("=", 1)
    z = np.load(path, allow_pickle=True)
    ds = resolve_ds_names(z, parquet)
    valid = z["valid_mask"].astype(bool)
    roi = valid & (z["roi_mask"] > 0.5)
    out = {}
    for dom, srcs in DOMAIN_SOURCES.items():
        idx = np.where(np.isin(ds, srcs))[0]
        mf = forecast_metrics(z["pred_quantiles"][idx], z["gt"][idx], valid[idx], z["quantile_levels"])
        mr = forecast_metrics(z["pred_quantiles"][idx], z["gt"][idx], roi[idx], z["quantile_levels"])
        out[dom] = {"n": len(idx), "full": mf, "roi": mr}
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=npz_path")
    ap.add_argument("--parquet", default=None, help="positional-alignment fallback when the npz has no dataset_names")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    runs = dict(load_run(s, args.parquet) for s in args.runs)
    names = [s.split("=", 1)[0] for s in args.runs]

    hdr = f"{'domain':10} {'n':>4} | " + " | ".join(f"{n:>26}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for dom in DOMAIN_SOURCES:
        cells = [f"full={runs[n][dom]['full']['CRPS']:.4f}/roi={runs[n][dom]['roi']['CRPS']:.4f}"
                 for n in names]
        print(f"{dom:10} {runs[names[0]][dom]['n']:>4} | " + " | ".join(f"{c:>26}" for c in cells))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "domain", "n", "CRPS_full", "CRPS_roi", "PCC_full", "PCC_roi"])
            for n in names:
                for dom in DOMAIN_SOURCES:
                    r = runs[n][dom]
                    w.writerow([n, dom, r["n"],
                                f"{r['full']['CRPS']:.4f}", f"{r['roi']['CRPS']:.4f}",
                                f"{r['full']['PCC']:.4f}", f"{r['roi']['PCC']:.4f}"])
        print(f"\nwrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
