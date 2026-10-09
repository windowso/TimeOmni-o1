"""Merge the `forecast_preds.shard{N}.npz` shard files produced by baseline_timeomnivl.py with
`--num_shards>1` into a single npz that eval_forecast.py can consume directly.

H (the forecast horizon width) is already guaranteed to be identical across shards (baseline_timeomnivl.py
computes H from the full test set before sharding, see the comments in that file), so the arrays are simply
concatenated along axis=0 (samples); `ids`/`dataset_names` are object arrays and are concatenated the same
way. Shards are concatenated in ascending shard_idx order -- since each shard is a strided slice
`iloc[shard_idx::num_shards]`, the merged sample order is **not** the original parquet row order (this does
not affect correctness: eval_forecast.py computes per-row metrics independently and then aggregates, without
relying on row order; if the original order is ever needed, sort by ids -- currently not required).

Usage:
  python chronos_llm/scripts/utils/merge_timeomnivl_forecast_shards.py \\
    outputs/eval/baseline_timeomnivl/forecast_preds.npz --num_shards 4
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="target npz path as if unsharded (the shard file names are derived from it)")
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--delete_shards", action="store_true", help="delete the shard files after a successful merge")
    args = ap.parse_args()

    base, ext = os.path.splitext(args.output)
    shard_paths = [f"{base}.shard{i}{ext}" for i in range(args.num_shards)]
    for p in shard_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"shard file missing: {p}")

    shards = [np.load(p, allow_pickle=True) for p in shard_paths]
    Hs = {int(s["gt"].shape[1]) for s in shards}
    if len(Hs) != 1:
        raise ValueError(f"H differs across shards {Hs}: re-run the shards with H computed from the full test set")

    merged = {
        "pred_quantiles": np.concatenate([s["pred_quantiles"] for s in shards], axis=0),
        "gt": np.concatenate([s["gt"] for s in shards], axis=0),
        "roi_mask": np.concatenate([s["roi_mask"] for s in shards], axis=0),
        "valid_mask": np.concatenate([s["valid_mask"] for s in shards], axis=0),
        "quantile_levels": shards[0]["quantile_levels"],
        "ids": np.concatenate([s["ids"] for s in shards], axis=0),
        "dataset_names": np.concatenate([s["dataset_names"] for s in shards], axis=0),
    }
    n_total = sum(len(s["ids"]) for s in shards)
    assert merged["pred_quantiles"].shape[0] == n_total

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **merged)
    print(f"merged {args.num_shards} shards -> {args.output} (N={n_total}, H={Hs.pop()})")

    if args.delete_shards:
        for p in shard_paths:
            os.remove(p)
        print(f"deleted {len(shard_paths)} shard files")


if __name__ == "__main__":
    main()
