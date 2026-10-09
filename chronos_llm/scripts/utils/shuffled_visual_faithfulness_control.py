"""Mismatched-partner control for the forecasting plot-vs-text judge (supportiveness).

The matched judge compares "a sample's own forecast plot + values vs its own generated reasoning
text"; this script instead pairs each sample with the reasoning text of **another sample**
(``--pairing``) while everything else (history / forecast values / ROI / domain / id) still belongs to
the evaluated sample.

Reuses ``load_records`` / rendering / judge calls / domain aggregation from
``forecast_visual_faithfulness.py``; only ``gen_text`` is swapped for another sample's before it is
fed to the judge.
"""
import argparse
import csv
import os
import random

import numpy as np

from chronos_llm.eval.forecast_visual_faithfulness import (
    build_openai_client,
    load_records,
    run_visual_judge_batch,
    summarize,
)
from chronos_llm.eval.text_alignment_eval import group_records_by_domain

DEFAULT_NPZ = (
    "outputs/eval/forecast/forecast_preds.npz"
)
DEFAULT_JSONL = (
    "outputs/eval/forecast/forecast_preds.jsonl"
)
DEFAULT_PARQUET = (
    "data/forecast/"
    "mmtr_forecast_corpus.parquet"
)


def build_pairing(records, sel, pairing="shift", shift=1, seed=0):
    """Pick a mismatched partner for every evaluated sample; returns {i: j}. Shares one dispatch with
    `split_text_judge_eval.build_pairing`, so for the same seed the text judge and the plot-vs-text
    judge use the same pairing. See that docstring for the semantics of each mode."""
    from chronos_llm.scripts.utils.split_text_judge_eval import build_pairing as _bp
    return _bp(len(records), sel, pairing=pairing, shift=shift, seed=seed)


def apply_pairing(records, partner):
    """Keep each sample's own history / forecast values / ROI / domain / id; only replace gen_text with the partner's."""
    out = []
    for i in sorted(partner):
        j = partner[i]
        assert j != i, f"self-pairing i={i}"
        rec = dict(records[i])
        rec["gen_text"] = records[j]["gen_text"]
        rec["paired_with_id"] = records[j]["id"]
        out.append(rec)
    return out


def build_shuffled_records(records, shift=1):
    """Keep each sample's own history / forecast values / ROI / domain / id; only replace gen_text
    with that of sample (i+shift)%N (guaranteed to be no self-pairing)."""
    n = len(records)
    assert n > shift, "the number of samples must exceed shift, otherwise a self-pairing occurs"
    out = []
    for i, rec in enumerate(records):
        j = (i + shift) % n
        new_rec = dict(rec)
        new_rec["gen_text"] = records[j]["gen_text"]
        new_rec["paired_with_id"] = records[j]["id"]
        out.append(new_rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--jsonl", default=DEFAULT_JSONL)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--shift", type=int, default=1)
    ap.add_argument("--pairing", choices=["shift", "random", "derangement"], default="derangement",
                    help="shift = row i paired with row i+shift (mostly same-dataset partners); "
                         "derangement = random derangement over the whole pool (for full evaluations); "
                         "random = independent redraw per sample (for subset evaluations)")
    ap.add_argument("--sample_n", type=int, default=0,
                    help=">0 draws this many samples at random by seed for evaluation (--limit takes the first N, "
                         "which all fall inside the first dataset; use this argument for sampled evaluations)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/eval/shuffled_visual_faithfulness")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hist_n", type=int, default=200)
    ap.add_argument("--judge_model", default="gpt-5.4")
    ap.add_argument("--api_base_url", default="https://api.openai.com/v1")
    ap.add_argument("--max_workers", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_records(args.npz, args.jsonl, args.parquet)
    n_pool = len(records)
    print(f"Loaded {n_pool} samples (npz={args.npz}, jsonl={args.jsonl})")
    if args.sample_n > 0:
        sel = sorted(random.Random(args.seed).sample(range(n_pool), min(args.sample_n, n_pool)))
    else:
        sel = list(range(n_pool))
    partner = build_pairing(records, sel, pairing=args.pairing, shift=args.shift, seed=args.seed)
    subset = apply_pairing(records, partner)
    same_ds = sum(1 for i in sel if records[i]["dataset_name"] == records[partner[i]]["dataset_name"])
    print(f"pairing={args.pairing} (shift={args.shift}), sampled {len(sel)}/{n_pool} (seed={args.seed}); "
          f"same-dataset rate {100 * same_ds / len(sel):.1f}% (history/forecast/domain unchanged, only gen_text swapped)")
    if args.limit > 0:
        subset = subset[: args.limit]
    client = build_openai_client(api_base_url=args.api_base_url)
    ckpt = os.path.join(args.out_dir, "visual_judge_raw.jsonl")
    print(f"Calling {args.judge_model} for plot-vs-text scores (mismatched pairing), n={len(subset)}, "
          f"workers={args.max_workers}, checkpoint={ckpt}")
    judge_results = run_visual_judge_batch(
        subset, client, model=args.judge_model, max_workers=args.max_workers,
        checkpoint_path=ckpt, hist_n=args.hist_n,
    )

    items = [{"score": judge_results.get(r["id"], {}).get("score")} for r in subset]
    overall = summarize(items)
    dom_groups = group_records_by_domain(subset)
    by_domain = {dom: summarize([{"score": judge_results.get(r["id"], {}).get("score")} for r in recs])
                 for dom, recs in dom_groups.items()}

    print(f"\n{'scope':10} {'n':>4} {'n_scored':>9} | {'mean':>6} | {'std':>6} | {'judge_fail_rate':>15}")
    print("-" * 66)
    for scope, s in [("ALL", overall)] + sorted(by_domain.items()):
        print(f"{scope:10} {s['n']:>4} {s['n_scored']:>9} | {s['mean']:6.3f} | {s['std']:6.3f} | "
              f"{s['judge_fail_rate']:15.4f}")

    summary_csv = os.path.join(args.out_dir, "visual_judge_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "n", "n_scored", "judge_fail_rate", "mean", "std",
                                           "dist_1", "dist_2", "dist_3", "dist_4", "dist_5"])
        w.writeheader()
        for scope, s in [("ALL", overall)] + sorted(by_domain.items()):
            row = {k: v for k, v in s.items() if k != "distribution"}
            row["scope"] = scope
            for k, v in s["distribution"].items():
                row[f"dist_{k}"] = v
            w.writerow(row)
    print(f"\nWrote {summary_csv}")

    per_sample_csv = os.path.join(args.out_dir, "per_sample.csv")
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "paired_with_id", "dataset_name", "domain", "score", "reason"])
        w.writeheader()
        for r in subset:
            jr = judge_results.get(r["id"], {})
            w.writerow({"id": r["id"], "paired_with_id": r["paired_with_id"],
                        "dataset_name": r["dataset_name"], "domain": r["domain"] or "Unknown",
                        "score": jr.get("score", ""), "reason": jr.get("reason", "")})
    print(f"Wrote {per_sample_csv}")


if __name__ == "__main__":
    main()
