r"""Summarize the "chronos2 + multi-task classification heads" (fine-tuned TSFM) understanding row
by **domain** and report its **feasible coverage** per domain.

The classification heads have no text interface, so only the subtasks with a fixed question and a
closed label set can be answered by them; subtasks with per-question templates, per-sample
observations, per-question free-text options or numeric answers **cannot be answered
structurally** => those cells are "not applicable", which is different from a score of 0 of a
baseline that ran but produced mismatching output.

This script therefore produces three things:
1. per domain, the **feasible count / total count** (coverage) -- decides whether the cell is `-`
   (entirely infeasible), `x%` partially feasible, or fully feasible;
2. the classification heads' score on the **feasible subset** of that domain (same protocol as the
   main table: each (subtask x domain) cell scored with its native protocol, then count-weighted);
3. the other runs' scores on the **same id subset**, for a like-for-like comparison.

    python chronos_llm/scripts/utils/headcls_by_domain.py \
        --headcls outputs/baseline_headcls/infer \
        ours=outputs/eval/understanding/infer \
        nocot=outputs/eval/understanding_nocot/infer cot=outputs/eval/understanding_cot/infer
"""
import argparse
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "chronos_llm/scripts/utils"))
from summarize_agg6_by_domain import (  # noqa: E402
    ORDER as DOMAIN_ORDER, cell_score, domain_of, domain_summary,
)


def load_dir(d):
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl") or ".part" in fn:
            continue
        for line in open(os.path.join(d, fn), encoding="utf-8"):
            rows.append(json.loads(line))
    return rows


def by_domain(rows, keep_ids=None):
    """-> ({domain: score}, {domain: distinct sample count}), same protocol as summarize_agg6_by_domain."""
    buckets = defaultdict(list)
    n_samples = defaultdict(int)
    for r in rows:
        if keep_ids is not None and r.get("id") not in keep_ids:
            continue
        buckets[(r.get("task"), domain_of(r))].append(r)
    cells = defaultdict(dict)
    for (task, dom), rs in buckets.items():
        n_samples[dom] += len(rs)
        for metric, score in cell_score(task, rs):
            cells[dom][(task, metric)] = (score, len(rs))
    return {d: domain_summary(c) for d, c in cells.items()}, dict(n_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=infer_dir (runs to compare)")
    ap.add_argument("--headcls", required=True, help="infer directory of the classification-head arm")
    ap.add_argument("--ref", default=None,
                    help="reference arm used to fill in input_text for the classification-head rows "
                         "(defaults to the first run)")
    args = ap.parse_args()

    runs = [r.split("=", 1) for r in args.runs]
    ref_dir = args.ref or runs[0][1]
    ref_rows = load_dir(ref_dir)
    id2text = {r.get("id"): r.get("input_text", "") for r in ref_rows}

    hc = load_dir(args.headcls)
    for r in hc:                      # the VeriTime Scenario family carries its domain in the question text
        r.setdefault("input_text", id2text.get(r.get("id"), ""))
    hc_ids = {r.get("id") for r in hc}
    hc_score, hc_n = by_domain(hc)

    full, sub = {}, {}
    for label, d in runs:
        rows = load_dir(d)
        full[label] = by_domain(rows)
        sub[label] = by_domain(rows, keep_ids=hc_ids)

    total_n = full[runs[0][0]][1]
    labels = [l for l, _ in runs]
    print(f"samples feasible for the classification heads: {len(hc_ids)} / {sum(total_n.values())} "
          f"({100*len(hc_ids)/sum(total_n.values()):.1f}%)\n")
    head = ["discipline", "total n", "feasible n", "coverage", "chronos2+heads"]
    head += [f"{l} (feasible subset)" for l in labels] + [f"{l} (full, main-table value)" for l in labels]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for dom in DOMAIN_ORDER:
        tn = total_n.get(dom, 0)
        if not tn:
            continue
        hn = hc_n.get(dom, 0)
        cov = f"{100*hn/tn:.0f}%" if hn else "-"
        cells = [dom, str(tn), str(hn) if hn else "-", cov,
                 f"{hc_score[dom]:.2f}" if hn else "**not applicable**"]
        for l in labels:
            cells.append(f"{sub[l][0][dom]:.2f}" if hn else "-")
        for l in labels:
            cells.append(f"{full[l][0][dom]:.2f}")
        print("| " + " | ".join(cells) + " |")

    # Aggregation protocol: equal-weight / count-weighted over the feasible domains only
    doms_ok = [d for d in DOMAIN_ORDER if hc_n.get(d) and d != "Unlabeled"]
    def avg(sc, w=None):
        vals = [sc[d] for d in doms_ok]
        if w is None:
            return sum(vals) / len(vals)
        ws = [w[d] for d in doms_ok]
        return sum(v * x for v, x in zip(vals, ws)) / sum(ws)
    print(f"\n**Comparison over the {len(doms_ok)} feasible disciplines** (equal-weight / count-weighted, "
          f"weights = feasible counts):")
    print(f"- chronos2+heads: {avg(hc_score):.2f} / {avg(hc_score, hc_n):.2f}")
    for l in labels:
        print(f"- {l} (same subset): {avg(sub[l][0]):.2f} / {avg(sub[l][0], hc_n):.2f}")
        print(f"- {l} (full domain, for reference only): {avg(full[l][0]):.2f} / {avg(full[l][0], hc_n):.2f}")


if __name__ == "__main__":
    main()
