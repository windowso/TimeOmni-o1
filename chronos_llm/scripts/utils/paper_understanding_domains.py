"""Aggregate per-dataset understanding metrics into discipline(domain)-level main-table numbers
following the SciTS protocol.

SciTS aggregation rules (from the main-table caption of the SciTS paper):
- discipline score = **simple arithmetic mean** of the primary metric of each task in that
  discipline (not weighted by instance count);
- task primary metric: understanding tasks use F1, MCQ tasks use accuracy (our MCQ = the
  MT_bench Stock/Temperature files whose third file-name segment is QA);
- CWRU multi-label lives in the suffix CSV and is split into two tasks as in SciTS
  (MFU01=position, MFU02=diameter), F1 each;
- MIMII_Due is excluded by default.

Input is the metrics CSV produced by eval_understanding.py (file-name column of the form
`Name-Domain-TaskType-Sub-test-N[_fixed].jsonl`, domain = 2nd segment, task type = 3rd segment);
the TimeOmni baseline CSV and our mid_eval CSV share the layout and can both be consumed directly.

Usage:
  python paper_understanding_domains.py <metrics.csv> [--suffix-csv X] [--label NAME] \
      [--exclude MIMII_Due] [--out-csv OUT]
Without --suffix-csv the sibling `{stem}_suffix.csv` is picked up automatically. Prints per-task
detail + per-domain aggregation; --out-csv appends a long table
(label,domain,task,metric_kind,value) so several models can be merged into the main table.
"""
import argparse
import csv
import os


def _rows(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get("filename", "").strip() not in ("", "AVERAGE")]


def parse_tasks(metrics_csv, suffix_csv, exclude):
    """Return [(domain, task_name, metric_kind, value)], one primary metric per task (SciTS protocol)."""
    tasks = []
    for r in _rows(metrics_csv):
        fn = r["filename"]
        stem = fn.split("-test-")[0]
        parts = stem.split("-")
        name, domain, ttype = parts[0], parts[1], parts[2] if len(parts) > 2 else "?"
        if any(e in fn for e in exclude):
            continue
        # SciTS: MCQ (our QA) uses accuracy, the other understanding tasks use F1
        if ttype == "QA":
            kind, val = "acc(MCQ)", r["accuracy"]
        else:
            kind, val = "f1", r["f1_score"]
        tasks.append((domain, stem, kind, float(val)))
    if suffix_csv and os.path.exists(suffix_csv):
        for r in _rows(suffix_csv):
            fn = r["filename"]
            if any(e in fn for e in exclude):
                continue
            stem = fn.split("-test-")[0]
            domain = stem.split("-")[1]
            # CWRU dual label = two SciTS tasks (position/diameter), F1 each
            for sub in ("position", "diameter"):
                col = f"f1_score_{sub}"
                if col in r and r[col] not in ("", None):
                    tasks.append((domain, f"{stem}[{sub}]", "f1", float(r[col])))
    return tasks


def aggregate(tasks):
    by_dom = {}
    for domain, task, kind, val in tasks:
        by_dom.setdefault(domain, []).append((task, kind, val))
    return {d: sum(v for _, _, v in lst) / len(lst) for d, lst in sorted(by_dom.items())}, by_dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv")
    ap.add_argument("--suffix-csv", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--exclude", default="MIMII_Due", help="comma-separated substrings to exclude")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    suffix = args.suffix_csv
    if suffix is None:
        stem, ext = os.path.splitext(args.metrics_csv)
        cand = f"{stem}_suffix{ext}"
        suffix = cand if os.path.exists(cand) else None
    exclude = [e.strip() for e in args.exclude.split(",") if e.strip()]
    label = args.label or os.path.basename(os.path.dirname(os.path.abspath(args.metrics_csv)))

    tasks = parse_tasks(args.metrics_csv, suffix, exclude)
    dom_scores, by_dom = aggregate(tasks)

    print(f"===== {label} (SciTS protocol: simple mean of task primary metrics; excluded {exclude}) =====")
    for d, score in dom_scores.items():
        print(f"\n[{d}] = {score:.2f}  ({len(by_dom[d])} tasks)")
        for task, kind, val in by_dom[d]:
            print(f"    {val:6.2f}  {kind:<9} {task}")
    overall = sum(dom_scores.values()) / len(dom_scores)
    task_avg = sum(v for _, _, _, v in tasks) / len(tasks)
    print(f"\nDomain macro mean = {overall:.2f}   Task macro mean = {task_avg:.2f}   "
          f"(domains={len(dom_scores)}, tasks={len(tasks)})")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["label", "domain", "task", "metric_kind", "value"])
            for domain, task, kind, val in tasks:
                w.writerow([label, domain, task, kind, f"{val:.2f}"])
            for d, score in dom_scores.items():
                w.writerow([label, d, "__DOMAIN_MEAN__", "scits", f"{score:.2f}"])
        print(f"appended -> {args.out_csv}")


if __name__ == "__main__":
    main()
