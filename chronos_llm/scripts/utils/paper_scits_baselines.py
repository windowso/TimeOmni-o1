"""Aggregate the per-task CSV transcribed from the SciTS appendix (scits_per_task_understanding.csv) under the
SciTS main-table protocol, with MIMII excluded.

Protocol = the SciTS main-table rules + one exclusion:
- exclude MFU03 (=MIMII) => Manufacturing keeps only MFU01/02 (CWRU);
- primary metric per task: acc for MCQ (MEU04/ECU03), f1 for everything else;
- discipline score = simple mean over its tasks; when a discipline has a missing task value (model failure /
  TLS / INF), report "(completed/total)" instead of a score, following the SciTS convention.

Usage: python paper_scits_baselines.py <per_task.csv> [--out-csv OUT]
Output: a model x discipline markdown table + an optional long-format CSV.
"""
import argparse
import csv
from collections import defaultdict

_DISC = {"AS": "Astronomy", "EA": "Earth_Science", "BI": "Bioacoustics", "ME": "Meteorology",
         "EC": "Economics", "NE": "Neuroscience", "PH": "Physiology", "UR": "Urbanism",
         "MF": "Manufacturing", "RA": "Radar"}
_MCQ = {"MEU04", "ECU03"}
_EXCLUDE = {"MFU03"}
# main-table column order (same display order as the SciTS understanding table)
_ORDER = ["Astronomy", "Bioacoustics", "Earth_Science", "Economics", "Meteorology",
          "Manufacturing", "Neuroscience", "Physiology", "Radar", "Urbanism"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_task_csv")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    per = defaultdict(dict)   # model -> task_id -> value(or None)
    for r in csv.DictReader(open(args.per_task_csv)):
        tid = r["task_id"].strip()
        if tid in _EXCLUDE:
            continue
        v = r["acc"] if tid in _MCQ else r["f1"]
        per[r["model"].strip()][tid] = float(v) if v.strip() not in ("", "-") else None

    tasks_by_disc = defaultdict(list)
    all_tasks = sorted({t for m in per.values() for t in m})
    for t in all_tasks:
        tasks_by_disc[_DISC[t[:2]]].append(t)

    rows = []
    for model, tv in per.items():
        cells = {}
        for disc in _ORDER:
            tasks = tasks_by_disc[disc]
            vals = [tv.get(t) for t in tasks]
            done = [v for v in vals if v is not None]
            cells[disc] = (f"{sum(done)/len(done):.1f}" if len(done) == len(tasks)
                           else f"({len(done)}/{len(tasks)})")
        rows.append((model, cells))

    hdr = "| Model | " + " | ".join(_ORDER) + " |"
    print(hdr)
    print("|" + "---|" * (len(_ORDER) + 1))
    for model, cells in rows:
        print(f"| {model} | " + " | ".join(cells[d] for d in _ORDER) + " |")
    print(f"\n(MFU03=MIMII excluded; MCQ={sorted(_MCQ)} use acc, the rest f1; task counts="
          f"{ {d: len(tasks_by_disc[d]) for d in _ORDER} })")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model"] + _ORDER)
            for model, cells in rows:
                w.writerow([model] + [cells[d] for d in _ORDER])
        print(f"written -> {args.out_csv}")


if __name__ == "__main__":
    main()
