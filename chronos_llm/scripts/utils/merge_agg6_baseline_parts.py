"""Merge the <stem>.partNN.jsonl files of a baseline batch-run output directory back into <stem>.jsonl, in order.

Companion of split_agg6_for_baselines.py (contiguous chunking => concatenating by part index restores the original
row order); "agg6" denotes the MMTR understanding test pool. Checks before merging:
- the number of parts per dataset matches the split output (a missing part = the batch run is incomplete; that
  dataset is refused);
- the merged row count == the row count of the original test jsonl (for smoke directories produced with the
  adapter's --limit, skip this with --no_strict).

While merging, `ground_truth` is normalised from list to str: the baseline adapters store `gt_text` as is (a list,
the format consumed by eval_understanding.py in the SciTS pipeline), whereas the agg6 scoring pipeline
(summarize_agg6_eval / eval_opentslm_answer) consumes the str format written by our infer_understanding.py
(= understanding_dataset._answer(): `gt_text[0]`, or json.dumps(gt_result) if empty). The normalisation lives in
this merge bridge so that neither main code path has to change.

    python chronos_llm/scripts/utils/merge_agg6_baseline_parts.py \
        outputs/eval/baseline_chattime/understanding_agg6
"""
import argparse
import json
import os

from split_agg6_for_baselines import PARTS_DIR, src_files


def normalize_line(line: str) -> str:
    """`ground_truth`/`input_text` list->str, matching the meta convention of understanding_dataset
    (`_answer()` and `input_text[0]`; the latter feeds the VeriTime question-text regex used by by_domain to extract the domain)."""
    r = json.loads(line)
    changed = False
    gt = r.get("ground_truth")
    if isinstance(gt, list):
        ans = gt[0] if gt and gt[0] else ""
        if not ans and r.get("gt_result"):
            ans = json.dumps(r["gt_result"], ensure_ascii=False)
        r["ground_truth"] = str(ans)
        changed = True
    it = r.get("input_text")
    if isinstance(it, list):
        r["input_text"] = str(it[0]) if it and it[0] else ""
        changed = True
    if changed:
        return json.dumps(r, ensure_ascii=False) + "\n"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="baseline batch-run output directory (containing *.partNN.jsonl)")
    ap.add_argument("--no_strict", action="store_true", help="skip the row-count check (for --limit smoke directories)")
    args = ap.parse_args()

    expected = {}   # stem -> (n_parts, n_rows)
    for jf in src_files():
        stem = os.path.basename(jf)[: -len(".jsonl")]
        n_parts = len([f for f in os.listdir(PARTS_DIR) if f.startswith(stem + ".part")])
        expected[stem] = (n_parts, sum(1 for _ in open(jf)))

    n_ok = 0
    for stem, (n_parts, n_rows) in expected.items():
        parts = sorted(f for f in os.listdir(args.out_dir) if f.startswith(stem + ".part"))
        if len(parts) != n_parts:
            print(f"[SKIP] {stem}: only {len(parts)}/{n_parts} parts present, batch run incomplete")
            continue
        merged = os.path.join(args.out_dir, stem + ".jsonl")
        n = 0
        with open(merged, "w") as out:
            for p in parts:
                for line in open(os.path.join(args.out_dir, p)):
                    out.write(normalize_line(line))
                    n += 1
        if not args.no_strict and n != n_rows:
            os.remove(merged)
            print(f"[FAIL] {stem}: merged {n} rows != expected {n_rows}; partial output removed")
            continue
        n_ok += 1
        print(f"[OK] {stem}.jsonl <- {len(parts)} parts, {n} rows")
    print(f"Merged {n_ok}/{len(expected)} datasets")


if __name__ == "__main__":
    main()
