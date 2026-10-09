"""Adapt the TimeReasoner baseline's raw_log.jsonl into the {id, gen_text, gt_conclusion} jsonl
format consumed by both text_alignment_eval.py and forecast_visual_faithfulness.py.

TimeReasoner is the forecasting baseline that generates natural-language reasoning text, so its
output can be scored with the same text evaluations as our model.

``gen_text`` = TimeReasoner's ``reasoning`` (the raw chain of thought) + ``answer`` (the final reply,
explanation + JSON value block) concatenated. Note that the ROI extraction regex
(``rl/reward.extract_roi``) can fire on text that describes the history/future window boundaries
(e.g. "history window is indices [0,480)") rather than a region of interest. ``gt_conclusion`` is
back-filled by id from the parquet's ``conclusion`` column (the same ground-truth source as the
``gt_conclusion`` field in our own model's forecast_preds.jsonl); the ``roi`` field is left empty to
skip build_records' coordinate self-consistency assertion (TimeReasoner has no relative ROI
coordinates of its own to cross-check).
"""
import argparse
import json

import pyarrow.parquet as pq


def build_adapted_jsonl(raw_log_path, parquet_path, out_path, split="test"):
    rows = []
    with open(raw_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pq.read_table(parquet_path, columns=["id", "split", "conclusion"]).to_pandas()
    df = df[df["split"] == split]
    assert df["id"].is_unique, f"ids are not unique within {parquet_path} split={split!r}"
    concl_by_id = dict(zip(df["id"], df["conclusion"]))

    out_rows = []
    missing = 0
    for r in rows:
        sid = r["id"]
        if sid not in concl_by_id:
            missing += 1
            continue
        reasoning = (r.get("reasoning") or "").strip()
        answer = (r.get("answer") or "").strip()
        gen_text = (reasoning + "\n\n" + answer).strip() if reasoning else answer
        out_rows.append({
            "id": sid,
            "gen_text": gen_text,
            "gt_conclusion": concl_by_id[sid],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"loaded {len(rows)} raw TimeReasoner records, {len(out_rows)} matched a parquet ground truth"
          f" ({missing} ids not found in the parquet {split} split), written -> {out_path}")
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_log", default="outputs/eval/baseline_timereasoner/raw_log.jsonl")
    ap.add_argument("--parquet", default=(
        "data/forecast/"
        "mmtr_forecast_corpus.parquet"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/eval/baseline_timereasoner/adapted_text_eval.jsonl")
    args = ap.parse_args()
    build_adapted_jsonl(args.raw_log, args.parquet, args.out, split=args.split)


if __name__ == "__main__":
    main()
