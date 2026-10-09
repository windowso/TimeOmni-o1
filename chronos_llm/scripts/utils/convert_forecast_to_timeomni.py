"""Convert our forecast parquet (the MMTR forecasting corpus) into TimeOmni-v2's forecast JSONL
format, used to fine-tune the TimeOmni forecasting-side baseline.

A TimeOmni-v2 forecast sample = one JSON per line (fields modelled on the ETT rows of its
Release_train_standard), with history/future stored as separate .npy files (its
data_provider/dataset.py load_ts_data reshapes a 1-D npy to (T,1) automatically); the event text
goes into input_text[0] (its analyze_with_ts path encodes that text with the LLM, compresses it
with a Q-former and injects it into TimesFM via cross-attention, so the baseline is
text-conditioned).

Splits:
- train  = 95% of parquet split=train (fixed seed)
- valid  = the other 5% of train (fed to the training script's --forecast_test_jsonl_file_path
  for per-epoch validation metrics; the test split is not used during training)
- test   = parquet split=test, **row order == parquet test row order** (the final npz is aligned
  by position to roi_mask / dataset_name, preventing id mix-ups)

Usage (pure CPU data conversion):
  python chronos_llm/scripts/utils/convert_forecast_to_timeomni.py \
    --parquet data/forecast/mmtr_forecast_corpus.parquet \
    --out_dir outputs/eval/baseline_timeomni_forecast/data
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

INSTRUCTION_TAIL = (
    "Please predict the next {n} time series points given information above."
)


def row_to_entry(row, ts_dir, idx, split):
    hist = np.asarray(row["history_values"], dtype=np.float32)
    fut = np.asarray(row["future_values"], dtype=np.float32)
    assert hist.ndim == 1 and fut.ndim == 1, f"expect 1-D series, got {hist.shape}/{fut.shape}"
    fl = int(row["future_len"])
    assert fl == len(fut), f"future_len={fl} != len(future_values)={len(fut)}"

    hist_path = os.path.join(ts_dir, f"{split}_{idx:05d}_hist.npy")
    gt_path = os.path.join(ts_dir, f"{split}_{idx:05d}_gt.npy")
    np.save(hist_path, hist)
    np.save(gt_path, fut)

    text = str(row["plain_prompt"]).strip()
    assert text, f"empty plain_prompt at {split}[{idx}] id={row['id']}"
    input_text = f"{text}\n\n{INSTRUCTION_TAIL.format(n=fl)}"

    return {
        "dataset_name": str(row["dataset_name"]),
        "domain": str(row["dataset_name"]).split("/")[0],
        "task": "Forecasting",
        "scene": str(row["dataset_name"]),
        "id": str(row["id"]),
        "uid": f"eventcap-{split}-{idx:05d}-{row['id']}",
        "data_type": "npy",
        "input_ts": {
            "already_segment": False,
            "channel": 1,
            "channel_detail": ["target"],
            "original": {
                "ori_path": os.path.abspath(hist_path),
                "ori_length": int(len(hist)),
                "ori_timestamps": [],
                "ori_fs": str(row.get("freq", "")),
            },
        },
        "input_text": [input_text],
        "gt_text": [None, None],
        "gt_result": {"timestamps": [], "channel": 1, "channel_detail": ["target"]},
        "gt_ts": {"path": os.path.abspath(gt_path), "length": fl},
        "old_dict": {},
    }


def write_jsonl(df, ts_dir, split, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(df.iterrows()):
            f.write(json.dumps(row_to_entry(row, ts_dir, idx, split), ensure_ascii=False) + "\n")
            n += 1
    print(f"[convert] {split}: {n} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/forecast/mmtr_forecast_corpus.parquet")
    ap.add_argument("--out_dir", default="outputs/eval/baseline_timeomni_forecast/data")
    ap.add_argument("--valid_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ts_dir = os.path.join(args.out_dir, "ts")
    os.makedirs(ts_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    train_all = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(train_all))
    n_valid = int(round(len(train_all) * args.valid_ratio))
    valid_idx = np.sort(perm[:n_valid])
    train_idx = np.sort(perm[n_valid:])
    train = train_all.iloc[train_idx].reset_index(drop=True)
    valid = train_all.iloc[valid_idx].reset_index(drop=True)

    write_jsonl(train, ts_dir, "train", os.path.join(args.out_dir, "eventcap_forecast_train.jsonl"))
    write_jsonl(valid, ts_dir, "valid", os.path.join(args.out_dir, "eventcap_forecast_valid.jsonl"))
    write_jsonl(test, ts_dir, "test", os.path.join(args.out_dir, "eventcap_forecast_test.jsonl"))
    print(f"[convert] parquet={args.parquet}")
    print(f"[convert] train/valid/test = {len(train)}/{len(valid)}/{len(test)}")


if __name__ == "__main__":
    main()
