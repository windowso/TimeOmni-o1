"""TimeOmni (SciTS; local copy of the TimeOmni-v2 repository) forecasting-side baseline inference: run the
fine-tuned checkpoint on our forecasting test split and write an npz in the same format as eval_forecast.py.

TimeOmni is one of the baselines that is fine-tuned on our forecasting training split before comparison.

Protocol decisions:
1. **No <FORECAST> token routing, no horizon head**: call ``model.forecast_infer`` directly and pass the
   **true future_len** via ``override_horizon_per_sample`` -- the same information every other baseline
   gets (the text already says "predict the next N points"). The horizon-head prediction and the routing
   first token are still logged in a side-car jsonl for reference; they do not affect the metrics.
2. **use_scaler=False** (raw scale): consistent with the fine-tuning protocol (its training forecast dataset
   does no rescaling; TimesFM normalises internally). Its bundled infer_benchmark.py uses use_scaler=True
   and leaves the output in standardised space, where raw-scale metrics cannot be computed, so it is not used.
3. **21 quantiles interpolated from its 10-quantile head**: the TimesFM 2.5 head outputs [point, q0.1..q0.9]
   (index 0 is the point forecast, nominally used as 0.5; indices 1..9 map monotonically to 0.1..0.9). We take
   the nine points q0.1..q0.9 and linearly interpolate to our 21 quantile levels; beyond both ends (<0.1 /
   >0.9) the nearest quantile is held flat (np.interp clamps by default). The raw 10-quantile array is
   stored in the npz as well (key=raw_quantiles).
4. Per-domain splits rely on the dataset_names stored in the npz (test jsonl row order == parquet test row
   order, aligned position by position with an id assert on every row to prevent cross-talk).

Usage (GPU node, TimeOmni's own Python environment):
  python chronos_llm/eval/baseline_timeomni.py \
    --model_path <EXP>/epoch_N/pytorch_model/mp_rank_00_model_states.pt \
    --output outputs/eval/baseline_timeomni_forecast/preds_epochN.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

TIMEOMNI_ROOT = "third_party/TimeOmni"
REPO = "."
DEFAULT_TEST_JSONL = f"{REPO}/outputs/eval/baseline_timeomni_forecast/data/eventcap_forecast_test.jsonl"
DEFAULT_PARQUET = ("data/forecast/"
                   "mmtr_forecast_corpus.parquet")

# The 21 quantile levels, identical to infer_forecast.py / the other baselines (Chronos-2 convention)
QUANTILE_LEVELS = np.array([
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
], dtype=np.float32)
TIMESFM_LEVELS = np.arange(1, 10, dtype=np.float32) / 10.0  # q0.1..q0.9 (head indices 1..9)


def interp_quantiles(q9):
    """(H, 9) monotone quantiles -> (21, H): linear interpolation, flat extrapolation at both ends."""
    H = q9.shape[0]
    out = np.empty((len(QUANTILE_LEVELS), H), dtype=np.float32)
    for h in range(H):
        out[:, h] = np.interp(QUANTILE_LEVELS, TIMESFM_LEVELS, q9[h])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help=".../epoch_N/pytorch_model/mp_rank_00_model_states.pt")
    ap.add_argument("--test_jsonl", default=DEFAULT_TEST_JSONL)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help=">0 runs only the first N rows (smoke test)")
    ap.add_argument("--log_routing", action="store_true", default=True)
    args = ap.parse_args()

    model_path = os.path.abspath(args.model_path)
    output = os.path.abspath(args.output)
    test_jsonl = os.path.abspath(args.test_jsonl)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Its config.json contains ./checkpoints relative paths, so the TimeOmni-v2 root must be the cwd
    sys.path.insert(0, TIMEOMNI_ROOT)
    os.chdir(TIMEOMNI_ROOT)
    from models import TimeOmni_v2
    from data_provider.dataset import Dataset_Unified, Collator_Unified
    from torch.utils.data import DataLoader

    print(f"[timeomni] loading checkpoint {model_path}")
    model = TimeOmni_v2.TimeOmniV2.load_checkpoint(model_path).eval().cuda()

    dataset = Dataset_Unified(
        tokenizer=model.tokenizer, jsonl_file_path=test_jsonl, base_dir=None,
        use_scaler=False, unfold_channels=False, inference_mode=True,
    )
    if args.limit:
        dataset.data = dataset.data[: args.limit]
    collator = Collator_Unified(tokenizer=model.tokenizer, infer_mode=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collator, drop_last=False)

    # Parquet test split for alignment (row order == the order the converter wrote the jsonl)
    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    N = args.limit or len(test)
    H = int(test["future_len"].iloc[:N].max()) if args.limit else int(test["future_len"].max())

    P = np.full((N, len(QUANTILE_LEVELS), H), np.nan, dtype=np.float32)
    RAWQ = np.full((N, 10, H), np.nan, dtype=np.float32)
    G = np.full((N, H), np.nan, dtype=np.float32)
    R = np.zeros((N, H), dtype=np.float32)
    V = np.zeros((N, H), dtype=bool)
    ids, ds_names = [], []
    side_rows = []

    pos = 0
    mono_viol = tot_pairs = 0
    for batch in loader:
        ts_list = [t.cuda() for t in batch["input_ts_list"]]
        input_ids = batch["input_ids"].cuda()
        sr = torch.tensor(batch["input_ts_sr_list"], device="cuda")
        x_lens = torch.tensor(batch["input_ts_len_list"], device="cuda")
        gt_lens = [int(g.shape[0]) for g in batch["gt_ts_list"]]

        with torch.no_grad():
            point_outs, quant_outs = model.forecast_infer(
                ts_list, input_ids, gt_ts=None, x_lens=x_lens, sr=sr,
                override_horizon_per_sample=gt_lens,
            )
            # Reference log: the horizon head's prediction (not used for the metrics)
            horizon_head_pred = None
            try:
                _, _, llm_chunk, llm_chunk_mask = model._build_timesfm_ctx(
                    input_ids, time_series=ts_list, x_lens=x_lens, sr=sr, return_llm_chunk=True)
                if llm_chunk is not None:
                    hp = model._predict_horizon(llm_chunk, llm_chunk_mask)
                    horizon_head_pred = model._decode_predicted_horizon(
                        hp, min_h=1, max_h=model.horizon_max_length).tolist()
            except Exception as e:  # a failure of the reference log does not affect the main path
                print(f"[timeomni] horizon-head log failed: {e}")

        for i in range(len(gt_lens)):
            row = test.iloc[pos]
            assert str(batch["ids"][i]) == str(row["id"]), \
                f"id mismatch at position {pos}: jsonl={batch['ids'][i]} parquet={row['id']}"
            fl = int(row["future_len"])
            assert fl == gt_lens[i], f"horizon mismatch at position {pos}: {fl} vs {gt_lens[i]}"

            q = quant_outs[i].float().cpu().numpy()  # (fl, C=1, 10)
            assert q.shape[0] == fl and q.shape[1] == 1 and q.shape[2] == 10, q.shape
            q = q[:, 0, :]                            # (fl, 10)
            q9 = q[:, 1:]                             # indices 1..9 = q0.1..0.9
            d = np.diff(q9, axis=1)
            mono_viol += int((d < -1e-6).sum()); tot_pairs += d.size
            q9 = np.sort(q9, axis=1)                  # fix_quantile_crossing fallback

            P[pos, :, :fl] = interp_quantiles(q9)
            RAWQ[pos, :, :fl] = q.T
            fut = np.asarray(row["future_values"], dtype=np.float32)[:fl]
            G[pos, :fl] = fut
            V[pos, :fl] = np.isfinite(fut)
            rs, re_ = row.get("roi_start_idx"), row.get("roi_end_idx")
            if rs is not None and re_ is not None and not (pd.isna(rs) or pd.isna(re_)):
                a = max(0, int(rs) - int(row["past_len"]))
                b = min(fl, int(re_) - int(row["past_len"]))
                if b > a:
                    R[pos, a:b] = 1.0
            ids.append(str(row["id"]))
            ds_names.append(str(row["dataset_name"]))
            side_rows.append({
                "id": str(row["id"]), "dataset_name": str(row["dataset_name"]),
                "future_len": fl,
                "horizon_head_pred": (horizon_head_pred[i] if horizon_head_pred else None),
                "point_median": [round(float(x), 4) for x in q[:, 0].tolist()],
            })
            pos += 1
        print(f"  {pos}/{N}")

    print(f"[timeomni] quantile monotonicity violation rate (before the sort fix): {mono_viol}/{tot_pairs} = "
          f"{mono_viol / max(tot_pairs,1):.2e}")
    np.savez_compressed(output, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
                        quantile_levels=QUANTILE_LEVELS, raw_quantiles=RAWQ,
                        ids=np.array(ids, dtype=object),
                        dataset_names=np.array(ds_names, dtype=object))
    with open(output[:-4] + ".jsonl", "w", encoding="utf-8") as f:
        for r in side_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[timeomni] {pos} forecasts saved to {output}")


if __name__ == "__main__":
    main()
