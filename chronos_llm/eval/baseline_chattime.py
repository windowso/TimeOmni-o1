"""ChatTime-1-7B (arXiv 2412.11376) zero-shot text-conditioned forecasting baseline.

Stand-alone script (run in a separate environment with a transformers 4.5x release compatible with
ChatTime's requirement of 4.53; PYTHONPATH points at the ChatTime repository root so its official
ChatTime class / discretisation / prompt are reused and its inference protocol is unchanged).

Protocol:
- official predict(): 10K-bin discretisation -> "foreign language" token generation, one chunk per
  16 steps, nanmedian over 8 samples within a chunk, autoregressive continuation => a **point
  forecast**. All 21 quantiles in the npz are set to that point trajectory (for a point forecast
  CRPS reduces to a weighted absolute error);
- text conditioning = plain_prompt (the official context argument, inserted under "Context knowledge
  you may consider:");
- history truncated to the most recent 512 points (its paper protocol hist<=512; LLaMA-2 4096-token
  hard limit); univariate, first channel.

Usage (GPU):
  python chronos_llm/eval/baseline_chattime.py --ckpt checkpoints/ChatTime-1-7B-Chat \
    --parquet data/forecast/mmtr_forecast_corpus.parquet --output outputs/eval/baseline_chattime/forecast_preds.npz
"""
import argparse
import os

import numpy as np
import pandas as pd

QUANTILE_LEVELS = np.array([
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
], dtype=np.float32)


def _first_channel(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    return np.asarray(v, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--hist_len", type=int, default=512)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from model.model import ChatTime   # ChatTime repository root is on PYTHONPATH

    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit]
    H = int(test["future_len"].max())
    N = len(test)
    print(f"[chattime] test N={N}, H={H}")

    P = np.full((N, len(QUANTILE_LEVELS), H), np.nan, dtype=np.float32)
    G = np.full((N, H), np.nan, dtype=np.float32)
    R = np.zeros((N, H), dtype=np.float32)
    V = np.zeros((N, H), dtype=bool)
    ids, ds_names = [], []

    model = None
    for i, row in test.iterrows():
        hist = _first_channel(row["history_values"])
        hist = hist[np.isfinite(hist)][-args.hist_len:]
        fut = _first_channel(row["future_values"])
        fl = int(row["future_len"])
        text = str(row.get("plain_prompt") or "")

        # the official class pins pred_len at construction => reset the attribute when fl differs per sample (predict only reads these two attributes)
        if model is None:
            model = ChatTime(model_path=args.ckpt, hist_len=args.hist_len, pred_len=fl,
                             num_samples=args.num_samples)
        model.pred_len = fl
        try:
            pred = model.predict(hist, context=text)     # (fl,) point trajectory
        except Exception as e:                            # parse failure etc.: leave NaN, counted as a failure
            print(f"  [warn] sample {i} failed: {e}")
            pred = np.full(fl, np.nan, dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)[:fl]
        P[i, :, :len(pred)] = pred[None, :]
        G[i, :fl] = fut[:fl]
        V[i, :fl] = np.isfinite(fut[:fl])
        rs, re_ = row.get("roi_start_idx"), row.get("roi_end_idx")
        if rs is not None and re_ is not None and not (pd.isna(rs) or pd.isna(re_)):
            a = max(0, int(rs) - int(row["past_len"]))
            b = min(fl, int(re_) - int(row["past_len"]))
            if b > a:
                R[i, a:b] = 1.0
        ids.append(str(row["id"]))
        ds_names.append(str(row["dataset_name"]))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
             quantile_levels=QUANTILE_LEVELS, ids=np.array(ids, dtype=object),
             dataset_names=np.array(ds_names, dtype=object))
    print(f"[chattime] npz -> {args.output}")


if __name__ == "__main__":
    main()
