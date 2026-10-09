"""UniTS (NeurIPS 2024, mims-harvard/UniTS) fine-tuned forecasting baseline (purely numeric unified model).

Standalone script: does not import chronos_llm; the official UniTS repository and units_timm_shim
must be on PYTHONPATH (launcher: chronos_llm/scripts/utils/baseline_units.sh). Only needs
torch/pandas/numpy + the timm shim, no additional packages.

Protocol (aligned with the other baseline npz files / evaluation pipeline):
- **Fine-tuning**: UniTS is compared after fine-tuning on our forecast train split. The backbone is initialised from the
  officially released units_x128_pretrain_checkpoint.pth (d_model=128/e_layers=3/n_heads=8/
  patch=stride=16/prompt_num=10, backbone+forecast_head ~3.1M parameters), loaded with strict=False
  for backbone/head only; the prompt/mask tokens of our data sources are freshly initialised
  following the official few-shot new-data protocol (prompt normal(0.02), mask zeros -- copied from
  Model.__init__).
- **UniTS has no text interface** => only the numeric series is fed (history->future); prompt/background
  text is entirely unused.
- **Variable-length adaptation**: the official implementation bakes seq_len/pred_len into the dataset
  config and requires (pred_len - patch right-pad) % 16 == 0. This script groups batches by
  (data source, history length L) (equal lengths within a group => no history padding and
  uncontaminated instance-norm statistics); per batch pred_len_eff = padding + 16*ceil((max_fl -
  padding)/16) >= the group's max future_len, and the forward uses forecast_flex() -- a line-by-line
  copy of the official Model.forecast() with only the token count derived from the actual input shape
  instead of the config constant; all module calls go through the official weights.
- **Loss**: masked normalized MSE -- pred/gt are normalised with the same instance-norm as the official
  tokenize (mean, sqrt(var+1e-5)) and averaged only within each sample's true future_len (on a
  mixed-scale corpus a raw MSE would be dominated by large-scale series; the official multi-task
  training uses per-dataset MSE, and per-sample scale normalisation is the counterpart for our
  single-corpus mixed batches).
- **Epoch selection**: 10% of the train split is held out as val (fixed seed); the best epoch is
  chosen by val masked-NMSE and test is run once.
- **Point forecast => CRPS convention**: UniTS has no quantile output, all 21 quantiles take the point
  forecast (CRPS reduces to a weighted absolute error) -- the same protocol as ChatTime/TimeOmni-VL.
- The npz has exactly the same format as doublecast/chattime (pred_quantiles/gt/roi_mask/valid_mask/
  quantile_levels/ids/dataset_names, original test-split row order => compatible with the
  position-mapped domain split).

Usage (the model has only ~8M parameters; a CPU node suffices):
  python chronos_llm/eval/baseline_units.py --mode both \
    --units_repo third_party/UniTS --ckpt checkpoints/UniTS/units_x128_pretrain_checkpoint.pth \
    --parquet data/forecast/mmtr_forecast_corpus.parquet --output_dir outputs/eval/baseline_units
"""
import argparse
import copy
import math
import os
import random
import sys
import types

import numpy as np
import pandas as pd
import torch

QUANTILE_LEVELS = np.array([
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
], dtype=np.float32)   # the same 21 levels as chronos-2 / all existing npz files

PATCH = 16


def _first_channel(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    return np.asarray(v, dtype=np.float32)


def _sanitize(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def build_model(units_repo, ckpt_path, source_names, device):
    sys.path.insert(0, units_repo)
    from models.UniTS import Model  # noqa: E402

    margs = types.SimpleNamespace(
        d_model=128, n_heads=8, e_layers=3, patch_len=PATCH, stride=PATCH,
        prompt_num=10, dropout=0.1)
    # One prompt/mask token per data source (the official multi-task design); seq/pred_len only satisfy
    # the constructor, the actual forward derives them from the input shape in forecast_flex.
    configs_list = [
        [f"LTF_{_sanitize(s)}", {
            "task_name": "long_term_forecast", "dataset": _sanitize(s),
            "seq_len": 576, "pred_len": 288, "enc_in": 1, "dec_in": 1, "c_out": 1,
        }] for s in source_names
    ]
    model = Model(margs, configs_list, pretrain=False)

    if ckpt_path:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw["student"] if isinstance(raw, dict) and "student" in raw else raw
        sd = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in sd.items() if "cls_prompts" not in k}
        msg = model.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(msg.unexpected_keys)
        print(f"[units] loaded {loaded} tensors from ckpt; "
              f"missing(new tokens)={len(msg.missing_keys)} unexpected(their tokens)={len(msg.unexpected_keys)}")
        assert not any(k.startswith(("blocks", "patch_embeddings", "forecast_head", "prompt2forecat",
                                     "position_embedding")) for k in msg.missing_keys), \
            "backbone weights were not fully loaded!"
    return model.to(device)


def forecast_flex(model, x, ds_key, pred_len):
    """Variable-length version of the official Model.forecast(): the token count is derived from the
    actual input; all module calls go through the official implementation.

    x: (B, L, 1); requires (pred_len - padding) % PATCH == 0 and >= PATCH.
    Returns (B, pred_len, 1).
    """
    L = x.shape[1]
    prefix_prompt = model.prompt_tokens[ds_key]
    task_prompt = model.mask_tokens[ds_key]

    x, means, stdev, n_vars, padding = model.tokenize(x)
    task_prompt_num = (pred_len - padding) // PATCH
    assert task_prompt_num >= 1 and (pred_len - padding) % PATCH == 0, (L, pred_len, padding)

    x = model.prepare_prompt(
        x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name="forecast")
    seq_token_len = x.shape[-2] - prefix_prompt.shape[2]
    x = model.backbone(x, prefix_prompt.shape[2], seq_token_len)
    x = model.forecast_head(x, L + padding + pred_len - padding, seq_token_len)  # = L + pred_len
    x = x[:, -pred_len:]
    x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
    x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
    return x


def eff_pred_len(L, h):
    """Smallest pred_len_eff >= h satisfying the fold constraint: padding + 16k with k>=1."""
    padding = (PATCH - L % PATCH) % PATCH
    k = max(1, math.ceil((h - padding) / PATCH))
    return padding + PATCH * k


def load_rows(parquet, split):
    df = pd.read_parquet(parquet)
    part = df[df["split"] == split].reset_index(drop=True)
    rows = []
    for i, r in part.iterrows():
        hist = _first_channel(r["history_values"])
        fut = _first_channel(r["future_values"])
        rows.append({
            "idx": i, "id": str(r["id"]), "source": str(r["dataset_name"]),
            "hist": hist, "fut": fut, "fl": int(r["future_len"]),
            "past_len": int(r["past_len"]),
            "roi": (r.get("roi_start_idx"), r.get("roi_end_idx")),
        })
    return rows


def group_key(row):
    return (row["source"], len(row["hist"]))


def make_batches(rows, bs, shuffle, rng):
    groups = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    batches = []
    for _, g in groups.items():
        if shuffle:
            rng.shuffle(g)
        batches += [g[i:i + bs] for i in range(0, len(g), bs)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def norm_stats(hist_t):
    """The same instance-norm statistics as the official tokenize. hist_t: (B, L, 1)"""
    means = hist_t.mean(1, keepdim=True)
    stdev = torch.sqrt(torch.var(hist_t, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return means, stdev


def batch_forward(model, batch, device):
    """Forward of one equal-length batch; returns (pred (B,pred_len), fl list)."""
    hist = torch.tensor(np.stack([r["hist"] for r in batch]), dtype=torch.float32,
                        device=device).unsqueeze(-1)          # (B, L, 1)
    h = max(r["fl"] for r in batch)
    pl = eff_pred_len(hist.shape[1], h)
    ds_key = _sanitize(batch[0]["source"])
    pred = forecast_flex(model, hist, ds_key, pl).squeeze(-1)  # (B, pl)
    return pred, hist


def masked_nmse(pred, batch, hist, device):
    """Masked normalized MSE (shared by the loss and the val metric)."""
    B = pred.shape[0]
    hmax = max(r["fl"] for r in batch)
    y = torch.full((B, hmax), float("nan"), device=device)
    for j, r in enumerate(batch):
        y[j, :r["fl"]] = torch.tensor(r["fut"][:r["fl"]], device=device)
    mask = torch.isfinite(y)
    y = torch.nan_to_num(y)
    _, stdev = norm_stats(hist)
    scale = stdev[:, 0, 0].unsqueeze(1)                        # (B,1)
    diff = (pred[:, :hmax] - y) / scale
    return (diff.pow(2) * mask).sum() / mask.sum().clamp(min=1)


def run_train(args, model, device):
    rng = random.Random(args.seed)
    rows = load_rows(args.parquet, "train")
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    if args.limit:
        train_rows, val_rows = train_rows[:args.limit], val_rows[:max(8, args.limit // 8)]
    print(f"[units] train={len(train_rows)} val={len(val_rows)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_ep = max(1, len(make_batches(train_rows, args.bs, False, rng)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_ep, eta_min=args.lr * 0.1)

    best_val, best_state, best_ep = float("inf"), None, -1
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in make_batches(train_rows, args.bs, True, rng):
            pred, hist = batch_forward(model, batch, device)
            loss = masked_nmse(pred, batch, hist, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for batch in make_batches(val_rows, args.bs, False, rng):
                pred, hist = batch_forward(model, batch, device)
                vtot += masked_nmse(pred, batch, hist, device).item() * len(batch)
                vn += len(batch)
        vloss = vtot / max(vn, 1)
        print(f"[units] epoch {ep+1}/{args.epochs} train_nmse={tot/max(nb,1):.4f} val_nmse={vloss:.4f}")
        if vloss < best_val:
            best_val, best_ep = vloss, ep + 1
            best_state = copy.deepcopy(model.state_dict())

    print(f"[units] best epoch={best_ep} val_nmse={best_val:.4f}")
    model.load_state_dict(best_state)
    save = os.path.join(args.output_dir, "units_finetuned.pth")
    torch.save({"state_dict": best_state, "best_epoch": best_ep, "val_nmse": best_val,
                "args": vars(args)}, save)
    print(f"[units] finetuned ckpt -> {save}")


def run_infer(args, model, device):
    rows = load_rows(args.parquet, "test")
    if args.limit:
        rows = rows[:args.limit]
    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    H = int(test["future_len"].max())
    N = len(rows)
    print(f"[units] infer test N={N}, H={H}")

    P = np.full((N, len(QUANTILE_LEVELS), H), np.nan, dtype=np.float32)
    G = np.full((N, H), np.nan, dtype=np.float32)
    R = np.zeros((N, H), dtype=np.float32)
    V = np.zeros((N, H), dtype=bool)
    ids = np.empty(N, dtype=object)
    ds_names = np.empty(N, dtype=object)

    model.eval()
    rng = random.Random(0)
    with torch.no_grad():
        for batch in make_batches(rows, args.bs, False, rng):
            pred, _ = batch_forward(model, batch, device)
            pred = pred.float().cpu().numpy()
            for j, r in enumerate(batch):
                i, fl = r["idx"], r["fl"]
                P[i, :, :fl] = pred[j, :fl][None, :]          # all 21 quantiles take the point forecast (protocol note)
                G[i, :fl] = r["fut"][:fl]
                V[i, :fl] = np.isfinite(r["fut"][:fl])
                rs, re = r["roi"]
                if rs is not None and re is not None and not (pd.isna(rs) or pd.isna(re)):
                    a = max(0, int(rs) - r["past_len"])
                    b = min(fl, int(re) - r["past_len"])
                    if b > a:
                        R[i, a:b] = 1.0
                ids[i] = r["id"]
                ds_names[i] = r["source"]

    out = os.path.join(args.output_dir, "forecast_preds.npz")
    np.savez(out, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
             quantile_levels=QUANTILE_LEVELS, ids=ids, dataset_names=ds_names)
    print(f"[units] npz -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer", "both"], default="both")
    ap.add_argument("--units_repo", required=True)
    ap.add_argument("--ckpt", default=None, help="official pretrain checkpoint (starting point for train)")
    ap.add_argument("--finetuned", default=None, help="fine-tuned checkpoint to load when mode=infer")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)          # same as the official few-shot finetune
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--clip_grad", type=float, default=100.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    df = pd.read_parquet(args.parquet)
    source_names = sorted(df["dataset_name"].unique())
    model = build_model(args.units_repo, args.ckpt if args.mode != "infer" else None,
                        source_names, device)
    print(f"[units] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M device={device}")

    if args.mode in ("train", "both"):
        run_train(args, model, device)
    if args.mode in ("infer", "both"):
        if args.mode == "infer":
            ft = args.finetuned or os.path.join(args.output_dir, "units_finetuned.pth")
            sd = torch.load(ft, map_location="cpu", weights_only=False)
            model.load_state_dict(sd["state_dict"])
            print(f"[units] loaded finetuned ckpt {ft} (best_epoch={sd.get('best_epoch')})")
        run_infer(args, model, device)


if __name__ == "__main__":
    main()
