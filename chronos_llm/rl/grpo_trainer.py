"""Shared GRPO utilities for the forecasting branch (rewards + one policy-update step); only the
**LLM LoRA** is updated.

Core design:
- rollout (model.rollout_forecast_rl): sample G groups of conclusions -> logps(grad) + preds(no_grad) + texts.
- reward[g,b] = -CRPS(sample_wql) + w_roi*ROI_IoU + w_mag*magnitude-band match (the latter two are dense shaping).
- within-group advantage = (r - mean_G)/(std_G+eps); policy loss = -(adv*logps).mean().
- chronos/qformer frozen (gradients reach only the LLM).
"""
import numpy as np
import torch

from chronos_llm.eval.metrics import sample_wql
from chronos_llm.rl.reward import extract_roi, text_reward


def compute_rewards(preds, texts, batch, levels, w_roi=0.3, w_mag=0.3):
    """preds: list[G] of (B,Q,H) torch; texts: list[G] of list[str]. Returns the (G,B) numpy reward + details.

    Main term -CRPS (sample_wql, per-sample normalised WQL, same convention as training/eval);
    auxiliary terms ROI IoU + magnitude-band match (the true ROI/magnitude are parsed from the
    gt_conclusion in batch['meta'], which requires the dataset's emit_meta=True)."""
    future = batch["future"].detach().cpu().numpy()             # (B, H) NaN-padded
    metas = batch.get("meta") or [{}] * future.shape[0]
    G, B = len(preds), future.shape[0]
    R = np.zeros((G, B), dtype=np.float64)
    crps_mat = np.zeros((G, B), dtype=np.float64)
    for g in range(G):
        p = preds[g].detach().cpu().float().numpy()            # (B, Q, H_pred)
        # chronos aligns to patches (16) => H_pred may exceed the future length (e.g. 78->80); crop to the common length.
        Hc = min(p.shape[-1], future.shape[-1])
        for b in range(B):
            fb, pb = future[b, :Hc], p[b, :, :Hc]
            valid = ~np.isnan(fb)
            crps = sample_wql(fb, pb, levels, valid)
            crps = 0.0 if (crps is None or np.isnan(crps)) else float(crps)
            gt_c = str(metas[b].get("gt_conclusion", ""))
            tr, _ = text_reward(texts[g][b], extract_roi(gt_c), gt_c, w_roi=w_roi, w_mag=w_mag)
            R[g, b] = -crps + tr
            crps_mat[g, b] = crps
    return R, crps_mat


def grpo_step(model, batch, optimizer, *, horizon, group_size=6, temperature=1.0,
              w_roi=0.3, w_mag=0.3, grad_clip=1.0, max_new_tokens=256):
    """One GRPO update. Returns a metrics dict. The model must already have its non-LLM-LoRA
    parameters frozen and be set to train() by the caller."""
    levels = model.chronos.quantiles.detach().cpu().float().numpy()
    # The rollout is entirely no_grad (saves memory): obtain the prefix + each group's gen_ids/preds/texts.
    prefix_e, prefix_a, gen_ids, preds, texts = model.rollout_forecast_rl(
        batch, horizon=horizon, group_size=group_size, temperature=temperature,
        max_new_tokens=max_new_tokens)
    R, crps = compute_rewards(preds, texts, batch, levels, w_roi, w_mag)  # (G,B)
    dev = prefix_e.device
    Rt = torch.as_tensor(R, dtype=torch.float32, device=dev)
    # Within-group (across G, per sample) normalised advantage -- the core of GRPO, no critic needed.
    adv = (Rt - Rt.mean(0, keepdim=True)) / (Rt.std(0, keepdim=True) + 1e-6)  # (G,B)

    optimizer.zero_grad()
    total_loss = 0.0
    # **Per-group** grad forward + backward (gradient accumulation): only one group's graph is alive at any time, saving G x memory.
    for g in range(group_size):
        logp_g = model.rl_logp(prefix_e, prefix_a, gen_ids[g])            # (B,) grad
        loss_g = -(adv[g].detach() * logp_g).mean() / group_size
        loss_g.backward()
        total_loss += float(loss_g.detach())
    trainable = [p for p in model.parameters() if p.requires_grad]
    gn = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
    optimizer.step()
    return {
        "loss": total_loss,
        "reward": float(R.mean()),
        "crps": float(crps.mean()),
        "reward_std_in_group": float(np.mean(R.std(0))),   # zero within-group variance => advantage all 0, no learning signal
        "grad_norm": float(gn),
    }
