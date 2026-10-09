"""LLM-free ablation arm for the MMTR understanding pool: chronos2 encoder + multi-task
classification heads (no language model at all).

Purpose: the controlled ablation "does the LLM bring a performance gain" on the understanding
side -- it shares the same chronos2 pretrained weights, the same training data (the sub-tasks of
the pool that have a fixed question + closed label set) and the same native scoring protocol as
the LLM arms; the only difference is whether the LLM or a per-sub-task linear head sits on top of
chronos2.

Structure:
  context (sum C, L) -- chronos2 encode (channels of one sample share a group, per-channel global
                       loc/scale, same convention as ``ChronosLLM._encode_history_to_soft_prompt``)
        -> per-row masked-mean over valid patches -> channel mean -> (B, d_model)
        [with use_stats, concatenate the 6-dim raw-scale ``_series_stats`` (channel mean) -- the
         LLM arms train with ``history_stats_token`` on by default, so the ablation arm receives
         the same feature set]
        -> LayerNorm -> Linear+GELU trunk -> per-sub-task Linear head.

Single chunk only (tl <= context_length=8192): all series of the applicable sub-tasks are
<= 5000 points; the dataset layer left-truncates as a safety net. gate/cross-attn are never used
(under zero init the numerics are strictly equal to the original Chronos-2).
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attn_chronos import load_chronos2_with_cross_attn

_STATS_N_FEATURES = 6


def _series_stats(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Same feature set as ``ChronosLLM._series_stats`` (loc/scale/log1p(scale)/range/log1p(range)/max|x|)."""
    x = x.float()
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return x.new_zeros(_STATS_N_FEATURES)
    rng = (finite.max() - finite.min()).clamp_min(0)
    mx = finite.abs().max()
    loc = loc.reshape(()).float()
    scale = scale.reshape(()).float().clamp_min(0)
    return torch.stack([loc, scale, torch.log1p(scale), rng, torch.log1p(rng), mx])


class ChronosHeadClassifier(nn.Module):
    def __init__(
        self,
        chronos,
        task_labels: dict[str, list[str]],
        use_stats: bool = True,
        trunk_hidden: int = 768,
        encode_max_rows: int = 128,
    ):
        super().__init__()
        self.chronos = chronos
        self.task_labels = {t: list(ls) for t, ls in task_labels.items()}
        self.use_stats = bool(use_stats)
        self.trunk_hidden = int(trunk_hidden)
        self.encode_max_rows = int(encode_max_rows)
        self.patch = int(chronos.chronos_config.input_patch_size)
        self.window = int(chronos.chronos_config.context_length)
        d = int(chronos.config.d_model)
        in_dim = d + (_STATS_N_FEATURES if self.use_stats else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.trunk = nn.Sequential(nn.Linear(in_dim, self.trunk_hidden), nn.GELU())
        self.heads = nn.ModuleDict(
            {t: nn.Linear(self.trunk_hidden, len(ls)) for t, ls in self.task_labels.items()}
        )

    # ------------------------------------------------------------------ encode
    def pooled_features(
        self,
        context: torch.Tensor,        # (sum C, L) left NaN-padded
        true_lengths: torch.Tensor,   # (sum C,)
        n_channels: torch.Tensor | None,  # (B,); None = every row is one single-channel sample
    ) -> torch.Tensor:
        SC, L = context.shape
        n_list = [1] * SC if n_channels is None else [int(x) for x in n_channels]
        B = len(n_list)
        assert sum(n_list) == SC, f"sum of n_channels {sum(n_list)} != number of rows {SC}"

        reals, locs, scales, valid_len, gids = [], [], [], [], []
        stats_rows = []
        r = 0
        for b, C_b in enumerate(n_list):
            tl = max(1, min(int(true_lengths[r].item()), L, self.window))  # all channels of a sample share the length
            for c in range(C_b):
                real = context[r + c, L - tl:]
                _, (lc, sc) = self.chronos.instance_norm(real.unsqueeze(0))
                reals.append(real)
                locs.append(lc)
                scales.append(sc)
                valid_len.append(tl)
                gids.append(b)
                if self.use_stats:
                    stats_rows.append(_series_stats(real, lc, sc))
            r += C_b

        pad_len = min(self.window, max(math.ceil(v / self.patch) * self.patch for v in valid_len))
        chunks = context.new_full((SC, pad_len), float("nan"))
        for i, seg in enumerate(reals):
            seg = seg[-pad_len:]
            chunks[i, pad_len - seg.shape[0]:] = seg.to(chunks.dtype)
        loc = torch.cat([x.reshape(1, 1) for x in locs], dim=0)
        scale = torch.cat([x.reshape(1, 1) for x in scales], dim=0)
        gid_t = torch.tensor(gids, dtype=torch.long, device=context.device)

        # Bucketed encode along sample boundaries (GroupSelfAttention's (T,R,R) grows quadratically with the row count; zero interaction between groups).
        bounds = [i for i in range(SC) if i == 0 or gids[i] != gids[i - 1]]
        buckets, cur = [], 0
        if self.encode_max_rows:
            for gs in bounds[1:]:
                if gs - cur >= self.encode_max_rows:
                    buckets.append((cur, gs))
                    cur = gs
        buckets.append((cur, SC))

        pooled_rows = context.new_zeros((SC, int(self.chronos.config.d_model)), dtype=torch.float32)
        for s, e in buckets:
            enc_out, _, _, ncp = self.chronos.encode(
                context=chunks[s:e], num_output_patches=1,
                group_ids=gid_t[s:e], loc_scale=(loc[s:e], scale[s:e]),
            )
            patches = enc_out[0][:, :ncp]                       # (e-s, ncp, d)
            for i in range(s, e):
                vp = min(ncp, max(1, math.ceil(valid_len[i] / self.patch)))
                pooled_rows[i] = patches[i - s, ncp - vp:].float().mean(dim=0)

        # Channel aggregation -> (B, d) (mean over the channels of each sample)
        out, r = [], 0
        for C_b in n_list:
            out.append(pooled_rows[r:r + C_b].mean(dim=0))
            r += C_b
        feats = torch.stack(out, dim=0)
        if self.use_stats:
            st = torch.stack(stats_rows, dim=0)                 # (sum C, 6)
            st_out, r = [], 0
            for C_b in n_list:
                st_out.append(st[r:r + C_b].mean(dim=0))
                r += C_b
            feats = torch.cat([feats, torch.stack(st_out, dim=0)], dim=1)
        return feats

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        context: torch.Tensor,
        true_lengths: torch.Tensor,
        n_channels: torch.Tensor | None,
        tasks: list[str],
        labels: torch.Tensor | None = None,   # (B,) int64; -100 skips that sample's loss
        candidates: list[list[int] | None] | None = None,
    ):
        """Returns ``(loss, preds)``: loss is the batch mean of per-sample CE (None when labels=None);
        preds (B,) are the label ids of the constrained argmax (restricted to the candidates when
        candidates[i] is not None)."""
        feats = self.pooled_features(context, true_lengths, n_channels)
        h = self.trunk(self.norm(feats.to(self.norm.weight.dtype)))
        B = h.shape[0]
        preds = torch.full((B,), -1, dtype=torch.long, device=h.device)
        loss_sum, n_loss = h.new_zeros(()), 0
        for task in sorted(set(tasks)):
            idx = [i for i, t in enumerate(tasks) if t == task]
            logits = self.heads[task](h[idx])                   # (n, n_cls)
            masked = logits.detach().float().clone()
            if candidates is not None:
                for j, i in enumerate(idx):
                    cand = candidates[i]
                    if cand:
                        keep = torch.zeros(masked.shape[1], dtype=torch.bool, device=masked.device)
                        keep[torch.as_tensor(cand, device=masked.device)] = True
                        masked[j, ~keep] = float("-inf")
            preds[idx] = masked.argmax(dim=1)
            if labels is not None:
                lb = labels[torch.as_tensor(idx, device=labels.device)]
                keep = lb != -100
                if keep.any():
                    loss_sum = loss_sum + F.cross_entropy(
                        logits[keep], lb[keep], reduction="sum"
                    )
                    n_loss += int(keep.sum())
        loss = loss_sum / max(1, n_loss) if labels is not None else None
        return loss, preds

    # ------------------------------------------------------------- save / load
    def save_pretrained(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "headcls_config.json"), "w") as f:
            json.dump({
                "task_labels": self.task_labels,
                "use_stats": self.use_stats,
                "trunk_hidden": self.trunk_hidden,
                "encode_max_rows": self.encode_max_rows,
            }, f, ensure_ascii=False, indent=1)

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, chronos_path: str):
        with open(os.path.join(ckpt_dir, "headcls_config.json")) as f:
            cfg = json.load(f)
        chronos = load_chronos2_with_cross_attn(chronos_path)
        model = cls(chronos, cfg["task_labels"], use_stats=cfg["use_stats"],
                    trunk_hidden=cfg["trunk_hidden"], encode_max_rows=cfg["encode_max_rows"])
        state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
