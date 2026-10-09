"""Collator: turns a homogeneous batch (all understanding or all forecast) into the dict the model's
forward expects.

- input_ids/labels: right padding (the supervised answer is at the tail).
- context: history truncated per **branch** (understanding 240k, forecast 8192 = chronos2
  context_length), truncated on the **left** (``h[-cap:]`` drops the oldest, keeps the most recent
  tail), then **left-padded with NaN** (so the most recent values sit at the end of the sequence and
  the padding at the front; chronos' InstanceNorm uses nanmean and Patch uses NaN, both robust to
  NaN padding; the time encoding is anchored at the most recent step ``[-C,...,-1]``). true_lengths
  records each row's valid length. Over-long understanding histories are encoded chunk by chunk on
  the model side by chronos2 with non-overlapping windows of context_length (see
  chronos_llm_model._encode_history_to_soft_prompt); the forecast branch is <= 8192, i.e. a single
  chronos2 window predicts directly.
- forecast branch additionally: future (right-padded NaN) and roi_mask (right-padded 0).
"""

from dataclasses import dataclass

import torch


@dataclass
class ChronosLLMCollator:
    tokenizer: object
    # Branch-specific history caps (both left-truncated, keeping the most recent tail):
    #   understanding up to 240k (longer is encoded by chronos2 sliding windows chunk by chunk);
    #   forecast up to 8192 (= chronos2 context_length).
    understanding_max_context: int = 240_000
    forecast_max_context: int = 8192

    def _cap_for(self, branch: str) -> int:
        return self.forecast_max_context if branch == "forecast" else self.understanding_max_context

    def _pad_text(self, seqs, pad_value):
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    @staticmethod
    def _to_2d(x):
        """1D (T,) -> (1, T); 2D (C, T) unchanged."""
        return x.unsqueeze(0) if x.dim() == 1 else x

    def _fold_context(self, histories, cap: int):
        """Fold all channels of all samples in the batch into (sum C, L) (time left-truncated to cap,
        left NaN-padded).

        Returns (context, group_ids, true_lengths, n_channels):
        - context (sum C, L): row order = channels of sample 0, channels of sample 1, ...;
          right-aligned (most recent values at the tail).
        - group_ids (sum C,): the sample index each row belongs to (used by chronos GroupSelfAttention
          for within-group cross-channel interaction).
        - true_lengths (sum C,): valid length of each row.
        - n_channels (B,): channels per sample (so the model can split rows back per sample).
        For single-channel input (1D history treated as (1,T)) sum C=B and group_ids=[0..B-1],
        equivalent to the old logic.
        """
        rows, group_ids, lens, n_ch = [], [], [], []
        for gi, h in enumerate(histories):
            h = self._to_2d(h)[:, -cap:]
            n_ch.append(h.shape[0])
            for c in range(h.shape[0]):
                rows.append(h[c])
                group_ids.append(gi)
                lens.append(h.shape[1])
        L = max(r.shape[0] for r in rows)
        ctx = torch.full((len(rows), L), float("nan"))
        for i, r in enumerate(rows):
            ctx[i, L - r.shape[0] :] = r.to(torch.float32)
        return (
            ctx,
            torch.tensor(group_ids, dtype=torch.long),
            torch.tensor(lens, dtype=torch.long),
            torch.tensor(n_ch, dtype=torch.long),
        )

    @staticmethod
    def _right_pad_series(series, pad_value):
        L = max(s.shape[0] for s in series)
        out = torch.full((len(series), L), pad_value, dtype=torch.float32)
        for i, s in enumerate(series):
            out[i, : s.shape[0]] = s.to(torch.float32)
        return out

    def _fold_rows(self, items_2d, pad_value):
        """items_2d: list[(n_i, fl_i)] (n_i>=0) -> ((sum n, FL) right-padded, counts(B,))."""
        counts = [it.shape[0] for it in items_2d]
        rows = [it[i] for it in items_2d for i in range(it.shape[0])]
        return self._right_pad_series(rows, pad_value), torch.tensor(counts, dtype=torch.long)

    def __call__(self, batch):
        branch = batch[0]["branch"]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0  # attention_mask covers it; the pad value itself takes no part in computation

        input_ids = self._pad_text([b["input_ids"] for b in batch], pad_id)
        # Built from the true lengths (not inferred via != pad_id): does not rely on the assumption
        # that the pad token never appears in the text, and guarantees the valid positions form a
        # contiguous prefix (the left-pad path of _splice_soft_prompt strips the right pad by attn.sum()).
        lengths = torch.tensor([len(b["input_ids"]) for b in batch], dtype=torch.long)
        attention_mask = (
            torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]
        ).long()
        labels = self._pad_text([b["labels"] for b in batch], -100)

        context, group_ids, true_lengths, n_channels = self._fold_context(
            [b["history"] for b in batch], self._cap_for(branch)
        )

        out = {
            "branch": branch,
            "context": context,
            "group_ids": group_ids,
            "true_lengths": true_lengths,
            "n_channels": n_channels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # Window-sampling metadata (optional, understanding branch): per sample (n_chunks, 2) or None
        # (not triggered), so the model can expand true per-patch positions (HF Trainer's
        # _prepare_input handles list elements recursively).
        if branch == "understanding" and any(b.get("chunk_pos") is not None for b in batch):
            out["chunk_pos"] = [b.get("chunk_pos") for b in batch]
        # Inference: pass through per-sample meta (id/gt/task etc.) so eval can align and compute
        # metrics (training samples carry no meta, so nothing is emitted).
        if "meta" in batch[0]:
            out["meta"] = [b["meta"] for b in batch]
        if branch == "forecast":
            # In-training scheduled sampling: inference prefix (right-padded; on SS batches the model
            # strips the pad with side="left" before generating).
            if batch[0].get("ss_prefix_ids") is not None:
                out["ss_prefix_ids"] = self._pad_text([b["ss_prefix_ids"] for b in batch], pad_id)
                sl = torch.tensor([len(b["ss_prefix_ids"]) for b in batch], dtype=torch.long)
                out["ss_prefix_mask"] = (
                    torch.arange(out["ss_prefix_ids"].shape[1])[None, :] < sl[:, None]
                ).long()
            # Target future ground truth: per sample (n_targets, fl) folded into (sum n_targets, FL).
            futures = [self._to_2d(b["future"]) for b in batch]
            out["future"], out["n_targets"] = self._fold_rows(futures, float("nan"))
            # ROI: per sample (fl,), broadcast to each target row of that sample.
            roi_rows = []
            for b, f in zip(batch, futures):
                roi_rows.extend([b["roi"]] * f.shape[0])
            out["roi_mask"] = self._right_pad_series(roi_rows, 0.0)
            # Known-future covariates (optional): emitted only when some n_fut>0 exists.
            fcs = [b.get("future_covariates") for b in batch]
            if any(fc is not None and self._to_2d(fc).shape[0] > 0 for fc in fcs):
                fc2d = [self._to_2d(fc) if fc is not None else context.new_zeros(0, 0) for fc in fcs]
                out["future_covariates"], out["n_future_covariates"] = self._fold_rows(fc2d, float("nan"))
        return out
