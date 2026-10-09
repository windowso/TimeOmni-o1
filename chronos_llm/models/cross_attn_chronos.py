"""Inject gated cross-attention into Chronos-2 so that it can consult information provided by
the LLM while forecasting.

Design notes:
- ``GatedCrossAttention``: a self-contained cross-attn (built on ``nn.MultiheadAttention``) whose
  query comes from the chronos2 hidden states and whose key/value come from the compressed LLM
  representation (cross_states). The upstream ``TimeCrossAttention`` is deliberately not reused:
  its ``MHA.shape()`` hard-codes the query ``seq_length`` onto the KV tensors, so the reshape fails
  whenever the KV length M differs from the query length.
- The gate is a scalar, zero-initialised by default and wrapped in ``tanh``: with the initial
  value 0, the layer is the identity whenever cross_states=None or gate=0, and the whole model is
  numerically identical to the original Chronos2Model (training starts from the original
  forecaster).
- ``Chronos2WithCrossAttn`` overrides ``encode``/``forward`` to pass ``cross_states`` through to
  every block; all other logic (patching / instance-norm / REG token / loss / de-normalisation)
  is inherited unchanged.

WARNING (loading): see ``load_chronos2_with_cross_attn`` -- under transformers 5.x,
from_pretrained is broken for the custom chronos-2 model (weights are not loaded, non-persistent
buffers are not materialised); the model must be constructed directly and the weights loaded with
a manual load_state_dict.
"""

import copy
import json
import os
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange, repeat

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import (
    Chronos2Encoder,
    Chronos2EncoderBlock,
    Chronos2EncoderBlockOutput,
    Chronos2EncoderOutput,
    Chronos2Model,
    Chronos2Output,
)


class GatedCrossAttention(nn.Module):
    """Gated cross-attention: chronos2 hidden states as query, compressed LLM representation as key/value."""

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        d_model = config.d_model
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        # Scalar gate, zero-initialised -> tanh(gate)=0 -> identity at the starting point.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: torch.Tensor,
        cross_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            hidden_states: (B, T, d_model) chronos2 hidden states (query).
            cross_states: (B, M, d_model) compressed LLM representation (key/value).
            cross_key_padding_mask: (B, M) bool, True=padding (ignored).
        """
        q = self.q_norm(hidden_states)
        kv = self.kv_norm(cross_states)
        attn_out, _ = self.attn(
            q, kv, kv, key_padding_mask=cross_key_padding_mask, need_weights=False
        )
        return hidden_states + torch.tanh(self.gate) * attn_out


class CrossAttnEncoderBlock(Chronos2EncoderBlock):
    """Insert gated cross-attn after TimeSelfAttn -> GroupSelfAttn and before the FeedForward."""

    def __init__(self, config):
        super().__init__(config)
        self.cross_attn = GatedCrossAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
        cross_states: torch.Tensor | None = None,
        cross_key_padding_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Chronos2EncoderBlockOutput:
        # time self-attention
        time_self_attn_outputs = self.layer[0](
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = time_self_attn_outputs[0]

        # group self-attention (along the batch axis)
        group_self_attn_outputs = self.layer[1](
            hidden_states, attention_mask=group_time_mask, output_attentions=output_attentions
        )
        hidden_states = group_self_attn_outputs[0]

        # gated cross-attention (only when cross_states is provided)
        if cross_states is not None:
            hidden_states = self.cross_attn(
                hidden_states, cross_states, cross_key_padding_mask=cross_key_padding_mask
            )

        # feed forward
        hidden_states = self.layer[2](hidden_states)

        return Chronos2EncoderBlockOutput(
            hidden_states=hidden_states,
            time_self_attn_weights=time_self_attn_outputs.attn_weights,
            group_self_attn_weights=group_self_attn_outputs.attn_weights,
        )


class CrossAttnEncoder(Chronos2Encoder):
    """Same as Chronos2Encoder, but the blocks are CrossAttnEncoderBlock and cross_states is passed through."""

    def __init__(self, config):
        super().__init__(config)
        self.block = nn.ModuleList(
            [CrossAttnEncoderBlock(config) for _ in range(config.num_layers)]
        )
        # Architecture ablation B3: allow injecting the LLM feedback into only a subset of encoder
        # blocks (None = all layers = default). Not nn.Module state and not part of the state_dict;
        # set via Chronos2WithCrossAttn.set_cross_attn_layers.
        self.cross_attn_layer_ids: set[int] | None = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cross_states: torch.Tensor | None = None,
        cross_states_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        stop_at_layer: int | None = None,
    ) -> Chronos2EncoderOutput:
        batch_size, seq_length = inputs_embeds.size()[:-1]

        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        extended_attention_mask = self._expand_and_invert_time_attention_mask(
            attention_mask, inputs_embeds.dtype
        )
        group_time_mask = self._construct_and_invert_group_time_mask(
            group_ids, attention_mask, inputs_embeds.dtype
        )
        # (B, M) valid-mask(1=valid) -> key_padding_mask(True=pad)
        cross_key_padding_mask = None
        if cross_states is not None and cross_states_mask is not None:
            cross_key_padding_mask = cross_states_mask <= 0

        all_time_self_attentions: tuple[torch.Tensor, ...] = ()
        all_group_self_attentions: tuple[torch.Tensor, ...] = ()

        hidden_states = self.dropout(inputs_embeds)

        # stop_at_layer (architecture ablation A4): run only the first k blocks and return -- when
        # the soft prompt is taken from an intermediate layer this also saves the compute of the
        # remaining (N-k) layers. k>=N or None => full path (including the final LN/dropout),
        # numerically identical to the default; k<N **skips final_layer_norm** (it was designed for
        # the last layer's output; the downstream Q-former has its own input_ln).
        n_layers = len(self.block)
        n_stop = n_layers if stop_at_layer is None else max(1, min(int(stop_at_layer), n_layers))
        allow = getattr(self, "cross_attn_layer_ids", None)
        for li, layer_module in enumerate(self.block):
            if li >= n_stop:
                break
            layer_outputs = layer_module(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attention_mask,
                group_time_mask=group_time_mask,
                cross_states=(cross_states if (allow is None or li in allow) else None),
                cross_key_padding_mask=cross_key_padding_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_time_self_attentions = (*all_time_self_attentions, layer_outputs.time_self_attn_weights)
                all_group_self_attentions = (*all_group_self_attentions, layer_outputs.group_self_attn_weights)

        if n_stop >= n_layers:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

        return Chronos2EncoderOutput(
            last_hidden_state=hidden_states,
            all_time_self_attn_weights=all_time_self_attentions,
            all_group_self_attn_weights=all_group_self_attentions,
        )


class Chronos2WithCrossAttn(Chronos2Model):
    """Chronos2Model + gated cross-attention in every layer.

    ``encode``/``forward`` gain the optional ``cross_states`` (B, M, d_model) and
    ``cross_states_mask`` (B, M, 1=valid); when None the model is numerically identical to the
    original.
    """

    def __init__(self, config):
        super().__init__(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = CrossAttnEncoder(encoder_config)
        self.post_init()

    def set_cross_attn_layers(self, spec: str | None) -> list[int]:
        """Architecture ablation B3: choose which encoder blocks receive the LLM feedback
        (default all = the standard configuration).

        ``spec`` accepts ``"all"`` (or None / empty string), ``"lastK"``, ``"firstK"``, or a
        comma-separated list of 0-based layer indices (e.g. ``"0,3,6,9"``). Returns the list of
        layer indices that are actually active (ascending), for logging / verification.

        Note: **the cross_attn sub-module of unselected layers still exists** (weights are saved /
        loaded as usual); it simply never receives ``cross_states`` in the forward pass, so it is an
        identity pass-through with no gradient. This keeps the checkpoint structure independent of
        the spec.
        """
        n = len(self.encoder.block)
        if spec is None or not str(spec).strip() or str(spec).strip().lower() == "all":
            self.encoder.cross_attn_layer_ids = None
            return list(range(n))
        t = str(spec).strip().lower()
        if t.startswith("last") and t[4:].isdigit():
            ids = list(range(max(0, n - int(t[4:])), n))
        elif t.startswith("first") and t[5:].isdigit():
            ids = list(range(0, min(n, int(t[5:]))))
        else:
            ids = sorted({int(x) for x in t.split(",") if x.strip() != ""})
            bad = [i for i in ids if not 0 <= i < n]
            if bad:
                raise ValueError(f"cross_attn_layers out of range ({n} layers in total): {bad}")
        if not ids:
            raise ValueError(f"cross_attn_layers={spec!r} parsed to an empty set")
        self.encoder.cross_attn_layer_ids = None if len(ids) == n else set(ids)
        return ids

    def _prepare_patched_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Identical to the base ``Chronos2Model._prepare_patched_context`` but accepts an external
        ``loc_scale``.

        When ``loc_scale`` is given (e.g. global statistics of the whole history), ``instance_norm``
        normalises with it directly instead of computing per-window statistics -- so the several
        chunks of one history share a single normalisation scale (global normalisation).
        ``loc_scale=None`` falls back to the original per-instance (per-window) normalisation and is
        numerically identical to the base class.
        """
        context_mask = (
            context_mask.to(context.dtype)
            if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, context_length = context.shape
        if context_length > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            context_mask = context_mask[..., -self.chronos_config.context_length :]

        context, loc_scale = self.instance_norm(context, loc_scale)  # key point: pass the external loc_scale through

        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        attention_mask = patched_mask.sum(dim=-1) > 0
        num_context_patches = attention_mask.shape[-1]

        final_context_length = num_context_patches * self.chronos_config.input_patch_size
        context_time_enc = torch.arange(
            start=-final_context_length, end=0, device=self.device, dtype=torch.float32
        )
        context_time_enc = (
            repeat(
                context_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_context_patches,
                p=self.chronos_config.input_patch_size,
            )
            .div(cast(int, self.chronos_config.time_encoding_scale))
            .to(self.dtype)
        )

        patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)
        return patched_context, attention_mask, loc_scale

    def encode(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        cross_states: torch.Tensor | None = None,
        cross_states_mask: torch.Tensor | None = None,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        stop_at_layer: int | None = None,
    ):
        self._validate_input(
            context=context,
            context_mask=context_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
        )

        batch_size = context.shape[0]
        # When loc_scale is not None, normalise with the external (global) statistics and skip the
        # per-window computation (see _prepare_patched_context).
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(
            context=context, context_mask=context_mask, loc_scale=loc_scale
        )
        num_context_patches = attention_mask.shape[-1]

        input_embeds = self.input_patch_embedding(patched_context)
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1), self.config.reg_token_id, device=input_embeds.device
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(self.dtype), torch.ones_like(reg_input_ids).to(self.dtype)], dim=-1
            )

        patched_future, patched_future_covariates_mask = self._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        future_attention_mask = torch.ones(
            batch_size, num_output_patches, dtype=self.dtype, device=self.device
        )
        future_embeds = self.input_patch_embedding(patched_future)
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            cross_states=cross_states,
            cross_states_mask=cross_states_mask,
            output_attentions=output_attentions,
            stop_at_layer=stop_at_layer,
        )
        return encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        cross_states: torch.Tensor | None = None,
        cross_states_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Chronos2Output:
        batch_size = context.shape[0]
        encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            cross_states=cross_states,
            cross_states_mask=cross_states_mask,
            output_attentions=output_attentions,
        )
        hidden_states = encoder_outputs[0]
        assert hidden_states.shape == (
            batch_size,
            num_context_patches + 1 + num_output_patches,
            self.model_dim,
        )

        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds = self.output_patch_embedding(forecast_embeds)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )

        loss = (
            self._compute_loss(
                quantile_preds=quantile_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                patched_future_covariates_mask=patched_future_covariates_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )
            if future_target is not None
            else None
        )

        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )

        return Chronos2Output(
            loss=loss,
            quantile_preds=quantile_preds,
            enc_time_self_attn_weights=encoder_outputs.all_time_self_attn_weights,
            enc_group_self_attn_weights=encoder_outputs.all_group_self_attn_weights,
        )

    def _masked_pinball(
        self,
        quantile_preds: torch.Tensor,
        future_target: torch.Tensor,
        mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Pinball loss in normalised space restricted to a mask (per-row average over masked positions).

        Same normalised space and same aggregation structure as the upstream ``_compute_loss``
        (mean_h -> sum_q -> mean_b); the difference is that the denominator is **the number of
        valid positions inside the mask of each row** (upstream divides by the fixed padded H, which
        dilutes short samples when variable-length futures are batched together). The mask is
        automatically AND-ed with ``isfinite(future_target)`` (NaN missing/padding positions do not
        participate); the batch mean only runs over rows whose mask is non-empty (empty ROI rows do
        not dilute the magnitude). The main forecast loss and the ROI-weighted term share this
        function, so their magnitudes are naturally consistent.

        Args:
            quantile_preds: (B, Q, H) quantile predictions in normalised space (before inverse).
            future_target: (B, fl) future ground truth in the original scale (NaN=missing/padding).
            mask: (B, fl) 1=position counted in the loss, 0=otherwise.
            loc_scale: the instance-norm (loc, scale).
        """
        H = quantile_preds.shape[-1]
        # AND the mask with "target is finite": otherwise NaN missing points inside the ROI would
        # enter the pinball with a fabricated 0 target (the main loss correctly masks the same
        # positions, so the two gradients would conflict); NaN -> 0 only blocks NaN*0=NaN contamination.
        mask = mask.to(quantile_preds.device).float() * torch.isfinite(future_target).float().to(
            quantile_preds.device
        )
        target, _ = self.instance_norm(future_target, loc_scale)
        target = torch.nan_to_num(target.to(quantile_preds.device), nan=0.0)
        # pad to H
        if H > target.shape[-1]:
            pad = (*target.shape[:-1], H - target.shape[-1])
            target = torch.cat([target, torch.zeros(pad).to(target)], dim=-1)
            mask = torch.cat([mask, torch.zeros(pad).to(mask)], dim=-1)
        target = target.unsqueeze(1)  # (B, 1, H)
        quantiles = self.quantiles.to(quantile_preds.device).view(1, -1, 1)
        pinball = 2 * torch.abs(
            (target - quantile_preds) * ((target <= quantile_preds).float() - quantiles)
        )  # (B, Q, H)
        m = mask.unsqueeze(1)  # (B, 1, H)
        denom = m.sum(dim=-1).clamp(min=1.0)  # (B, 1)
        loss = (pinball * m).sum(dim=-1) / denom  # (B, Q)
        # The batch mean only runs over rows with a non-empty mask: the dataset's clamp turns
        # out-of-range ROIs into all-zero rows, and counting them as zero loss in the denominator
        # would dilute the loss magnitude in proportion to the share of bad ROIs (the eval side
        # returns nan for an empty mask and drops it; training follows the same convention).
        # Returns 0 when everything is empty (stays differentiable, never produces NaN).
        per_row = loss.sum(dim=-1)  # (B,)
        row_has = (m.sum(dim=(-2, -1)) > 0).float()  # (B,)
        return (per_row * row_has).sum() / row_has.sum().clamp(min=1.0)

    def forecast_losses(
        self,
        context: torch.Tensor,
        future_target: torch.Tensor,
        num_output_patches: int,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        cross_states: torch.Tensor | None = None,
        cross_states_mask: torch.Tensor | None = None,
        roi_mask: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        target_idx: torch.Tensor | None = None,
    ) -> dict:
        """Forecast + losses: returns ``pred_loss``, optional ``roi_loss`` and the de-normalised
        ``quantile_preds``.

        Multivariate: ``context`` is (sum_C, L) (all channels of all samples), ``group_ids`` lets
        chronos exchange information across channels within a group, ``future_covariates`` is
        (sum_C, fl) (values on known-future rows, NaN elsewhere). ``target_idx`` (sum_C,) bool marks
        the target rows -- **the loss is computed on target rows only** (qp/loc_scale/pfm are sliced
        to the target rows, aligned with ``future_target`` (sum_n_targets, fl)). For a single channel
        ``target_idx=None`` -> no slicing, equivalent to the old logic. Note that ``future_target``
        is not passed to encode (in the multichannel case its row count sum_n_targets differs from the
        context row count sum_C).
        """
        encoder_outputs, loc_scale, _pfm, _ncp = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            cross_states=cross_states,
            cross_states_mask=cross_states_mask,
        )
        hidden_states = encoder_outputs[0]
        forecast_embeds = hidden_states[:, -num_output_patches:]
        qp = self.output_patch_embedding(forecast_embeds)
        qp = rearrange(
            qp,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )  # normalised space (sum_C, Q, H)

        loc, scale = loc_scale
        if target_idx is not None:
            # Keep target rows only (aligned with the (sum_n_targets, ...) of future_target/roi_mask).
            qp = qp[target_idx]
            loc = loc[target_idx]
            scale = scale[target_idx]
        loc_scale_t = (loc, scale)

        # The upstream _compute_loss is not reused: its mean(dim=-1) divides by the fixed padded H --
        # the collator right-pads variable-length futures to the batch maximum, so the loss of
        # short-horizon samples is silently diluted by fl_i/H. _masked_pinball
        # averages per row over the valid positions, with the same aggregation structure as
        # upstream; for equal lengths without missing values the two differ only by the constant
        # factor (fl/H). The extra known-future covariate exclusion mask (pfm) that upstream
        # multiplies in is identically 0 on target rows (row-order convention [targets, cov...];
        # target rows have NaN future_covariates => patched mask=0, inverted mask=1), so it is not needed.
        valid = torch.isfinite(future_target).float()
        pred_loss = self._masked_pinball(qp, future_target, valid, loc_scale_t)
        roi_loss = (
            self._masked_pinball(qp, future_target, roi_mask, loc_scale_t)
            if roi_mask is not None
            else None
        )

        # De-normalise to obtain predictions in the original scale (monitoring / inference only)
        H = num_output_patches * self.chronos_config.output_patch_size
        qp_un = self.instance_norm.inverse(rearrange(qp, "b q h -> b (q h)"), loc_scale_t)
        qp_un = rearrange(qp_un, "b (q h) -> b q h", q=self.num_quantiles, h=H)
        return {"pred_loss": pred_loss, "roi_loss": roi_loss, "quantile_preds": qp_un}


def _load_safetensors_state_dict(ckpt_path: str) -> dict:
    """Read the weight state dict of a chronos-2 checkpoint (single file or sharded)."""
    import safetensors.torch as st

    single = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(single):
        return st.load_file(single)
    index = os.path.join(ckpt_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        state: dict = {}
        for shard in sorted(set(weight_map.values())):
            state.update(st.load_file(os.path.join(ckpt_path, shard)))
        return state
    raise FileNotFoundError(f"No safetensors weights found under {ckpt_path}")


def load_chronos2_with_cross_attn(
    ckpt_path: str,
    dtype: torch.dtype | None = None,
    random_init: bool = False,
) -> "Chronos2WithCrossAttn":
    """Load Chronos2WithCrossAttn and correctly install the pretrained weights.

    IMPORTANT: under transformers 5.x, ``PreTrainedModel.from_pretrained`` is broken for the custom
    chronos-2 modelling code -- it neither actually loads the safetensors weights (parameters stay
    randomly initialised) nor materialises ``persistent=False`` buffers (RoPE's ``inv_freq`` becomes
    uninitialised garbage), so the forward pass immediately produces NaN. This function therefore
    **bypasses from_pretrained**: the model is constructed normally on CPU from the config
    (__init__ computes the inv_freq / quantiles buffers correctly), then the weights are installed
    with a manual ``load_state_dict(strict=False)``; the newly added cross-attn / gate parameters
    remain missing keys and keep their zero / default initialisation.

    ``random_init=True`` (ablation: does TSFM pretraining matter?): only the checkpoint's
    config.json is used to define the architecture and the weight load_state_dict is skipped --
    ``Chronos2WithCrossAttn.__init__`` already initialises every parameter and the RoPE / quantile
    buffers correctly, so skipping the load yields a chronos2 of the same architecture with fully
    random initialisation.
    """
    config = Chronos2CoreConfig.from_pretrained(ckpt_path)
    model = Chronos2WithCrossAttn(config)
    if not random_init:
        raw = _load_safetensors_state_dict(ckpt_path)
        missing, unexpected = model.load_state_dict(raw, strict=False)
        leftover = [k for k in missing if ".cross_attn." not in k]
        if leftover:
            raise RuntimeError(f"Unexpected missing chronos-2 weights after manual load: {leftover[:8]}")
        if unexpected:
            raise RuntimeError(f"Unexpected extra keys in chronos-2 ckpt: {unexpected[:8]}")
    if dtype is not None:
        model = model.to(dtype)
    return model.eval()
