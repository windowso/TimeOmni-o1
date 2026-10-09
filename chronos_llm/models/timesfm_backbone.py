"""Wrap TimesFM-3.0 in the interface shape of chronos-2, so that the ``ChronosLLM`` trunk can switch
backbones with almost no change at all.

This is the "swap the TSFM" arm of the backbone ablation. An adapter was chosen over adding
``if isinstance(...)`` branches inside the trunk so that the change can be removed wholesale and
introduces no risk whatsoever to the default Chronos-2 path.

The trunk's interface surface onto the TSFM (29 call sites) reduces to the following items, each of
which this class provides one by one: ``.config.d_model`` /
``.chronos_config.{input_patch_size,output_patch_size,context_length}`` / ``.encoder.block`` (layer
count) / ``.dtype`` / ``.quantiles`` / ``.num_quantiles`` / ``.instance_norm`` /
``.set_cross_attn_layers`` / ``.encode`` / ``.forecast_losses`` / ``__call__``.

## Three protocol decisions (they decide whether the two arms are comparable -- read before changing)

1. **context_length is pinned to 8192** (the same as chronos-2, even though TimesFM itself can take
   ~15k). It determines how the sliding-window Q-former chunks the history; if the two arms disagreed
   here, the measured difference would be contaminated by "different window partitioning".

2. **Global normalisation is kept, but only its linear half.** The design is "one (loc, scale) per
   series/channel taken over the whole history"; chronos receives it through ``encode(loc_scale=...)``
   and forwards it to its instance_norm. TimesFM's CPM-RevIN is causal running-stats plus linear
   detrending and cannot take an external loc_scale -- **but it does not need to be changed**:
   applying ``(x - loc) / scale`` to the context before feeding it in (purely linear, and TimesFM's
   RevIN is approximately invariant under linear transforms) already preserves the cross-window scale
   differences in the values, so the design intent is met without touching a single line of the
   backbone.
   This half deliberately **omits arcsinh**: arcsinh is a non-linear distortion and would damage
   TimesFM's pretrained capability.

3. **The loss space is strictly isomorphic to the other arm.** The pinball loss borrows the function
   object ``Chronos2WithCrossAttn._masked_pinball`` directly (not a copy -- a drifting loss
   implementation is the least acceptable risk of all); it uses ``self.instance_norm`` and
   ``self.quantiles``, both of which this class supplies with chronos semantics: ``instance_norm``
   carries ``use_arcsinh=True``. Both arms therefore compute the pinball loss in
   arcsinh((x - loc) / scale) space with exactly the same weighting.
   (arcsinh is monotone => quantiles stay quantiles under it, so computing the pinball loss in that
   space is legitimate.)

## Known asymmetries (they must be stated alongside any result)

- input_patch 32 vs chronos's 16: an intrinsic property of the backbone that cannot be changed; a
  history of the same length yields half as many patches on this arm, and the soft-token budget
  (``qformer.plan`` derives it from P x C) changes accordingly.
- 9 quantiles vs 21: TimesFM's 9 are exactly a subset of chronos's 21, so cross-arm CRPS comparisons
  are always recomputed on the common 9-quantile subset.
- Covariates: the forecasting corpus is **univariate with no covariates**, so the ``future_covariates`` path is not implemented and raises if it is passed --
  silently ignoring it would leave "the model cannot see the known future" unprovable either way.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from chronos.chronos_bolt import InstanceNorm

from chronos_llm.models.cross_attn_chronos import Chronos2WithCrossAttn
from chronos_llm.models.cross_attn_timesfm import (
    TimesFM3WithCrossAttn,
    load_timesfm3_with_cross_attn,
)

# Sliding-window chunk size, aligned with chronos-2 (see decision 1 in the module docstring).
CHRONOS_CONTEXT_LENGTH = 8192


def _group_to_variates(gids: torch.Tensor):
    """Gather the folded rows by group: returns (the per-group row counts, and the row -> (group id,
    position within the group) index pair).

    Chronos's group granularity is "all channels of the same sample and the same chunk" => one group
    is exactly one set of variates on TimesFM's variate axis. The original row order (= channel order)
    is preserved within a group, which the stable sort guarantees.
    """
    uniq, inv = torch.unique(gids, return_inverse=True)
    n_groups = int(uniq.numel())
    counts = torch.bincount(inv, minlength=n_groups)
    order = torch.argsort(inv, stable=True)
    inv_sorted = inv[order]
    offs = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    pos_sorted = torch.arange(gids.numel(), device=gids.device) - offs[inv_sorted]
    g_idx = torch.empty_like(inv)
    v_idx = torch.empty_like(inv)
    g_idx[order] = inv_sorted
    v_idx[order] = pos_sorted
    return counts, g_idx, v_idx


class TimesFM3Backbone(nn.Module):
    """TimesFM3WithCrossAttn behind a chronos-2 shaped interface."""

    def __init__(self, tfm: TimesFM3WithCrossAttn, context_length: int = CHRONOS_CONTEXT_LENGTH):
        super().__init__()
        self.tfm = tfm
        # The loss space (isomorphic to the chronos arm, arcsinh included) and the TimesFM input space
        # (purely linear) are kept separate; see the module docstring.
        self.instance_norm = InstanceNorm(use_arcsinh=True)
        self.input_norm = InstanceNorm(use_arcsinh=False)
        self.register_buffer("quantiles", torch.tensor(tfm.quantiles, dtype=torch.float32), persistent=False)
        self.num_quantiles = len(tfm.quantiles)

        d_model = tfm.transformer_config.transformer.model_dims
        self.config = SimpleNamespace(d_model=d_model)
        self.chronos_config = SimpleNamespace(
            input_patch_size=tfm.input_patch_len,
            output_patch_size=tfm.output_patch_len,
            context_length=int(context_length),
            quantiles=list(tfm.quantiles),
        )

    # ---- odds and ends of the chronos interface ----

    @property
    def encoder(self):
        """The trunk reads the layer count as ``len(chronos.encoder.block)``.

        Deliberately a property rather than an instance attribute: PEFT's ``modules_to_save``
        deep-copies the whole ``chronos`` submodule, and a ModuleList referenced from an instance
        attribute would be copied along with it into a duplicate detached from the module tree
        (330M parameters, about 1.3G of memory, and never used). A property does not enter the object
        graph.
        """
        return SimpleNamespace(block=self.tfm.transformer_stack.layers)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.tfm.parameters()).dtype

    def set_cross_attn_layers(self, spec):
        return self.tfm.set_cross_attn_layers(spec)

    # The pinball loss borrows the chronos-side implementation directly (an unbound function, bound to
    # this instance) -- there must be exactly one loss implementation.
    _masked_pinball = Chronos2WithCrossAttn._masked_pinball

    # ---- the core: folded rows <-> the TimesFM variate axis ----

    def _run_tfm(
        self,
        context: torch.Tensor,
        gids: torch.Tensor | None,
        loc: torch.Tensor,
        scale: torch.Tensor,
        horizon: int,
        cross_states: torch.Tensor | None,
        cross_states_mask: torch.Tensor | None,
        context_mask: torch.Tensor | None = None,
    ):
        """(R, L) folded rows -> assembled into a variate axis per group -> TimesFM -> scattered back
        row by row.

        Returns ``(hidden (R, n_patch, d), qp_norm (R, Q, horizon))``, where qp lives in the
        **linearly normalised space** (the context fed in has already been through (x - loc) / scale,
        and TimesFM's output shares the space of its input).

        **Bucketing by channel count instead of padding**: TimesFM's attention goes through SDPA, and
        a row whose patch_mask is all True produces an all-False attn_mask => softmax returns NaN
        (upstream has no all-masked fallback). Treating pad rows as valid rows filled with zeros
        instead would pollute the representations of the real channels inside the variate attention.
        So the rows are bucketed by the number of variates in each group; every bucket is then
        naturally rectangular with no padding at all. A batch usually contains only 1-2 distinct
        channel counts, so there are very few buckets.
        """
        R, L = context.shape
        if gids is None:
            gids = torch.arange(R, device=context.device)
        counts, g_idx, v_idx = _group_to_variates(gids)

        # Apply the linear global normalisation before feeding TimesFM (see decision 2 above).
        ctx_n, _ = self.input_norm(context, (loc, scale))

        # WARNING: the padding must be handed to TimesFM as a mask; nan_to_num-ing it into zeros is
        # not enough. Histories are **left-padded with NaN**, and if the pad positions go in as real
        # zeros, TimesFM's causal running-stats compute std=0 over the all-zero prefix => RevIN divides
        # by zero => the forward pass is masked back into finite values by the subsequent where/clip
        # (so the loss looks normal) while the **backward pass is NaN throughout**.
        # As a side effect this also fixes "padding polluting the forecast as if it were a real 0".
        invalid = ~torch.isfinite(context)
        if context_mask is not None:
            invalid = invalid | (context_mask.to(context.device) <= 0)
        ctx_n = torch.nan_to_num(ctx_n, nan=0.0, posinf=0.0, neginf=0.0)

        d_model = self.config.d_model
        hidden_out: torch.Tensor | None = None
        qp_out: torch.Tensor | None = None

        for n_var in torch.unique(counts).tolist():
            sel_groups = (counts == n_var).nonzero(as_tuple=True)[0]
            row_sel = torch.isin(g_idx, sel_groups)
            rows = row_sel.nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            # Renumber the group ids into [0, G_bucket); positions within a group keep using v_idx.
            remap = torch.full((int(counts.numel()),), -1, dtype=torch.long, device=context.device)
            remap[sel_groups] = torch.arange(sel_groups.numel(), device=context.device)
            gb = remap[g_idx[rows]]
            vb = v_idx[rows]

            bucket = ctx_n.new_zeros(sel_groups.numel(), n_var, L)
            bucket[gb, vb] = ctx_n[rows]
            # The input must be cast to the backbone parameters' dtype -- under ZeRO-2 the parameters
            # are bf16 while the collator hands over an fp32 context, and feeding that straight in
            # raises "expected mat1 and mat2 to have the same dtype". The chronos side does the very
            # same thing inside _prepare_patched_context (context.to(self.dtype)).
            # WARNING: a single-GPU bf16 smoke test does not catch this -- on that path autocast
            # promotes automatically and the parameters are in fact still fp32.
            bucket = bucket.to(self.dtype)
            # True=masked (the same convention as TimesFM's target_mask). A bucket is rectangular with
            # no pad rows, so only genuine time-axis padding is flagged here.
            bucket_mask = torch.zeros(sel_groups.numel(), n_var, L, dtype=torch.bool, device=context.device)
            bucket_mask[gb, vb] = invalid[rows]

            cs = None
            if cross_states is not None:
                # cross_states arrives per row as (R, M, d), but all channels of one group share the
                # same stretch of LLM representation => taking the first row of each group is enough,
                # assembled into the (G, M, d) that TimesFM wants.
                cs = cross_states.new_zeros(sel_groups.numel(), cross_states.shape[1], cross_states.shape[2])
                cs[gb] = cross_states[rows]
            csm = None
            if cross_states_mask is not None and cs is not None:
                csm = cross_states_mask.new_zeros(sel_groups.numel(), cross_states_mask.shape[1])
                csm[gb] = cross_states_mask[rows]

            # WARNING: bf16 requires wrapping this in autocast. TimesFM is an inference-only
            # implementation (the whole model is fp32 by default) and hard-codes intermediates to fp32
            # in several places -- masks go through `.float()`, the initial running-stats (n, mu, sigma)
            # are fp32, and RoPE's sin/cos are computed as `pos.float()/timescale` (which promotes q/k
            # to fp32 while v, which skips RoPE, stays bf16). Patching them one at a time is
            # whack-a-mole; autocast is designed for exactly this situation: the inputs of
            # linear/matmul/SDPA are unified to the target dtype automatically, while RMSNorm and the
            # like keep computing in fp32 (which is more accurate). Under fp32 training enabled=False
            # and nothing changes value-wise.
            _amp = self.dtype in (torch.bfloat16, torch.float16)
            with torch.autocast(device_type=bucket.device.type, dtype=self.dtype, enabled=_amp):
                logits, aux = self.tfm.forecast(
                    bucket, horizon, target_mask=bucket_mask,
                    cross_states=cs, cross_states_mask=csm, return_aux_outputs=True
                )
            hid = aux["__call__:transformer_output"]        # (G, V, n_patch, d)
            if hidden_out is None:
                # Inherit the dtype of the backbone output (bf16 during bf16 training), matching the
                # convention of what chronos's encode returns.
                hidden_out = hid.new_zeros(R, hid.shape[2], d_model)
                qp_out = logits.new_zeros(R, horizon, self.num_quantiles)
            hidden_out[rows] = hid[gb, vb].to(hidden_out.dtype)
            qp_out[rows] = logits[gb, vb].to(qp_out.dtype)

        return hidden_out, qp_out.transpose(1, 2)           # (R, n_patch, d), (R, Q, horizon)

    # ---- encode, in chronos shape ----

    def encode(
        self,
        context: torch.Tensor,
        *,
        num_output_patches: int = 1,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        cross_states: torch.Tensor | None = None,
        cross_states_mask: torch.Tensor | None = None,
        stop_at_layer: int | None = None,
    ):
        """Returns ``(encoder_outputs, loc_scale, patched_future_cov_mask, num_context_patches)``.

        ``encoder_outputs[0]`` is (R, n_patch, d_model) with the context patches first -- the same
        convention as chronos, so both of the trunk's access patterns (``enc_out[0][:, :ncp]`` and
        ``hidden[:, -nop:]``) remain valid.
        """
        if future_covariates is not None:
            raise NotImplementedError(
                "The TimesFM adapter does not implement known-future covariates (the forecasting "
                "corpus is univariate throughout, with no covariates); silently ignoring "
                "them would leave 'the model cannot see the known future' unprovable, hence this "
                "explicit error.")
        if stop_at_layer is not None:
            raise NotImplementedError(
                "The TimesFM adapter does not implement stop_at_layer (architecture ablation A4 is "
                "out of scope for the backbone ablation)")

        if loc_scale is None:
            _, loc_scale = self.instance_norm(context)
        loc, scale = loc_scale
        horizon = max(1, int(num_output_patches)) * self.chronos_config.output_patch_size
        hidden, _ = self._run_tfm(
            context, group_ids, loc, scale, horizon, cross_states, cross_states_mask,
            context_mask=context_mask,
        )
        ncp = -(-context.shape[-1] // self.chronos_config.input_patch_size)
        return (hidden,), loc_scale, None, ncp

    # ---- forecast + losses, in chronos shape ----

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
        """Same signature and same return value as ``Chronos2WithCrossAttn.forecast_losses``."""
        if future_covariates is not None:
            raise NotImplementedError("The TimesFM adapter does not implement known-future covariates (see the note in encode)")

        _, loc_scale = self.instance_norm(context)
        loc, scale = loc_scale
        H = int(num_output_patches) * self.chronos_config.output_patch_size
        _, qp_lin = self._run_tfm(
            context, group_ids, loc, scale, H, cross_states, cross_states_mask,
            context_mask=context_mask,
        )  # (R, Q, H), in the linearly normalised space

        if target_idx is not None:
            qp_lin = qp_lin[target_idx]
            loc = loc[target_idx]
            scale = scale[target_idx]
        loc_scale_t = (loc, scale)

        # Move into arcsinh space to line up with the chronos arm's pinball loss: _masked_pinball runs
        # the target through instance_norm(use_arcsinh=True) internally, so the prediction side only
        # has to add the arcsinh (the linear half has already been applied).
        qp = torch.arcsinh(qp_lin)

        valid = torch.isfinite(future_target).float()
        pred_loss = self._masked_pinball(qp, future_target, valid, loc_scale_t)
        roi_loss = (
            self._masked_pinball(qp, future_target, roi_mask, loc_scale_t)
            if roi_mask is not None
            else None
        )
        qp_un = qp_lin * scale.unsqueeze(1) + loc.unsqueeze(1)   # de-normalise back to the original scale
        return {"pred_loss": pred_loss, "roi_loss": roi_loss, "quantile_preds": qp_un}

    def forward(
        self,
        context: torch.Tensor,
        num_output_patches: int = 1,
        group_ids: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        cross_states: torch.Tensor | None = None,
        cross_states_mask: torch.Tensor | None = None,
        **_,
    ):
        """Inference entry point: returns an object carrying ``.quantile_preds`` (original scale,
        (R, Q, H)), shaped exactly like the chronos one."""
        if future_covariates is not None:
            raise NotImplementedError("The TimesFM adapter does not implement known-future covariates (see the note in encode)")
        _, (loc, scale) = self.instance_norm(context)
        H = int(num_output_patches) * self.chronos_config.output_patch_size
        _, qp_lin = self._run_tfm(
            context, group_ids, loc, scale, H, cross_states, cross_states_mask,
            context_mask=context_mask,
        )
        return SimpleNamespace(quantile_preds=qp_lin * scale.unsqueeze(1) + loc.unsqueeze(1))


def load_timesfm3_backbone(
    ckpt_path: str,
    dtype: torch.dtype | None = None,
    random_init: bool = False,
    context_length: int = CHRONOS_CONTEXT_LENGTH,
) -> TimesFM3Backbone:
    """Load TimesFM-3.0 and wrap it in the chronos interface shape (``ChronosLLM`` then uses it
    directly as its ``chronos``)."""
    tfm = load_timesfm3_with_cross_attn(ckpt_path, dtype=dtype, random_init=random_init)
    return TimesFM3Backbone(tfm, context_length=context_length)
