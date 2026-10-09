"""Inject gated cross-attention into TimesFM-3.0 -- the host-side work for the backbone-ablation arm
that swaps the TSFM.

This is the same construction as ``cross_attn_chronos.py``, for a different host. The injected
``GatedCrossAttention`` is **deliberately line-for-line isomorphic to the chronos version** (same
LayerNorm + nn.MultiheadAttention + scalar tanh gate; only d_model / num_heads come from the host):
the ablation must vary one thing only, namely the host TSFM. If the injection mechanism differed as
well, an observed difference could not be attributed to "a different backbone" rather than to "a
different injection implementation".

**Injection point**: every layer of TimesFM's ``MixingTransformer`` is
``sequence attention -> variate attention -> FFN``, which maps segment for segment onto Chronos-2's
``time self-attn -> group self-attn -> FFN`` (sequence<->time, variate<->group), so the injection
point is likewise taken **after the variate-axis attention and before the FFN**.

**Why the upstream forward/decode are not copied**: ``TimesFM3Torch.decode`` carries
``@torch.no_grad()`` (upstream is an inference-only implementation), but the 285 lines of patching /
stitching / CPM-RevIN / linear detrending inside it are exactly what must be reused value-for-value --
a copy would be left to drift as upstream evolves. Two tricks avoid copying anything:
  1. the undecorated ``decode`` is recovered through ``__wrapped__`` (``torch.no_grad`` wraps with
     functools.wraps, so the original function is directly reachable) => the training path gets the
     very same implementation and can backpropagate through it;
  2. ``cross_states`` never enters the upstream signature; a context manager attaches it to the
     transformer stack instead => upstream ``forward``/``decode`` pass the feedback through without a
     single line of change.

WARNING: ``decode`` here **no longer carries no_grad** (training needs the graph). Wrap
inference/evaluation call sites in ``torch.no_grad()`` yourself -- our eval path already runs inside
a no_grad context.
"""

from __future__ import annotations

import contextlib
import json
import os

import torch
import torch.nn as nn

from timesfm3.torch import configs as tfm_configs
from timesfm3.torch import util as tfm_util
from timesfm3.torch.model import TimesFM3Torch
from timesfm3.torch.transformer import MixingTransformer, StackedMixingTransformer


# ---------------------------------------------------------------------------
# **Backward**-pass numerical-stability patch for the upstream CPM-RevIN statistics
# (the forward pass stays bit-identical)
# ---------------------------------------------------------------------------
# Upstream ``util.update_running_stats`` contains three ``x / n`` divisions and two ``sqrt(v)``
# calls. When a patch is **entirely masked** (inc_n=0, i.e. the leading patches of a left-padded
# history) or the variance happens to be exactly 0:
#   the forward pass is fine -- the outer ``torch.where(inc_n == 0, zeros, ...)`` discards the 0/0
#   result;
#   the backward pass is all NaN -- the branch that was not selected still participates in
#   differentiation, ``d(a/n)/da = 1/n = inf`` and ``d sqrt(v)/dv = 1/(2*sqrt(0)) = inf``, and
#   multiplying that by the 0 handed down by ``where`` gives ``0 * inf = NaN``.
# Upstream is inference-only (``decode`` carries ``@torch.no_grad``) so this path is unreachable
# there; training walks straight into it: with NaN left-padding every backbone gradient is NaN
# while the loss itself stays finite.
# The fix is the standard double-``where``: feed a safe value into the dangerous branch and let the
# outer ``where`` still select the true value => **the forward pass is bit-identical**, only the
# backward pass stops producing inf. ``test_timesfm_running_stats_patch`` guards exactly this.


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Divide by 1 when den==0 (that branch's value is discarded by the outer where), which keeps the
    backward pass from producing 1/0=inf."""
    return num / torch.where(den == 0, torch.ones_like(den), den)


def _safe_sqrt(v: torch.Tensor) -> torch.Tensor:
    """Return 0 in the forward pass when v<=0 (same as sqrt(0)), with gradient 0 instead of inf."""
    positive = v > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, v, torch.ones_like(v))),
                       torch.zeros_like(v))


def _update_running_stats_safe(n, mu, sigma, x, mask):
    """Backward-safe version of ``util.update_running_stats``; the forward pass is bit-identical to
    upstream."""
    is_legit = ~mask
    inc_n = is_legit.float().sum(dim=-1)

    x_masked = torch.where(is_legit, x, torch.zeros_like(x))
    inc_sum = x_masked.sum(dim=-1)
    inc_mu = torch.where(inc_n == 0, torch.zeros_like(inc_sum), _safe_div(inc_sum, inc_n))

    x_diff_sq = torch.where(is_legit, (x - inc_mu.unsqueeze(-1)) ** 2, torch.zeros_like(x))
    inc_var = torch.where(inc_n == 0, torch.zeros_like(inc_sum),
                          _safe_div(x_diff_sq.sum(dim=-1), inc_n))
    inc_sigma = _safe_sqrt(inc_var)

    new_n = n + inc_n
    new_mu = torch.where(new_n == 0, torch.zeros_like(mu),
                         _safe_div(n * mu + inc_mu * inc_n, new_n))
    new_var = torch.where(
        new_n == 0,
        torch.zeros_like(sigma),
        _safe_div(
            n * sigma * sigma
            + inc_n * inc_sigma * inc_sigma
            + n * (mu - new_mu) * (mu - new_mu)
            + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu),
            new_n,
        ),
    )
    return new_n, new_mu, _safe_sqrt(new_var)


_ORIG_UPDATE_RUNNING_STATS = tfm_util.update_running_stats
tfm_util.update_running_stats = _update_running_stats_safe


class GatedCrossAttention(nn.Module):
    """Gated cross-attn: TimesFM hidden states as query, the compressed LLM representation as key/value.

    Isomorphic to ``cross_attn_chronos.GatedCrossAttention``. The gate is a scalar wrapped in
    ``tanh``: with gate=0 or cross_states=None the layer is strictly the identity, so the whole model
    is numerically identical to the original TimesFM3.
    (Production training instead uses ``GATE_INIT=1.0`` together with ``--zero_cross_out_proj``, which
    moves the identity fallback onto the zero-initialised out_proj. Reason: a zero-initialised gate
    deadlocks the gradient -- the cross-attn weights' gradient is scaled away by ``tanh(g)=0``, so the
    feedback path never learns, opening the gate buys nothing, and the gate therefore never opens.)
    """

    def __init__(self, config: tfm_configs.TransformerConfig):
        super().__init__()
        d_model = config.model_dims
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        # TimesFM's TransformerConfig has no dropout field (an inference-oriented implementation with
        # no dropout anywhere in the network), so 0.0 is used here; the chronos version takes its
        # host's dropout_rate=0.1. This is a pre-existing difference inherited from the host.
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: torch.Tensor,
        cross_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            hidden_states: (B, T, d_model) TimesFM hidden states (query).
            cross_states: (B, M, d_model) compressed LLM representation (key/value).
            cross_key_padding_mask: (B, M) bool, True=padding (ignored).
        """
        q = self.q_norm(hidden_states)
        kv = self.kv_norm(cross_states)
        attn_out, _ = self.attn(
            q, kv, kv, key_padding_mask=cross_key_padding_mask, need_weights=False
        )
        return hidden_states + torch.tanh(self.gate) * attn_out


class CrossAttnMixingTransformer(MixingTransformer):
    """MixingTransformer + a gated cross-attn inserted after the variate attention and before the FFN.

    The forward body is copied from upstream verbatim (to stay value-identical); the only addition is
    the injection block in the middle.
    """

    def __init__(self, config: tfm_configs.TransformerConfig, use_variate_attention: bool = True):
        super().__init__(config, use_variate_attention=use_variate_attention)
        self.cross_attn = GatedCrossAttention(config)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        segment_pos: torch.Tensor | None = None,
        decode_cache=None,
        var_segment_pos: torch.Tensor | None = None,
        cross_states: torch.Tensor | None = None,
        cross_key_padding_mask: torch.Tensor | None = None,
    ):
        b, v, n, d = input_embeddings.shape

        # --- Sequence Attention (verbatim from upstream) ---
        seq_attn_in = self.pre_seq_attn_ln(input_embeddings)
        seq_attn_in_flat = seq_attn_in.reshape(b * v, n, d)
        patch_mask_flat = patch_mask.reshape(b * v, n)

        seq_seg_ids_flat = None
        if segment_ids is not None:
            seq_seg_ids_flat = segment_ids.unsqueeze(1).expand(b, v, n).reshape(b * v, n)
        seq_seg_pos_flat = None
        if segment_pos is not None:
            seq_seg_pos_flat = segment_pos.unsqueeze(1).expand(b, v, n).reshape(b * v, n)

        seq_attn_out_flat, decode_cache, seq_attn_mask = self.seq_attn(
            seq_attn_in_flat,
            segment_ids=seq_seg_ids_flat,
            segment_pos=seq_seg_pos_flat,
            decode_cache=decode_cache,
            patch_mask=patch_mask_flat,
        )
        seq_attn_out = seq_attn_out_flat.view(b, v, n, d)
        h1 = self.post_seq_attn_ln(seq_attn_out) + input_embeddings

        # --- Variate Attention (verbatim from upstream) ---
        if self.use_variate_attention:
            var_attn_in = self.pre_var_attn_ln(h1)
            var_attn_in_flat = var_attn_in.permute(0, 2, 1, 3).reshape(b * n, v, d)
            var_patch_mask = patch_mask.permute(0, 2, 1).reshape(b * n, v)
            var_attn_out_flat, _, _ = self.var_attn(
                var_attn_in_flat,
                segment_pos=var_segment_pos,
                decode_cache=None,
                patch_mask=var_patch_mask,
            )
            var_attn_out = var_attn_out_flat.view(b, n, v, d).permute(0, 2, 1, 3)
            h2 = self.post_var_attn_ln(var_attn_out) + h1
        else:
            h2 = h1

        # --- Gated cross-attention (only when cross_states is provided) ---
        # The query is flattened along the variate axis into (b*v, n, d): every variate of a sample
        # reads the same stretch of LLM representation. cross_states (b, M, d) is expanded into
        # (b*v, M, d) with repeat_interleave -- reshape(b*v, ...) orders b on the outside and v on the
        # inside, so it must be interleave and not repeat (the latter would cross samples over).
        if cross_states is not None:
            q = h2.reshape(b * v, n, d)
            kv = cross_states.repeat_interleave(v, dim=0)
            kpm = None
            if cross_key_padding_mask is not None:
                kpm = cross_key_padding_mask.repeat_interleave(v, dim=0)
            h2 = self.cross_attn(q, kv, cross_key_padding_mask=kpm).view(b, v, n, d)

        # --- FeedForward (verbatim from upstream) ---
        ff_out = self.ff1(self.activation(self.ff0(self.pre_ff_ln(h2))))
        output_embeddings = self.post_ff_ln(ff_out) + h2

        return output_embeddings, decode_cache, seq_attn_mask


class CrossAttnStackedMixingTransformer(StackedMixingTransformer):
    """Same as StackedMixingTransformer, but every layer is a CrossAttnMixingTransformer.

    ``cross_states`` does not travel through the forward signature (the upstream
    ``TimesFM3Torch.forward`` call site is fixed); ``TimesFM3WithCrossAttn._cross_ctx`` attaches it to
    this object temporarily instead, so upstream forward/decode pass the feedback through without a
    single line of change. These attributes are not nn.Module state and do not enter the state_dict.
    """

    def __init__(self, config: tfm_configs.StackedTransformersConfig, use_variate_attention: bool = True):
        super().__init__(config, use_variate_attention=use_variate_attention)
        self.layers = nn.ModuleList(
            [
                CrossAttnMixingTransformer(
                    config=config.transformer, use_variate_attention=use_variate_attention
                )
                for _ in range(config.num_layers)
            ]
        )
        # Same as the chronos-side architecture ablation B3: inject the feedback into a subset of the
        # layers only (None = all layers = default).
        self.cross_attn_layer_ids: set[int] | None = None
        self._cross_states: torch.Tensor | None = None
        self._cross_kpm: torch.Tensor | None = None

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        segment_pos: torch.Tensor | None = None,
        decode_cache=None,
        var_segment_pos: torch.Tensor | None = None,
    ):
        if decode_cache is None:
            decode_cache = [None] * len(self.layers)

        cross_states = self._cross_states
        cross_kpm = self._cross_kpm
        allow = self.cross_attn_layer_ids

        output = input_embeddings
        new_caches, attn_masks = [], []
        for i, layer in enumerate(self.layers):
            output, layer_cache, layer_mask = layer(
                output,
                patch_mask,
                segment_ids,
                segment_pos,
                decode_cache[i],
                var_segment_pos,
                cross_states=(cross_states if (allow is None or i in allow) else None),
                cross_key_padding_mask=cross_kpm,
            )
            new_caches.append(layer_cache)
            attn_masks.append(layer_mask)

        return output, new_caches, attn_masks


def _cast_input_to_weight_dtype(module, args):
    """forward_pre_hook: cast the input to this module's weight dtype (only when they differ)."""
    if not args:
        return None
    x = args[0]
    w = module.hidden_layer.weight
    if isinstance(x, torch.Tensor) and x.dtype != w.dtype:
        return (x.to(w.dtype),) + tuple(args[1:])
    return None


class TimesFM3WithCrossAttn(TimesFM3Torch):
    """TimesFM3Torch + gated cross-attention in every layer.

    Three extra entry points:
      - ``forecast(context, horizon, cross_states=...)``: a differentiable decode (the undecorated
        upstream decode implementation), returning quantiles (b, v, horizon, num_quantiles);
      - ``encode(context, cross_states=...)``: (b, v, n_ctx_patch, d_model) patch representations to
        serve as KV for the history Q-former;
      - ``set_cross_attn_layers(spec)``: restrict the injection to a subset of layers (for ablations).
    """

    # The original decode implementation, without @torch.no_grad (see the module docstring).
    _decode_impl = staticmethod(TimesFM3Torch.decode.__wrapped__)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap in the stack that carries cross-attn (weights are installed by load_* below).
        self.transformer_stack = CrossAttnStackedMixingTransformer(
            config=self.transformer_config,
            use_variate_attention=self.use_variate_attention,
        )
        # dtype alignment for bf16 training (required once ZeRO-2 casts the parameters to bf16):
        # upstream ``_preprocess`` does ``torch.cat([values_cat, masks_cat.float()], -1)``, hard-coding
        # the mask to fp32; concatenating it with bf16 values promotes the whole tensor back to fp32,
        # which then collides with the bf16 resblock weights => "expected mat1 and mat2 to have the
        # same dtype". ``get_running_stats`` likewise hard-codes the initial (n, mu, sigma) to fp32,
        # and revin pulls the values back to fp32 the same way. Both are leftovers of the upstream
        # inference-only design (the whole model is fp32 there).
        # Rather than copying 75 lines of upstream _preprocess to change one line, a forward_pre_hook
        # at the resblock entry aligns the input to the weight dtype -- zero copying, and since the
        # whole chain downstream of the resblock consumes its output, everything below is bf16 anyway.
        self.pre_transformer_resblock.register_forward_pre_hook(_cast_input_to_weight_dtype)

    # ---- passing cross_states through ----

    @contextlib.contextmanager
    def _cross_ctx(self, cross_states: torch.Tensor | None, cross_states_mask: torch.Tensor | None):
        """Attach cross_states to the transformer stack temporarily; always cleaned up on exit.

        cross_states_mask is a (B, M) **valid mask (1=valid)**, the same convention as on the chronos
        side; it is converted here into the key_padding_mask (True=pad) that nn.MultiheadAttention
        expects.
        """
        stack = self.transformer_stack
        prev = (getattr(stack, "_cross_states", None), getattr(stack, "_cross_kpm", None))
        kpm = None
        if cross_states is not None and cross_states_mask is not None:
            kpm = cross_states_mask <= 0
        stack._cross_states, stack._cross_kpm = cross_states, kpm
        try:
            yield
        finally:
            stack._cross_states, stack._cross_kpm = prev

    def set_cross_attn_layers(self, spec: str | None) -> list[int]:
        """Choose the subset of layers that receive the feedback: None/"all" = every layer;
        "last6"/"first6"; or an explicit list such as "0,3,5"."""
        n = len(self.transformer_stack.layers)
        if spec is None or spec == "all":
            self.transformer_stack.cross_attn_layer_ids = None
            return list(range(n))
        if spec.startswith("last"):
            ids = list(range(max(0, n - int(spec[4:])), n))
        elif spec.startswith("first"):
            ids = list(range(min(n, int(spec[5:]))))
        else:
            ids = [int(x) for x in spec.split(",") if x.strip() != ""]
            bad = [i for i in ids if not (0 <= i < n)]
            if bad:
                raise ValueError(f"cross_attn_layers out of range: {bad} ({n} layers in total)")
        self.transformer_stack.cross_attn_layer_ids = set(ids)
        return sorted(ids)

    # ---- forward entry points ----

    def forward(self, inputs, *, cross_states=None, cross_states_mask=None, **kwargs):
        # WARNING: cross_states=None must **not** enter _cross_ctx. ``forecast`` first sets up the
        # context and then runs the upstream decode, and decode internally calls back into
        # ``self.forward(inputs, ...)`` without cross_states -- if that inner call also entered the
        # context, it would overwrite the cross_states the outer call had just attached with None and
        # the feedback would silently stop working (check 4 of test_timesfm_cross_attn guards this).
        if cross_states is None:
            return super().forward(inputs, **kwargs)
        with self._cross_ctx(cross_states, cross_states_mask):
            return super().forward(inputs, **kwargs)

    def forecast(
        self,
        context: torch.Tensor,
        horizon: int,
        *,
        cross_states: torch.Tensor | None = None,
        cross_states_mask: torch.Tensor | None = None,
        return_aux_outputs: bool = False,
        **decode_kwargs,
    ):
        """Differentiable version of decode. The arguments match upstream ``decode``
        (target/past_only_covariates/...).

        WARNING: the only difference from upstream decode is that there is **no no_grad**; wrap
        inference call sites in torch.no_grad() yourself.
        """
        with self._cross_ctx(cross_states, cross_states_mask):
            return type(self)._decode_impl(
                self, context, horizon, return_aux_outputs=return_aux_outputs, **decode_kwargs
            )

    def encode(
        self,
        context: torch.Tensor,
        *,
        cross_states: torch.Tensor | None = None,
        cross_states_mask: torch.Tensor | None = None,
        **decode_kwargs,
    ) -> torch.Tensor:
        """Return the (b, v, n_ctx_patch, d_model) patch representations that serve as KV for the
        history Q-former.

        It takes the same path as ``forecast`` (reusing the upstream patching / normalisation instead
        of writing a second version that could drift out of sync), picks transformer_output out of aux
        and cuts off the patches belonging to the horizon. The horizon is set to the smallest legal
        value of one output_patch_len (decode requires horizon>0) and the extra patches are simply
        discarded.
        """
        _, aux = self.forecast(
            context,
            self.output_patch_len,
            cross_states=cross_states,
            cross_states_mask=cross_states_mask,
            return_aux_outputs=True,
            **decode_kwargs,
        )
        hid = aux["__call__:transformer_output"]          # (b, v, n_total_patch, d)
        ctx_len = context.shape[-1]
        # Upstream **left**-pads a context that is not a whole number of patches.
        n_ctx_patches = -(-ctx_len // self.input_patch_len)
        return hid[:, :, :n_ctx_patches, :]


def load_timesfm3_with_cross_attn(
    ckpt_path: str,
    dtype: torch.dtype | None = None,
    random_init: bool = False,
) -> TimesFM3WithCrossAttn:
    """Construct TimesFM3WithCrossAttn and install the pretrained weights.

    ``PyTorchModelHubMixin.from_pretrained`` is deliberately bypassed: it loads strictly, so the
    cross-attn parameters we added would be reported as unexpected/missing. Here the model is
    constructed explicitly and loaded with ``load_state_dict(strict=False)``, followed by a check that
    "the only missing keys may be cross_attn ones" -- the same fail-loudly reasoning as in
    ``load_chronos2_with_cross_attn`` (chronos-2 is the case where the weights were silently not
    loaded at all).

    ``random_init=True``: take only the architecture from the config and skip the weight load (used by
    the ablation on whether TSFM pretraining matters).
    """
    from safetensors.torch import load_file

    with open(os.path.join(ckpt_path, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)

    model = TimesFM3WithCrossAttn(**cfg)
    if not random_init:
        raw = load_file(os.path.join(ckpt_path, "model.safetensors"))
        missing, unexpected = model.load_state_dict(raw, strict=False)
        leftover = [k for k in missing if ".cross_attn." not in k]
        if leftover:
            raise RuntimeError(f"After loading the TimesFM3 weights, {len(leftover)} non-cross_attn tensors are still missing: {leftover[:8]}")
        if unexpected:
            raise RuntimeError(f"TimesFM3 checkpoint contains keys the model does not know: {unexpected[:8]}")
    if dtype is not None:
        model = model.to(dtype)
    return model
