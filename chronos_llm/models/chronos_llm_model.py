"""Composite model: Chronos-2 (+cross-attn) + Qwen3.5 (LoRA) + two Q-formers.

Data flow (see the repository README):
- Understanding branch: history -> chronos2.encode -> sliding-window Q-former -> soft prompt
  injected into the LLM -> text CE.
- Forecasting branch: same soft prompt; the LLM is teacher-forced on
  `<think>reasoning</think>conclusion` to produce a CE loss, its hidden states are compressed by
  the feedback Q-former into (B,M,768) and fed back into chronos2 as `cross_states` through gated
  cross-attention; chronos2 then combines them with the history to produce the final forecast ->
  quantile loss + ROI-weighted loss.

Trainable parameters: LLM LoRA + (optional modules_to_save) + full chronos2 + both Q-formers
(chronos2 / Q-formers are ordinary submodules of this module, trainable by construction and not
wrapped by PEFT).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from .cross_attn_chronos import Chronos2WithCrossAttn, load_chronos2_with_cross_attn
from .qformer import QFormer, SegmentMeanPooler, SlidingWindowPooler, SlidingWindowQFormer

TS_START = "<ts_start>"
TS_END = "<ts_end>"

# LoRA is applied to the LLM only: the regex is anchored on the ^llm. prefix so that the out_proj
# of the MHA blocks inside chronos2 / the Q-formers is not matched by accident.
# Qwen3.5 has both full attention (q/k/v/o_proj) and linear attention (in_proj_*/out_proj) plus MLP.
LLM_LORA_TARGET_REGEX = (
    r"^llm\..*\.(q_proj|k_proj|v_proj|o_proj|"
    r"in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|"
    r"gate_proj|up_proj|down_proj)$"
)
# Non-LLM submodules marked trainable via PEFT modules_to_save and saved with the adapter
# (lm_head is deliberately excluded).
TRAINABLE_MODULES = ("chronos", "history_qformer", "feedback_qformer")


class ChronosLLMConfig(PretrainedConfig):
    model_type = "chronos_llm"

    def __init__(
        self,
        chronos_ckpt: str = "",
        llm_path: str = "",
        # Sliding-window Q-former (history -> LLM): ladder windows, non-overlapping, fixed budget,
        # minimum number of windows.
        sw_queries_per_window: int = 4,
        sw_target_windows: int = 16,
        sw_min_windows: int = 4,
        sw_window_ladder=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        # Number of global queries (G queries cross-attend over the whole C*P patch sequence to
        # provide the global view that the per-window branch lacks; their output is prepended to
        # the local tokens). Default 0: older checkpoints have no such key in config.json, so
        # loading them must not conjure up a randomly initialised global branch; the training
        # entry point train.py enables 16 by default.
        sw_global_queries: int = 0,
        # Log-linear token budget: #soft-prompt tokens = clamp(token_min + b*log2(P×C), token_min, token_max).
        sw_token_min: int = 40,
        sw_token_max: int = 200,
        sw_token_pc_ref: int = 32768,
        qformer_num_heads: int = 8,
        qformer_num_layers: int = 2,
        qformer_dropout: float = 0.0,
        # Internal working width of the Q-formers (BLIP-2 style: attention/FFN run at this width,
        # a final Linear projects to out_dim). 768 = chronos d_model (the information bottleneck of
        # the KV source); running internally at the LLM hidden width (4096) would make the
        # parameter count grow quadratically.
        qformer_hidden_dim: int = 768,
        # Sliding-window chunk size for long histories in chronos2; 0 = use chronos' own
        # context_length (recommended).
        chronos_window: int = 0,
        # Max number of rows per encode call during sliding-window encoding
        # (row = sample × channel × chunk); 0 = no bucketing (one large batch).
        # GroupSelfAttention's mask/scores are (T, R, R) and grow quadratically with the total row
        # count R, while groups never interact (block-diagonal) -- bucketing makes the cost linear
        # in R and is numerically equivalent to a single call.
        encode_max_rows: int = 128,
        # Multi-channel: size of the channel-id embedding table and the per-window KV channel cap of
        # the 2-D sliding window; switch for the relative intra-window temporal PE.
        max_channels: int = 64,
        sw_intra_window_pos: bool = True,
        # Short-window detection enhancements (off by default -> older checkpoints have no such
        # key in config.json and load unchanged):
        # - history_stats_token: window-scale statistics [loc, scale, range, max|x|, ...] of each
        #   history (per channel) are projected by a small MLP into one "statistics token" that is
        #   prepended to the soft prompt. The discriminative signal (anomalous windows fluctuate
        #   more) is exactly the scale/range that chronos2's per-series instance_norm divides
        #   away; this feeds it back to the LLM explicitly.
        # - short_min_patches: very short single-chunk histories (fewer patches than this value)
        #   are upsampled to that many patches so the chronos2 encoder / sliding-window Q-former
        #   have actual temporal structure to attend to (patch_size=16 is too coarse for a
        #   16-point series, which originally yields a single patch).
        history_stats_token: bool = False,
        short_min_patches: int = 0,
        # Feedback Q-former (LLM -> chronos)
        fb_num_query_tokens: int = 16,
        # -- Architecture-ablation switches (defaults = the released configuration; older
        #    checkpoints lack these keys in config.json => loading behaviour is unchanged) --
        # A1/B1: replace either Q-former by a segment mean-pooling control without learned queries
        #   (token count / positions / preprocessing identical, only the compression operator
        #   changes). "qformer" (default) | "pool".
        history_compressor: str = "qformer",
        feedback_compressor: str = "qformer",
        # A4: which chronos2 encoder layer the soft prompt is taken from (1-indexed; early exit
        #   also saves compute). 0 = last layer + final_layer_norm (default).
        history_encode_layer: int = 0,
        # B2: which LLM layer's hidden states are fed back (1-indexed). 0 = last layer (default).
        feedback_llm_layer: int = 0,
        # B3: which chronos2 encoder blocks receive the feedback ("all"/"lastK"/"firstK"/"0,3,6").
        cross_attn_layers: str = "all",
        # Ablation switches (defaults = the released configuration; older checkpoints lack these
        # keys in config.json and load unchanged):
        # - chronos_only: the forecasting branch skips the LLM / soft prompt / feedback entirely
        #   and only fine-tunes chronos2 (cross_states=None) with the quantile/ROI losses
        #   (lower-bound control for the "textual feedback gain").
        # - feedback_scope: which span of the LLM hidden states is fed back to chronos --
        #   "all" (whole sequence, default) / "conclusion" (only the conclusion span after
        #   </think>; soft prompt / prompt / reasoning are all masked out).
        chronos_only: bool = False,
        feedback_scope: str = "all",
        # - forecast_prompt_only: no reasoning is generated; the hidden states of the (condensed,
        #   plain) prompt text are fed back directly. The dataset emits prompt-only ids (labels all
        #   -100 => text CE = 0); evaluation runs a plain forward pass, no autoregression.
        forecast_prompt_only: bool = False,
        # - chronos_random_init: ablation on whether TSFM pretraining matters -- only the
        #   config.json of chronos_ckpt is used to define the architecture, the pretrained weights
        #   are skipped and chronos2 is trained from random initialisation (architecture and the
        #   rest of the pipeline unchanged).
        chronos_random_init: bool = False,
        # - tsfm_backbone: swap the TSFM backbone (backbone ablation). "chronos2" is the released
        #   recipe; "timesfm3" runs TimesFM-3.0 instead -- chronos_ckpt then points at the TimesFM
        #   weight directory and the timesfm_backbone adapter wraps it into the chronos interface
        #   shape. Older checkpoints lack this key => defaults to chronos2, behaviour unchanged.
        tsfm_backbone: str = "chronos2",
        # - llm_only: LLM-only baseline (the symmetric ablation that removes the TSFM) -- chronos2
        #   and both Q-formers are not loaded at all; the time series is textualised on the dataset
        #   side (ts_as_text) and enters the LLM as plain text. Both branches only have the text CE;
        #   forecast values are parsed from the generated text (a point forecast replicated over
        #   the 21 quantiles). Older checkpoints lack this key => defaults to False, loading unchanged.
        llm_only: bool = False,
        # Loss weights
        text_loss_weight: float = 1.0,
        pred_loss_weight: float = 1.0,
        roi_loss_weight: float = 1.0,
        analysis_loss_weight: float = 1.0,
        # Misc
        llm_dtype: str = "bfloat16",
        chronos_max_output_patches: int = 64,
        # Base lineage for "merge, then open a fresh LoRA" continued training: when non-empty,
        # from_config assembles the bare base and then merges that checkpoint's adapter
        # (LoRA + modules_to_save) into it before returning -- the training start point and every
        # later from_pretrained reload share this single replay path, so the new adapter never
        # ends up stacked on the wrong base (base weights are never written to disk; deltas merged
        # into the LLM can only be recovered by replay). Default None = legacy behaviour.
        init_merged_from: str | None = None,
        **kwargs,
    ):
        self.chronos_ckpt = chronos_ckpt
        self.llm_path = llm_path
        self.sw_queries_per_window = sw_queries_per_window
        self.sw_target_windows = sw_target_windows
        self.sw_min_windows = sw_min_windows
        self.sw_window_ladder = tuple(sw_window_ladder)
        self.sw_global_queries = sw_global_queries
        self.sw_token_min = sw_token_min
        self.sw_token_max = sw_token_max
        self.sw_token_pc_ref = sw_token_pc_ref
        self.qformer_num_heads = qformer_num_heads
        self.qformer_num_layers = qformer_num_layers
        self.qformer_dropout = qformer_dropout
        self.qformer_hidden_dim = qformer_hidden_dim
        self.chronos_window = chronos_window
        self.encode_max_rows = encode_max_rows
        self.max_channels = max_channels
        self.sw_intra_window_pos = sw_intra_window_pos
        self.history_stats_token = history_stats_token
        self.short_min_patches = short_min_patches
        self.fb_num_query_tokens = fb_num_query_tokens
        self.history_compressor = history_compressor
        self.feedback_compressor = feedback_compressor
        self.history_encode_layer = history_encode_layer
        self.feedback_llm_layer = feedback_llm_layer
        self.cross_attn_layers = cross_attn_layers
        self.chronos_only = chronos_only
        self.feedback_scope = feedback_scope
        self.forecast_prompt_only = forecast_prompt_only
        self.chronos_random_init = chronos_random_init
        self.tsfm_backbone = tsfm_backbone
        self.llm_only = llm_only
        self.text_loss_weight = text_loss_weight
        self.pred_loss_weight = pred_loss_weight
        self.roi_loss_weight = roi_loss_weight
        self.analysis_loss_weight = analysis_loss_weight
        self.llm_dtype = llm_dtype
        self.chronos_max_output_patches = chronos_max_output_patches
        self.init_merged_from = init_merged_from
        super().__init__(**kwargs)


def _maybe_replay_merged_init(model, config):
    """If config.init_merged_from is set, merge that checkpoint's adapter into the bare base and return it.

    Lineage replay for "merge, then open a fresh LoRA" continued training: both the training start
    point (train.py --init_merge_reopen) and every later from_pretrained reload go through
    from_config -> this function, which guarantees the new adapter is always stacked on the same
    base ("original base + deltas merged in the previous round"). The LLM LoRA deltas and the
    trained chronos / Q-former values are all restored by replaying the adapter -- the base weights
    themselves are never written to disk. Only one level is supported: the replayed checkpoint must
    not itself carry init_merged_from (chained replay is not implemented; we refuse loudly instead
    of silently dropping the deltas from two rounds back).
    """
    merged_from = getattr(config, "init_merged_from", None)
    if not merged_from:
        return model
    inner = ChronosLLMConfig.from_pretrained(merged_from)
    if getattr(inner, "init_merged_from", None):
        raise ValueError(
            f"init_merged_from does not support nesting: {merged_from} itself carries "
            f"init_merged_from={inner.init_merged_from} (chained merge replay is not implemented)"
        )
    from peft import PeftModel

    peft = PeftModel.from_pretrained(model, merged_from, is_trainable=False)
    return peft.merge_and_unload()


class ChronosLLM(PreTrainedModel):
    config_class = ChronosLLMConfig
    supports_gradient_checkpointing = True
    # Feature dimension of history_stats_token: [loc, scale, log1p(scale), range, log1p(range), max|x|]
    _STATS_N_FEATURES = 6

    def __init__(
        self,
        config: ChronosLLMConfig,
        chronos: Chronos2WithCrossAttn,
        llm: nn.Module,
        tokenizer,
    ):
        super().__init__(config)
        self.chronos = chronos
        self.llm = llm
        self.tokenizer = tokenizer

        llm_hidden = llm.get_input_embeddings().weight.shape[1]
        self.llm_hidden = llm_hidden

        if getattr(config, "llm_only", False):
            # LLM-only baseline: no chronos, no Q-formers, no statistics token -- the time series is
            # textualised by the dataset and enters the LLM directly. The soft prompt is always
            # empty (_encode_history_to_soft_prompt returns (0, llm_hidden) per sample); the
            # <ts_start>/<ts_end> placeholders are kept (splicing an empty prompt is a no-op, so
            # the chat rendering is exactly isomorphic to the composite model).
            if chronos is not None:
                raise ValueError("chronos must be None when llm_only=True")
            self.history_qformer = None
            self.feedback_qformer = None
            self.history_stats_proj = None
            self.history_stats_token = False
            self.history_encode_layer = None
            self._fb_hook_out = None
            self.short_min_patches = 0
            self.chronos_window = 8192
            self.encode_max_rows = 0
            self._encode_grad_ckpt = False
            self.input_patch_size = 16
            self.ts_start_id = tokenizer.convert_tokens_to_ids(TS_START)
            self.ts_end_id = tokenizer.convert_tokens_to_ids(TS_END)
            self.think_end_id = tokenizer.convert_tokens_to_ids("</think>")
            return

        d_model = chronos.config.d_model

        hist_cls = (SlidingWindowPooler
                    if str(getattr(config, "history_compressor", "qformer")).lower() == "pool"
                    else SlidingWindowQFormer)
        self.history_qformer = hist_cls(
            in_dim=d_model,
            out_dim=llm_hidden,
            queries_per_window=config.sw_queries_per_window,
            target_windows=config.sw_target_windows,
            min_windows=config.sw_min_windows,
            window_ladder=config.sw_window_ladder,
            global_queries=getattr(config, "sw_global_queries", 0),
            num_heads=config.qformer_num_heads,
            num_layers=config.qformer_num_layers,
            dropout=config.qformer_dropout,
            max_channels=config.max_channels,
            intra_window_pos=config.sw_intra_window_pos,
            hidden_dim=config.qformer_hidden_dim,
            token_min=getattr(config, "sw_token_min", 40),
            token_max=getattr(config, "sw_token_max", 200),
            token_pc_ref=getattr(config, "sw_token_pc_ref", 32768),
        )
        # chronos2 sliding-window chunk size (= its context_length unless overridden explicitly).
        ctx_len = int(chronos.chronos_config.context_length)
        cw = int(config.chronos_window) if config.chronos_window else ctx_len
        self.chronos_window = max(1, min(cw, ctx_len))
        self.encode_max_rows = max(0, int(getattr(config, "encode_max_rows", 0)))
        # Tied to gradient_checkpointing_enable: once on, per-bucket encode / per-sample qformer
        # calls go through torch.utils.checkpoint, so peak activation memory drops to a single
        # bucket and no longer depends on the total patch count of a sample.
        self._encode_grad_ckpt = False
        self.input_patch_size = int(chronos.chronos_config.input_patch_size)
        if self.chronos_window % self.input_patch_size:
            import warnings
            # With several chunks each full window is rounded up to a multiple of the patch size
            # on its own, so the real patch count exceeds the whole-series estimate
            # ceil(tl/patch) -- the soft-token budget of dynamic batching (qformer.plan) is
            # underestimated.
            warnings.warn(
                f"chronos_window={self.chronos_window} is not a multiple of input_patch_size="
                f"{self.input_patch_size}: the soft-token estimate used by dynamic batching will be too small")
        # Feedback side: in_dim=llm_hidden (4096) is likewise narrowed internally to
        # qformer_hidden_dim so the compression happens at low width (out_dim=d_model is 768 anyway).
        if str(getattr(config, "feedback_compressor", "qformer")).lower() == "pool":
            self.feedback_qformer = SegmentMeanPooler(
                in_dim=llm_hidden, out_dim=d_model,
                num_query_tokens=config.fb_num_query_tokens,
            )
        else:
            self.feedback_qformer = QFormer(
                in_dim=llm_hidden,
                out_dim=d_model,
                num_query_tokens=config.fb_num_query_tokens,
                num_heads=config.qformer_num_heads,
                num_layers=config.qformer_num_layers,
                dropout=config.qformer_dropout,
                hidden_dim=config.qformer_hidden_dim,
            )
        # B3: subset of injection layers ("all" = every layer = default, changes no numbers).
        self.chronos.set_cross_attn_layers(getattr(config, "cross_attn_layers", "all"))
        # A4: which chronos2 layer the soft prompt comes from (1-indexed; 0 / beyond the layer count = last layer = default).
        n_enc = len(self.chronos.encoder.block)
        hel = int(getattr(config, "history_encode_layer", 0) or 0)
        if hel < 0 or hel > n_enc:
            raise ValueError(f"history_encode_layer={hel} out of range (chronos2 has {n_enc} layers, 0=last)")
        self.history_encode_layer = None if hel in (0, n_enc) else hel

        # Short-window detection enhancements (see the ChronosLLMConfig comments). Both are off by
        # default and independent; short_min_patches only changes the number of encoded patches
        # and adds no parameters; history_stats_token introduces one small projection trained from
        # scratch (assigned to the qformer lr group).
        self.short_min_patches = max(0, int(getattr(config, "short_min_patches", 0)))
        self.history_stats_token = bool(getattr(config, "history_stats_token", False))
        self.history_stats_proj = None
        if self.history_stats_token:
            hid = config.qformer_hidden_dim
            self.history_stats_proj = nn.Sequential(
                nn.Linear(self._STATS_N_FEATURES, hid),
                nn.GELU(),
                nn.LayerNorm(hid),
                nn.Linear(hid, llm_hidden),
            )

        self.ts_start_id = tokenizer.convert_tokens_to_ids(TS_START)
        self.ts_end_id = tokenizer.convert_tokens_to_ids(TS_END)
        # Used by ablation 1 (feedback_scope="conclusion"): locate </think> to cut out the conclusion span.
        self.think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        self._fb_hook_out = None
        self._install_fb_layer_hook()

    # ------------------------------------------------- B2: feed back an intermediate LLM layer
    def _llm_decoder_layers(self):
        dec = getattr(self.llm, "model", None) or self.llm.get_decoder()
        for obj in (dec, getattr(dec, "language_model", None), getattr(dec, "model", None)):
            layers = getattr(obj, "layers", None) if obj is not None else None
            if layers is not None:
                return layers
        raise RuntimeError("Could not locate the LLM decoder layers (required by feedback_llm_layer)")

    def _install_fb_layer_hook(self) -> None:
        """When ``feedback_llm_layer=k>0``, register a forward hook on the k-th decoder block that caches its output.

        Why a hook rather than ``output_hidden_states=True``: the latter materialises the
        ``(B, L, 4096)`` activations of all 33 layers (gigabytes at the several-hundred-thousand
        tokens seen under dynamic batching), whereas we only need one layer. Paths that already
        call ``self.llm(..., output_hidden_states=True)`` (generate / scheduled sampling) index the
        tuple directly and do not depend on the hook. Reads always go through ``_fb_hidden``,
        whose shape check guards against stale per-step residue left behind by generate.
        """
        k = int(getattr(self.config, "feedback_llm_layer", 0) or 0)
        if k <= 0:
            return
        layers = self._llm_decoder_layers()
        if k > len(layers):
            raise ValueError(f"feedback_llm_layer={k} out of range (LLM has {len(layers)} layers, 0=last)")

        def _hook(_mod, _inp, out):
            self._fb_hook_out = out[0] if isinstance(out, (tuple, list)) else out

        layers[k - 1].register_forward_hook(_hook)

    def _fb_hidden(self, default: torch.Tensor, all_hidden=None) -> torch.Tensor:
        """Select the LLM hidden states used for feedback: ``feedback_llm_layer<=0`` => last layer (default, bit-identical)."""
        k = int(getattr(self.config, "feedback_llm_layer", 0) or 0)
        if k <= 0:
            return default
        if all_hidden is not None:
            return all_hidden[k]
        h = self._fb_hook_out
        if h is None or h.shape[:2] != default.shape[:2]:
            raise RuntimeError(
                f"hook cache for feedback_llm_layer={k} does not match this forward pass "
                f"(cache={None if h is None else tuple(h.shape[:2])} vs {tuple(default.shape[:2])})")
        return h

    # --------------------------------------------------------- construction / loading entry points
    @classmethod
    def from_config(cls, config: ChronosLLMConfig) -> "ChronosLLM":
        """Cold start: build the bare base from a config object + two external checkpoints (chronos_ckpt / llm_path).

        Loads the pretrained chronos + the base LLM and creates the two Q-formers, **without LoRA**.
        For training, call ``add_lora`` on the returned base to attach PEFT. This is the entry point
        for a training start; use ``from_pretrained`` to load training artefacts.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        backbone = str(getattr(config, "tsfm_backbone", "chronos2")).lower()
        random_init = getattr(config, "chronos_random_init", False)
        if getattr(config, "llm_only", False):
            chronos = None  # LLM-only baseline: chronos2 is not loaded at all
        elif backbone == "timesfm3":
            # Backbone ablation: TimesFM-3.0 wrapped into the chronos interface shape by the
            # adapter in timesfm_backbone (same gated cross-attention feedback).
            from chronos_llm.models.timesfm_backbone import load_timesfm3_backbone

            chronos = load_timesfm3_backbone(config.chronos_ckpt, random_init=random_init)
            chronos.train()
        elif backbone == "chronos2":
            chronos = load_chronos2_with_cross_attn(config.chronos_ckpt, random_init=random_init)
            chronos.train()
        else:
            raise ValueError(f"unknown tsfm_backbone={backbone!r} (choose chronos2 / timesfm3)")

        tokenizer = AutoTokenizer.from_pretrained(config.llm_path, trust_remote_code=True)
        _add_ts_special_tokens(tokenizer)

        dtype = getattr(torch, config.llm_dtype)
        llm = AutoModelForCausalLM.from_pretrained(
            config.llm_path, trust_remote_code=True, dtype=dtype
        )
        # Qwen3.5's embedding / lm_head reserve spare rows (vocab_size=248320 > len(tokenizer)); the
        # newly added <ts_*> token ids fall inside that reserved region, so no resize is needed.
        # Forcing a resize to len(tokenizer) would actually **truncate** the embedding (dropping the
        # reserved rows). Hence we only grow when the vocabulary genuinely **exceeds** the current
        # number of embedding rows (never shrink).
        emb_rows = llm.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > emb_rows:
            llm.resize_token_embeddings(len(tokenizer))
        llm.config.use_cache = False
        return _maybe_replay_merged_init(cls(config, chronos, llm, tokenizer), config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, merge: bool = True,
                        is_trainable: bool = False, **kwargs):
        """Loading entry point for inference / continued training.

        ``path`` is a save directory (containing the ``ChronosLLMConfig`` config.json):
        1. Read the config -> ``from_config`` **rebuilds the base** from ``chronos_ckpt`` /
           ``llm_path`` (note: the base weights come from those two external checkpoints, not from
           the save directory -- the directory only holds the PEFT adapter).
        2. If the directory contains a PEFT adapter (``adapter_config.json``):
           ``PeftModel.from_pretrained`` stacks the LoRA + the trained chronos / qformer
           (modules_to_save) on top; with ``merge=True`` (default) ``merge_and_unload`` folds the
           LoRA into the base and returns a plain ``ChronosLLM`` (use this for inference).
        3. No adapter: return the bare base (equivalent to ``from_config``).

        **Continued training** comes in two flavours with different semantics:
        - ``merge=False, is_trainable=True``: returns a trainable ``PeftModel`` (keep training the
          same adapter). ``is_trainable`` defaults to False, which is PEFT's inference mode -- the
          whole adapter (including modules_to_save) is frozen, so training it directly either
          fails with "no trainable parameters" or silently does not update; continued training
          **must** set it to True explicitly.
        - ``merge=True`` followed by ``add_lora`` to open a new round: the adapter saved by the new
          round only contains the new LoRA plus the chronos / qformer snapshot; **the deltas already
          merged into the LLM in the previous round are not written to disk** (a later
          ``from_pretrained`` still rebuilds the original LLM from config.llm_path, so the new LoRA
          would be stacked on the wrong base). To take this route the new run's config must carry
          ``init_merged_from=<previous checkpoint>`` (this is what train.py ``--init_merge_reopen``
          means): from_config replays the previous merge before returning the base (see
          ``_maybe_replay_merged_init``), keeping the training start and the reload lineage
          consistent; otherwise this route is only suitable for the "final round".

        Overrides the base ``PreTrainedModel.from_pretrained`` -- this model's base comes from
        external checkpoints plus adapter deltas, so the standard "all weights in one directory"
        semantics of from_pretrained do not apply.
        """
        import os

        path = str(pretrained_model_name_or_path)
        config = ChronosLLMConfig.from_pretrained(path, **kwargs)
        base = cls.from_config(config)

        if os.path.exists(os.path.join(path, "adapter_config.json")):
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, path, *model_args, is_trainable=is_trainable)
            if merge:
                model = model.merge_and_unload()  # returns the merged base (plain ChronosLLM)
            return model
        return base

    # -------------------------------------------------------------- internals
    def _series_stats(self, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Statistics token: window-scale statistics of the **raw** (un-normalised) series -> feature vector (``_STATS_N_FEATURES``,) fed back to the LLM.

        The discriminative signal (anomalous windows fluctuate strongly, normal windows are flat) is
        exactly the scale/range that chronos2's per-series ``instance_norm`` (division by the window
        std = scale) erases -- here it is handed back to the LLM explicitly. ``loc``/``scale`` reuse
        the values already computed by instance_norm (= window mean/std); range and max|x| are
        computed here (NaN-safe).
        """
        x = x.float()
        finite = x[torch.isfinite(x)]
        if finite.numel() == 0:
            return x.new_zeros(self._STATS_N_FEATURES)
        rng = (finite.max() - finite.min()).clamp_min(0)
        mx = finite.abs().max()
        loc = loc.reshape(()).float()
        scale = scale.reshape(()).float().clamp_min(0)
        return torch.stack([loc, scale, torch.log1p(scale), rng, torch.log1p(rng), mx])

    def _effective_patches(self, p: int, allow_upsample: bool = True) -> int:
        """Number of patches a sample actually feeds into chronos / the Q-former after short-series upsampling (keeps cost estimates consistent with encode).

        Very short single-chunk histories (1 <= P < short_min_patches) are upsampled to
        short_min_patches patches; everything else is returned unchanged (a full-window chunk has
        P = W/patch >> threshold and never triggers). ``allow_upsample=False`` (forecasting branch)
        returns the original P -- forecasting never upsamples."""
        if allow_upsample and self.short_min_patches and 1 <= int(p) < self.short_min_patches:
            return self.short_min_patches
        return int(p)

    def soft_token_count(self, p: int, c: int = 1, allow_upsample: bool = True) -> int:
        """Per-sample soft-prompt token count = sliding-window Q-former tokens (from the upsampled patch count) + statistics tokens (1 per channel).

        Used by the cost estimates of dynamic batching / length bucketing (matches what
        ``_encode_history_to_soft_prompt`` actually produces per sample). ``allow_upsample=False``
        is for the forecasting branch: the upsampled patch count is not used."""
        if self.chronos is None:
            return 0  # llm_only: no soft prompt; the series cost is already counted in the text tokens (ts_as_text rendering)
        n = self.history_qformer.num_tokens_for_length(self._effective_patches(p, allow_upsample), int(c))
        if self.history_stats_token:
            n += int(c)
        return n

    def _encode_history_to_soft_prompt(
        self,
        context: torch.Tensor,
        true_lengths: torch.Tensor,
        n_channels: torch.Tensor | None = None,
        chunk_pos: list | None = None,
        allow_upsample: bool = True,
    ) -> list[torch.Tensor]:
        """Encode each (possibly multi-channel) history into a variable-length soft prompt (one per sample).

        ``context`` is the collator-folded ``(ΣC, L)`` tensor (all channels of all samples,
        sample-major row order); ``n_channels (B,)`` gives the channel count per sample (None means
        single-channel, one sample per row).

        chronos2 encodes long histories with **non-overlapping sliding windows of maximal size
        (= context_length)**: each channel of each sample is cut into non-overlapping chunks along
        time; **the C channels of the same (sample, chunk) share one ``group_id``**, so chronos'
        ``GroupSelfAttention`` lets the channels interact within the group (different chunks are
        different groups and independent, continuing the single-channel per-chunk independence).
        **Per-channel global normalisation**: every channel uses the (loc, scale) of its own whole
        history. All chunk rows are encoded, then per sample the valid patches of all channels and
        chunks are gathered into ``(C, P_b, d)`` and passed through the 2-D sliding-window Q-former
        to obtain ``(N_b, hidden)`` (the token count depends only on the temporal patch count P_b,
        not on the channel count C).

        Single-channel input (``n_channels`` all 1 or None) reduces to: every chunk is its own
        group with per-sample global normalisation, i.e. the same logical structure as the
        single-channel case.

        ``chunk_pos``: per sample ``(n_b, 2) int64=[window start, window span]`` (in points; only
        for samples where window sampling triggered) or None (not triggered / not passed for the
        batch). When present, ``_patch_positions`` converts it (pos0=start/patch,
        dpos=span/chronos_window) and expands it into per-patch true positions for the sinusoidal
        PE of the Q-former's global branch; None uses ordinal PE, bit-identical to the default.
        """
        SC, L = context.shape
        if SC == 0:
            return []  # empty batch (e.g. evaluation shard boundary): max() over an empty sequence below would raise ValueError
        if self.chronos is None:
            # llm_only: the soft prompt is always empty (the series is already textualised into
            # input_ids by the dataset); return (0, llm_hidden) per sample => splicing is a no-op.
            B = SC if n_channels is None else len(n_channels)
            return [context.new_zeros(0, self.llm_hidden) for _ in range(B)]
        W = self.chronos_window
        patch = self.input_patch_size
        n_list = [1] * SC if n_channels is None else [int(x) for x in n_channels]
        B = len(n_list)
        row_starts, acc = [], 0
        for nc in n_list:
            row_starts.append(acc)
            acc += nc

        # 1) Cut chunks per sample and channel; the C channels of one (sample, chunk) share a group;
        #    per-channel global loc/scale.
        chunk_segs: list[torch.Tensor] = []
        chunk_valid_len: list[int] = []
        enc_group_ids: list[int] = []
        loc_rows: list[torch.Tensor] = []
        scale_rows: list[torch.Tensor] = []
        sample_rowidx: list[list[list[int]]] = []  # [b][c] -> encode row indices in chunk time order
        stats_list: list[torch.Tensor | None] = []  # statistics token: per-sample (C_b, n_feat) raw statistics; None = disabled
        ridx, g = 0, 0
        for b in range(B):
            C_b, r0 = n_list[b], row_starts[b]
            tl = max(1, min(int(true_lengths[r0].item()), L))   # all channels of a sample share a length; take the first channel
            cp_b0 = chunk_pos[b] if chunk_pos is not None else None
            reals, locs, scales, feats = [], [], [], []
            for c in range(C_b):
                real_c = context[r0 + c, L - tl:]
                _, (lc, sc) = self.chronos.instance_norm(real_c.unsqueeze(0))
                reals.append(real_c); locs.append(lc); scales.append(sc)
                if self.history_stats_token:
                    feats.append(self._series_stats(real_c, lc, sc))   # raw-series statistics (before upsampling)
            stats_list.append(torch.stack(feats, dim=0) if feats else None)
            # Short-series upsampling: very short single-chunk series are upsampled to
            # short_min_patches patches so the chronos2 encoder / sliding-window Q-former have real
            # temporal structure to attend to (the patch size is too coarse for 16 points, which
            # originally yields a single patch). Applies only to samples that were **not window
            # sampled** (cp_b0=None), with tl<=W (single chunk) and too few patches -- window-sampled
            # samples are extremely long anyway, never trigger, and their chunk_pos length guard
            # must not be invalidated by a length change. (loc, scale, statistics) are still
            # computed on the raw series. allow_upsample=False (forecasting branch) always skips:
            # the forecasting soft prompt must reflect the **true history** and never see
            # interpolated synthetic points (the quantile forecast itself uses the raw context, but
            # interpolation would pollute the LLM reasoning -> feedback path).
            tl_eff = tl
            if (allow_upsample and self.short_min_patches and cp_b0 is None and 2 <= tl <= W
                    and math.ceil(tl / patch) < self.short_min_patches):
                tl_eff = self.short_min_patches * patch
                reals = [F.interpolate(r.float().reshape(1, 1, -1), size=tl_eff,
                                       mode="linear", align_corners=False).reshape(-1).to(r.dtype)
                         for r in reals]
            n_ch_b = max(1, math.ceil(tl_eff / W))
            rem = tl_eff - (n_ch_b - 1) * W
            starts = [0] + [rem + j * W for j in range(n_ch_b - 1)]
            lens = [rem] + [W] * (n_ch_b - 1)
            rowidx_bc = [[] for _ in range(C_b)]
            for s, ln in zip(starts, lens):
                for c in range(C_b):  # the C channels of one chunk are contiguous rows in the same group g
                    chunk_segs.append(reals[c][s:s + ln])
                    chunk_valid_len.append(ln)
                    enc_group_ids.append(g)
                    loc_rows.append(locs[c]); scale_rows.append(scales[c])
                    rowidx_bc[c].append(ridx)
                    ridx += 1
                g += 1
            sample_rowidx.append(rowidx_bc)

        # Common chunk length: the maximum over all chunks of the valid length rounded up to a
        # multiple of the patch size, capped at W.
        chunk_pad_len = min(W, max(int(math.ceil(v / patch) * patch) for v in chunk_valid_len))
        chunks = context.new_full((len(chunk_segs), chunk_pad_len), float("nan"))
        for i, seg in enumerate(chunk_segs):
            chunks[i, chunk_pad_len - seg.shape[0]:] = seg.to(chunks.dtype)
        loc_per = torch.cat([l.reshape(1, 1) for l in loc_rows], dim=0)
        scale_per = torch.cat([s.reshape(1, 1) for s in scale_rows], dim=0)
        gids = torch.tensor(enc_group_ids, dtype=torch.long, device=context.device)

        # 2) Bucketed encode by group (per-channel global normalisation; group_ids let the channels
        #    of one (sample, chunk) interact within the group). GroupSelfAttention's mask/scores are
        #    (T, R, R) and grow **quadratically** with the total row count R, while groups never
        #    interact (block-diagonal mask) -- packing whole groups into buckets of
        #    <= encode_max_rows rows makes the cost linear in R and is numerically equivalent to a
        #    single large-batch encode (the pad length is global, and each row's computation only
        #    depends on its own group).
        total = len(chunk_segs)
        mr = self.encode_max_rows
        grp_starts = [i for i in range(total) if i == 0 or enc_group_ids[i] != enc_group_ids[i - 1]]
        buckets, cur = [], 0
        if mr:
            # Greedy bucketing at group boundaries: a bucket closes once it reaches mr rows
            # (overshoot at most C-1 rows; a single group larger than mr gets its own bucket).
            for gs in grp_starts[1:]:
                if gs - cur >= mr:
                    buckets.append((cur, gs))
                    cur = gs
        buckets.append((cur, total))
        row_patches: list[torch.Tensor] = [None] * total  # valid patches of each row (vp_i, d_model)
        # Backward needs the forward activations of **all** buckets => without checkpointing the
        # activation memory grows linearly with the total patch count (a 210k-patch sample needs
        # 20GB+ for the encode activations alone). With per-bucket checkpointing the forward pass
        # only stores the bucket inputs and backward recomputes one bucket at a time, so the peak
        # is independent of the sample length; the price is running the encode forward twice.
        use_ckpt = self._encode_grad_ckpt and self.training and torch.is_grad_enabled()
        for s, e in buckets:
            if use_ckpt:
                patches = torch.utils.checkpoint.checkpoint(
                    self._encode_bucket, chunks[s:e], gids[s:e],
                    loc_per[s:e], scale_per[s:e], use_reentrant=False,
                )
            else:
                patches = self._encode_bucket(
                    chunks[s:e], gids[s:e], loc_per[s:e], scale_per[s:e]
                )
            ncp = patches.shape[1]   # (e-s, ncp, d_model)
            for i in range(s, e):
                vp = min(ncp, max(1, math.ceil(chunk_valid_len[i] / patch)))
                row_patches[i] = patches[i - s, ncp - vp:]

        # 3) Per sample, gather the valid patches of all channels and chunks into (C, P_b, d) -> 2-D
        #    sliding-window Q-former. The qformer's global branch cross-attends over the whole C×P
        #    and the window branch flattens C×W, so its activations also grow linearly with C×P --
        #    checkpointed per sample under the same switch as the encode.
        soft_list: list[torch.Tensor] = []
        for b in range(B):
            seg_per_channel = []
            for c in range(n_list[b]):
                seg = [row_patches[rid] for rid in sample_rowidx[b][c]]
                seg_per_channel.append(torch.cat(seg, dim=0))     # (P_b, d)
            h_valid = torch.stack(seg_per_channel, dim=0).unsqueeze(0)  # (1, C_b, P_b, d)
            cp_b = chunk_pos[b] if chunk_pos is not None else None
            if cp_b is not None:
                # Total-length guard: window sampling produces a compact array whose length is
                # always n×W. A coincidentally matching row count with a mismatched total length
                # (max_context cap truncation, sample_window != chronos_window, ...) would
                # silently misalign positions.
                tl_b = max(1, min(int(true_lengths[row_starts[b]].item()), L))
                if tl_b != cp_b.shape[0] * W:
                    raise ValueError(
                        f"compact sequence length {tl_b} does not equal chunk_pos rows {cp_b.shape[0]} × chronos_window "
                        f"{W} -- check that sample_window==chronos_window and that max_context did not truncate "
                        "the compact sequence")
            pos = self._patch_positions(
                cp_b, [row_patches[rid].shape[0] for rid in sample_rowidx[b][0]],
                W, h_valid.device)
            if use_ckpt:
                soft = torch.utils.checkpoint.checkpoint(
                    self.history_qformer, h_valid, pos, use_reentrant=False
                )
            else:
                soft = self.history_qformer(h_valid, pos)
            soft_b = soft[0]                                            # (N_b, hidden)
            # Statistics token: prepend the window-scale statistics tokens (one per channel) to the
            # soft prompt -- the scale/range information divided away by instance_norm is handed
            # back to the LLM explicitly. The projection is trained from scratch (qformer lr
            # group); for C=1 this is a single token.
            if stats_list[b] is not None and self.history_stats_proj is not None:
                pdt = next(self.history_stats_proj.parameters()).dtype
                st = self.history_stats_proj(stats_list[b].to(pdt))     # (C_b, hidden)
                soft_b = torch.cat([st.to(soft_b.dtype), soft_b], dim=0)  # (C_b + N_b, hidden)
            soft_list.append(soft_b)
        return soft_list

    def _encode_bucket(
        self,
        chunks: torch.Tensor,
        gids: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single bucket and return the context patch representations (R, ncp, d_model).

        Pure tensors in and out (the caller recovers ncp from the shape), so it can be wrapped by
        torch.utils.checkpoint.
        """
        enc_out, _, _, ncp = self.chronos.encode(
            context=chunks, num_output_patches=1,
            group_ids=gids, loc_scale=(loc, scale),
            stop_at_layer=getattr(self, "history_encode_layer", None),
        )
        return enc_out[0][:, :ncp]

    @staticmethod
    def _patch_positions(cp, vps: list[int], window: int, device, patch: int = 16):
        """chunk_pos ``(n,2) int64=[start, span]`` (in points) + valid patch count per chunk -> per-patch
        true positions ``(ΣP,)`` (in patch units: pos0=start/patch, dpos=span/window).

        The int64 payload is deliberate: under DeepSpeed bf16 the HF Trainer ``_prepare_input``
        casts floating-point tensors to bf16 (at a start of ~15000 the ulp is 64, which destroys
        the high-frequency PE), whereas integer tensors pass through untouched; the floating-point
        conversion is done here in float32 (exact for starts < 2^24).
        None passes through (the Q-former uses ordinal PE). A row count that differs from the
        actual chunk count raises loudly -- otherwise positions would silently land on the wrong chunk.
        """
        if cp is None:
            return None
        if len(vps) != cp.shape[0]:
            raise ValueError(
                f"chunk_pos has {cp.shape[0]} rows != actual chunk count {len(vps)} -- window sampling in the "
                "data layer is misaligned with the model's chunking (the dataset's sample_window must equal "
                "the model's chronos_window)")
        cp = cp.to(device=device, dtype=torch.float32)   # int64 payload -> float32 conversion (exact for starts < 2^24)
        segs = [
            cp[ci, 0] / patch + (cp[ci, 1] / window) * torch.arange(vp, device=device, dtype=torch.float32)
            for ci, vp in enumerate(vps)
        ]
        return torch.cat(segs)

    def _splice_soft_prompt(self, embeds, labels, attn, input_ids, soft_list, side: str = "right"):
        """Insert each sample's (variable-length) soft prompt right after its <ts_start>, then pad to the batch maximum.

        soft_list is a list of length B with elements (N_b, hidden); N_b may differ per sample, so
        the total lengths differ after insertion.
        ``side``:
        - ``"right"`` (training): right padding (embeds=0, attn=0, labels=-100 at pad positions).
          Under teacher forcing the pad positions have label=-100 and no autoregressive dependency,
          so right padding is harmless.
        - ``"left"`` (batched generation): left padding, and **first strip the right padding the
          collator appended to the text** (keep only the valid prefix where attn>0) so that the real
          content sits contiguously at the **right end** of the sequence -- a decoder-only model
          continues from the last position's logits, and this is required for B>1 batched
          generation of unequal lengths not to degrade.
        """
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        B, _, _ = embeds.shape
        out_e, out_l, out_a = [], [], []
        for b in range(B):
            soft_b = soft_list[b].to(embeds.dtype)
            N = soft_b.shape[0]
            hit = (input_ids[b] == self.ts_start_id).nonzero(as_tuple=False)
            if hit.numel() == 0:
                raise ValueError("input_ids is missing the <ts_start> placeholder token")
            pos = int(hit[0, 0]) + 1  # insert right after ts_start
            # Left padding is the generation path: drop the right padding at the text tail (under
            # right padding the valid positions form the contiguous prefix [0:nval]).
            end = int(attn[b].sum().item()) if side == "left" else attn.shape[1]
            out_e.append(torch.cat([embeds[b, :pos], soft_b, embeds[b, pos:end]], dim=0))
            out_a.append(torch.cat([attn[b, :pos], attn.new_ones(N), attn[b, pos:end]], dim=0))
            if labels is not None:
                out_l.append(
                    torch.cat([labels[b, :pos], labels.new_full((N,), -100), labels[b, pos:end]], dim=0)
                )
        maxlen = max(e.shape[0] for e in out_e)
        H = embeds.shape[-1]
        new_e = embeds.new_zeros(B, maxlen, H)
        new_a = attn.new_zeros(B, maxlen)
        new_l = labels.new_full((B, maxlen), -100) if labels is not None else None
        for b in range(B):
            ln = out_e[b].shape[0]
            sl = slice(maxlen - ln, maxlen) if side == "left" else slice(0, ln)
            new_e[b, sl] = out_e[b]
            new_a[b, sl] = out_a[b]
            if new_l is not None:
                new_l[b, sl] = out_l[b]
        return new_e, new_l, new_a

    def _run_llm(self, input_ids, attention_mask, labels, soft_list):
        """LLM forward + chunked CE over the supervised positions only.

        We do not use the internal loss of ``self.llm(labels=...)`` -- it materialises
        ``(Σtoken, vocab≈248k)`` logits for **every** position (including prompt/pad positions with
        label=-100), and the fp32 upcast of the CE plus its gradient grow linearly with Σtoken (at a
        token budget of 30000 the logits gradient alone is 28GiB, the observed OOM culprit).
        Instead: run the backbone (without lm_head) to get last_hidden_state -> ``_chunked_ce``
        passes only the supervised positions through lm_head in chunks; numerically equivalent to
        the HF internal CE (mean over valid positions).
        """
        from types import SimpleNamespace

        embeds = self.llm.get_input_embeddings()(input_ids)
        new_e, new_l, new_a = self._splice_soft_prompt(embeds, labels, attention_mask, input_ids, soft_list)
        decoder = getattr(self.llm, "model", None)
        if decoder is None:
            decoder = self.llm.get_decoder()
        hidden = decoder(
            inputs_embeds=new_e,
            attention_mask=new_a,
            use_cache=False,
        ).last_hidden_state
        loss = self._chunked_ce(hidden, new_l) if new_l is not None else None
        # Legacy interface shape: callers only use .loss and .hidden_states[-1] (= last_hidden_state
        # after the final norm, identical to the last element under output_hidden_states=True).
        return SimpleNamespace(loss=loss, hidden_states=(hidden,)), new_a

    def _chunked_ce(self, hidden: torch.Tensor, labels: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """Chunked CE over the supervised positions, numerically equivalent to the internal loss of ``self.llm(labels=...)``.

        After the shift, only positions with ``label != -100`` are gathered (answers in the
        understanding / forecasting data are far shorter than the prompt, so supervised positions
        are a small fraction), then passed through lm_head + fp32 CE in blocks of ``chunk``
        (sum accumulated / total valid positions). In training mode each block is checkpointed:
        backward recomputes one block of logits at a time, so the peak is one block × vocab,
        independent of the total number of supervised tokens in the batch. Returns 0 when every
        label is -100 (keeps the graph connected, consistent with the data-hygiene convention
        rather than HF's nan).
        """
        h = hidden[:, :-1, :]
        y = labels[:, 1:]
        mask = y != -100
        n = int(mask.sum())
        if n == 0:
            return hidden.sum() * 0.0
        hs = h[mask]   # (N, H)
        ys = y[mask]   # (N,)
        lm_head = self.llm.get_output_embeddings()

        def _ce_sum(hc, yc):
            return nn.functional.cross_entropy(
                lm_head(hc).float(), yc, reduction="sum"
            )

        use_ckpt = self.training and torch.is_grad_enabled()
        total = hidden.new_zeros((), dtype=torch.float32)
        for s in range(0, n, chunk):
            hc, yc = hs[s:s + chunk], ys[s:s + chunk]
            if use_ckpt:
                total = total + torch.utils.checkpoint.checkpoint(
                    _ce_sum, hc, yc, use_reentrant=False)
            else:
                total = total + _ce_sum(hc, yc)
        return total / n

    # ---------------------------------------------------------------- forward
    def forward_understanding(self, batch: dict) -> dict:
        context = batch["context"]
        true_lengths = batch["true_lengths"]
        soft = self._encode_history_to_soft_prompt(
            context, true_lengths, batch.get("n_channels"), batch.get("chunk_pos"))
        out, _ = self._run_llm(batch["input_ids"], batch["attention_mask"], batch["labels"], soft)
        ce = out.loss
        return {"loss": self.config.analysis_loss_weight * ce, "text_loss": ce.detach()}

    def _forecast_layout(self, batch: dict, context):
        """Derive the multivariate forecasting layout from the batch: group_ids, the target-row mask,
        and the known-future covariates scattered into a (ΣC, fl) tensor (NaN on non-covariate rows).
        Single-channel: target_idx all True, covariates None.

        Row order within a sample's channel block = [targets(n_targets), known-future cov(n_fut), past-only cov].
        """
        SC = context.shape[0]
        group_ids = batch.get("group_ids")
        if group_ids is None:
            group_ids = torch.arange(SC, device=context.device)
        group_ids = group_ids.to(context.device)

        n_ch = batch["n_channels"].tolist() if "n_channels" in batch else [1] * SC
        B = len(n_ch)
        n_tg = batch["n_targets"].tolist() if "n_targets" in batch else [1] * B
        n_fut_t = batch.get("n_future_covariates")
        n_fut = n_fut_t.tolist() if n_fut_t is not None else [0] * B

        target_idx = torch.zeros(SC, dtype=torch.bool, device=context.device)
        cov_positions = []
        row = 0
        for b in range(B):
            if n_tg[b] + n_fut[b] > n_ch[b]:
                # Without this check, targets/covariates would silently spill into the rows of the
                # neighbouring sample (and use the wrong row's normalisation statistics).
                raise ValueError(
                    f"sample {b}: n_targets({n_tg[b]}) + n_future_covariates({n_fut[b]}) exceeds the channel count "
                    f"{n_ch[b]} -- the history channel-row convention is [targets, known-future cov, past-only cov]; "
                    "every covariate must occupy its own channel row")
            target_idx[row:row + n_tg[b]] = True
            cov_positions.extend(row + n_tg[b] + j for j in range(n_fut[b]))
            row += n_ch[b]

        future_covariates = None
        fcov = batch.get("future_covariates")
        if fcov is not None and cov_positions:
            future_covariates = context.new_full((SC, fcov.shape[-1]), float("nan"))
            for i, pos in enumerate(cov_positions):
                future_covariates[pos] = fcov[i].to(context.dtype)
        return group_ids, target_idx, future_covariates

    def _num_output_patches(self, horizon: int) -> int:
        """horizon -> number of chronos output patches; raise explicitly when the horizon exceeds the
        predictable maximum (instead of surfacing as an assert / broadcast failure deep inside the
        upstream ``_compute_loss``)."""
        ops = int(self.chronos.chronos_config.output_patch_size)
        max_h = self.config.chronos_max_output_patches * ops
        if horizon > max_h:
            raise ValueError(
                f"forecast horizon={horizon} exceeds the chronos output limit "
                f"{max_h} (= chronos_max_output_patches({self.config.chronos_max_output_patches})"
                f" × output_patch_size({ops})); increase config.chronos_max_output_patches or shorten future"
            )
        return math.ceil(horizon / ops)

    def _conclusion_feedback_mask(self, input_ids, soft_list, new_attn):
        """Ablation 1: src_key_padding_mask for feedback_qformer that only admits the conclusion span
        (after </think> up to the valid end); soft prompt / prompt text / reasoning are all masked.

        After splicing, the conclusion position = original </think> position + this sample's soft
        token count (the soft prompt is inserted after <ts_start> and before </think>; training uses
        right padding => the valid positions are the contiguous prefix [0:new_attn.sum())).
        True = masked. If </think> is missing or the conclusion span is empty, fall back to "mask
        only the padding" (an all-masked row would make attention all -inf -> NaN).
        """
        B, L = new_attn.shape
        mask = torch.ones(B, L, dtype=torch.bool, device=new_attn.device)
        for b in range(B):
            n_soft = int(soft_list[b].shape[0])
            hit = (input_ids[b] == self.think_end_id).nonzero(as_tuple=False)
            c_end = int(new_attn[b].sum().item())
            if hit.numel() == 0:
                mask[b] = new_attn[b] <= 0
                continue
            c_start = int(hit[0, 0]) + 1 + n_soft  # first token after </think> (including the soft-prompt offset)
            if c_end > c_start:
                mask[b, c_start:c_end] = False
            else:
                mask[b] = new_attn[b] <= 0
        return mask

    def forward_forecast(self, batch: dict) -> dict:
        context = batch["context"]
        context_mask = batch.get("context_mask")
        cfg = self.config

        # LLM-only baseline: no chronos / feedback; the forecasting branch is just the text CE
        # (supervised span = <think>reasoning</think> + conclusion + full-precision future values
        # as text, rendered by the dataset's ts_as_text).
        if getattr(cfg, "llm_only", False):
            soft = self._encode_history_to_soft_prompt(
                context, batch["true_lengths"], batch.get("n_channels"))  # empty soft prompt per sample
            out, _ = self._run_llm(batch["input_ids"], batch["attention_mask"], batch["labels"], soft)
            ce = out.loss
            return {"loss": cfg.text_loss_weight * ce, "text_loss": ce.detach()}

        # Ablation 2 (chronos_only): skip the LLM / soft prompt / feedback entirely and only
        # fine-tune chronos2 (cross_states=None) with the quantile/ROI losses. Pure lower-bound
        # control for the "textual feedback gain".
        if getattr(cfg, "chronos_only", False):
            group_ids, target_idx, future_covariates = self._forecast_layout(batch, context)
            future = batch["future"]
            nop = self._num_output_patches(future.shape[-1])
            res = self.chronos.forecast_losses(
                context=context, future_target=future, num_output_patches=nop,
                context_mask=context_mask, group_ids=group_ids,
                cross_states=None, cross_states_mask=None,
                roi_mask=batch.get("roi_mask"), future_covariates=future_covariates,
                target_idx=target_idx)
            pred_loss, roi_loss = res["pred_loss"], res["roi_loss"]
            total = cfg.pred_loss_weight * pred_loss
            if roi_loss is not None:
                total = total + cfg.roi_loss_weight * roi_loss
            return {"loss": total, "text_loss": context.new_zeros(()),
                    "pred_loss": pred_loss.detach(),
                    "roi_loss": (roi_loss.detach() if roi_loss is not None else None)}

        true_lengths = batch["true_lengths"]
        soft = self._encode_history_to_soft_prompt(
            context, true_lengths, batch.get("n_channels"), batch.get("chunk_pos"),
            allow_upsample=False)  # no upsampling in the forecasting branch: the soft prompt must reflect the true history
        out, new_attn = self._run_llm(batch["input_ids"], batch["attention_mask"], batch["labels"], soft)
        text_loss = out.loss

        # In-training scheduled sampling: with probability 1-ss_teacher_ratio, the feedback
        # condition of this batch switches to **self-generated** reasoning+conclusion (LLM
        # generation and hidden extraction fully under no_grad -- SS only teaches chronos+feedback
        # to adapt to the generated distribution; the text CE always comes from the teacher forward
        # above, so the LLM keeps learning to generate). ss_teacher_ratio is linearly annealed by a
        # trainer callback over training (e.g. 1.0->0.7); the default 1.0, or a batch without
        # ss_prefix, takes the original path entirely.
        p_teacher = float(getattr(self, "ss_teacher_ratio", 1.0))
        use_gen = (self.training and p_teacher < 1.0
                   and batch.get("ss_prefix_ids") is not None)
        if use_gen:
            # The trigger decision prefers the callback's deterministic per-global_step draw
            # (ss_force_gen, synchronised across ranks -- otherwise under ZeRO-2 one rank would
            # generate while another idles); without a callback fall back to a local draw
            # (single-process tests). Lazy draw: the non-triggering path consumes no RNG, so the
            # random stream is bit-identical to the original path.
            force = getattr(self, "ss_force_gen", None)
            use_gen = force if force is not None else (torch.rand(()).item() >= p_teacher)
        if use_gen:
            pad_id = self.tokenizer.pad_token_id
            # Temporarily switch to eval for the generation segment: in train mode gradient
            # checkpointing forces use_cache=False (full-length forward per token without KV
            # cache -- a hundred times slower, and SDPA's 4-D bias hits "last dimension must be
            # contiguous"); eval takes exactly the same cache path as mid-training evaluation /
            # generate_forecast, so the hidden distribution matches inference (which is precisely
            # what SS is meant to align with).
            was_training = self.llm.training
            self.llm.eval()
            # The hidden states are only needed for feedback -- bypass lm_head: the CausalLM
            # forward would materialise (B, L'', vocab≈248k) logits, 70+GiB at a dynamic training
            # bs of several hundred (observed OOM culprit, same lesson as _run_llm).
            decoder = getattr(self.llm, "model", None)
            if decoder is None:
                decoder = self.llm.get_decoder()
            # Generate in micro-batches of size chunk: the dynamic training bs can reach several
            # hundred, so KV cache / activations are hard-capped per chunk; feedback_qformer is
            # independent along the batch dimension => running it per chunk and concatenating is
            # bit-identical to the whole batch (the qformer forward carries grad -- the pred loss
            # of an SS batch trains feedback_qformer).
            CH = int(getattr(self, "ss_gen_chunk", 32))
            B_ss = batch["ss_prefix_ids"].shape[0]
            c_parts = []
            try:
                for s in range(0, B_ss, CH):
                    e = min(s + CH, B_ss)
                    with torch.no_grad():
                        embeds0 = self.llm.get_input_embeddings()(batch["ss_prefix_ids"][s:e])
                        pe, _, pa = self._splice_soft_prompt(
                            embeds0, None, batch["ss_prefix_mask"][s:e],
                            batch["ss_prefix_ids"][s:e], soft[s:e], side="left")
                        gen = self.llm.generate(
                            inputs_embeds=pe, attention_mask=pa,
                            max_new_tokens=int(getattr(self, "ss_max_new_tokens", 256)),
                            do_sample=False, return_dict_in_generate=True, use_cache=True,
                            **self._gen_defaults({}))
                        seq = gen.sequences
                        gen_e = self.llm.get_input_embeddings()(seq)
                        full_e = torch.cat([pe, gen_e], dim=1)
                        gen_a = (seq != pad_id).long() if pad_id is not None else pa.new_ones(seq.shape)
                        full_a = torch.cat([pa, gen_a], dim=1)
                        hidden_g = decoder(inputs_embeds=full_e, attention_mask=full_a,
                                           use_cache=False).last_hidden_state
                        hidden_g = self._fb_hidden(hidden_g)   # B2: may take an intermediate layer
                    c_parts.append(self.feedback_qformer(
                        hidden_g.to(self.feedback_qformer.input_proj.weight.dtype),
                        src_key_padding_mask=(full_a <= 0)))
            finally:
                if was_training:
                    self.llm.train()
            c_llm = torch.cat(c_parts, dim=0)  # (B, M, d)
        else:
            hidden = self._fb_hidden(out.hidden_states[-1])  # (B, L', llm_hidden); B2 may take an intermediate layer
            # Which span of the hidden states is fed back: by default the whole sequence (only
            # padding masked); ablation 1 admits only the conclusion span (ablation 1 is not
            # combined with SS: an SS batch has a generated-sequence layout and uses the pad mask).
            if getattr(cfg, "feedback_scope", "all") == "conclusion":
                fb_mask = self._conclusion_feedback_mask(batch["input_ids"], soft, new_attn)
            else:
                fb_mask = new_attn <= 0  # True = pad
            c_llm = self.feedback_qformer(hidden.to(self.feedback_qformer.input_proj.weight.dtype),
                                          src_key_padding_mask=fb_mask)  # (B, M, d)
        c_llm = c_llm.to(self.chronos.dtype)

        group_ids, target_idx, future_covariates = self._forecast_layout(batch, context)
        # Broadcast cross_states by group to every channel row (ΣC, M, d): each row uses the
        # compressed LLM representation of the sample it belongs to.
        cross_states = c_llm[group_ids]
        cross_mask = torch.ones(cross_states.shape[0], cross_states.shape[1], device=cross_states.device)

        future = batch["future"]  # (Σn_targets, fl) in the original scale (NaN padding)
        fl = future.shape[-1]
        nop = self._num_output_patches(fl)
        res = self.chronos.forecast_losses(
            context=context,
            future_target=future,
            num_output_patches=nop,
            context_mask=context_mask,
            group_ids=group_ids,
            cross_states=cross_states,
            cross_states_mask=cross_mask,
            roi_mask=batch.get("roi_mask"),
            future_covariates=future_covariates,
            target_idx=target_idx,
        )
        pred_loss = res["pred_loss"]
        roi_loss = res["roi_loss"]
        cfg = self.config
        total = cfg.text_loss_weight * text_loss + cfg.pred_loss_weight * pred_loss
        if roi_loss is not None:
            total = total + cfg.roi_loss_weight * roi_loss
        return {
            "loss": total,
            "text_loss": text_loss.detach(),
            "pred_loss": pred_loss.detach(),
            "roi_loss": (roi_loss.detach() if roi_loss is not None else None),
        }

    def forward(self, batch: dict) -> dict:
        if batch.get("branch") == "forecast":
            return self.forward_forecast(batch)
        return self.forward_understanding(batch)

    # ---------------------------------------------------------------- generate
    def _gen_defaults(self, gen_kwargs: dict) -> dict:
        """eos/pad fallbacks for generate (not overridden when the caller passes them explicitly).

        The Qwen3.5 checkpoint ships no generation_config.json, so transformers builds
        eos=<|endoftext|>(248044) from the config; the training supervision, however, ends with
        <|im_end|>(248046, = tokenizer.eos_token) as per the chat template -- without an explicit
        eos, generation never stops at the learned answer end, every sample runs to max_new_tokens
        and rambles past the answer (polluting the understanding metrics and the forecast feedback
        hidden states)."""
        if self.tokenizer.eos_token_id is not None:
            gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        return gen_kwargs

    @torch.no_grad()
    def generate_understanding(self, batch: dict, max_new_tokens: int = 256, **gen_kwargs):
        """Inject the soft prompt and generate the text answer autoregressively; returns list[str]."""
        was_training = self.training
        self.eval()
        try:
            soft = self._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch.get("n_channels"),
                batch.get("chunk_pos"))
            embeds = self.llm.get_input_embeddings()(batch["input_ids"])
            new_e, _, new_a = self._splice_soft_prompt(
                embeds, None, batch["attention_mask"], batch["input_ids"], soft, side="left"
            )
            gen = self.llm.generate(
                inputs_embeds=new_e, attention_mask=new_a,
                max_new_tokens=max_new_tokens, use_cache=True, **self._gen_defaults(gen_kwargs),
            )
            return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        finally:
            if was_training:
                self.train()  # restore the mode: mid-training generation must not leave the model in eval (dropout disabled)

    def _forecast_chronos_only(self, batch: dict, horizon: int):
        """Ablation 2 evaluation: no LLM / feedback, chronos2 (cross_states=None) directly predicts the target channels."""
        context = batch["context"]
        group_ids, target_idx, future_covariates = self._forecast_layout(batch, context)
        nop = self._num_output_patches(horizon)
        qp = self.chronos(
            context, num_output_patches=nop, group_ids=group_ids,
            context_mask=batch.get("context_mask"), future_covariates=future_covariates,
            cross_states=None, cross_states_mask=None,
        ).quantile_preds
        return qp[target_idx]

    def _conclusion_mask_generate(self, seq, prefix_len, full_a, pad_id):
        """Ablation 1, autoregressive evaluation: in the [left-padded prefix (length prefix_len) + generated seq]
        layout, admit only the conclusion span after </think> within the generated segment.
        True = masked; if missing / empty, fall back to masking only the padding."""
        B = seq.shape[0]
        mask = torch.ones(B, full_a.shape[1], dtype=torch.bool, device=full_a.device)
        for b in range(B):
            hit = (seq[b] == self.think_end_id).nonzero(as_tuple=False)
            gen_valid = int((seq[b] != pad_id).sum()) if pad_id is not None else seq.shape[1]
            c_end = prefix_len + gen_valid
            if hit.numel() == 0 or c_end <= prefix_len + int(hit[0, 0]) + 1:
                mask[b] = full_a[b] <= 0
                continue
            mask[b, prefix_len + int(hit[0, 0]) + 1:c_end] = False
        return mask

    def _forecast_from_hidden(self, batch: dict, hidden, attn, horizon: int, fb_mask=None):
        """Shared tail: LLM hidden -> feedback Q-former -> cross_states fed back into chronos -> quantile forecast.
        A non-None fb_mask replaces the default pad mask (ablation 1 admits only the conclusion span)."""
        hid_pad = fb_mask if fb_mask is not None else (attn <= 0)
        c_llm = self.feedback_qformer(
            hidden.to(self.feedback_qformer.input_proj.weight.dtype), src_key_padding_mask=hid_pad
        ).to(self.chronos.dtype)  # (B, M, d)

        context = batch["context"]
        group_ids, target_idx, future_covariates = self._forecast_layout(batch, context)
        cross_states = c_llm[group_ids]  # broadcast by group to (ΣC, M, d)
        cross_mask = torch.ones(cross_states.shape[0], cross_states.shape[1], device=cross_states.device)

        nop = self._num_output_patches(horizon)
        # context must be passed positionally: after the modules_to_save wrapping, wrapper.forward(x, ...)
        # requires its first positional argument.
        qp = self.chronos(
            context, num_output_patches=nop, group_ids=group_ids,
            context_mask=batch.get("context_mask"),  # passed through symmetrically with the training path (forecast_losses)
            future_covariates=future_covariates,
            cross_states=cross_states, cross_states_mask=cross_mask,
        ).quantile_preds  # (ΣC, Q, H), de-normalised
        return qp[target_idx]  # only the target channels' forecasts (Σn_targets, Q, H)

    @torch.no_grad()
    def _generate_forecast_llm_only(self, batch: dict, horizon: int, max_new_tokens: int = 256,
                                    **gen_kwargs):
        """Forecast generation for the LLM-only baseline: generate text (reasoning -> conclusion ->
        numeric array), then **parse the values from the text**; the point forecast is replicated
        across the 21 quantiles (the same pre-declared protocol as ChatTime / TimeReasoner / UniTS;
        CRPS degenerates to a weighted MAE). Parse failures are left as NaN and counted honestly as
        failures -- no fabricated fill-in values.

        Only single-target samples are supported (history / future in this forecasting corpus are
        single-channel; multiple targets would need a per-row text protocol, out of scope for
        LLM-only). Returns the same structure as the FULL path: {"text": list[str],
        "quantile_preds": (B, Q=21, horizon)}.
        """
        from chronos_llm.data.ts_text import CHRONOS2_QUANTILE_LEVELS, parse_forecast_values

        n_tg = batch.get("n_targets")
        if n_tg is not None and any(int(x) != 1 for x in n_tg):
            raise NotImplementedError("llm_only forecasting only supports single-target samples")
        was_training = self.training
        self.eval()
        try:
            soft = self._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch.get("n_channels"))  # empty soft prompt per sample
            embeds = self.llm.get_input_embeddings()(batch["input_ids"])
            new_e, _, new_a = self._splice_soft_prompt(
                embeds, None, batch["attention_mask"], batch["input_ids"], soft, side="left"
            )
            gen = self.llm.generate(
                inputs_embeds=new_e, attention_mask=new_a,
                max_new_tokens=max_new_tokens, use_cache=True, **self._gen_defaults(gen_kwargs),
            )
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            B = len(texts)
            Q = len(CHRONOS2_QUANTILE_LEVELS)
            qp = torch.full((B, Q, horizon), float("nan"),
                            dtype=torch.float32, device=batch["input_ids"].device)
            # Per-sample true horizon: the `horizon` argument is the padded maximum FL of the batch;
            # in a mixed batch, parsing a short-fl sample against it would be misjudged as "too
            # short" -- so recover each sample's own fl from the last finite position of its future
            # row (trailing NaN padding does not count towards the length, genuine NaN gaps inside
            # the row are kept). Single target was validated above => row = sample.
            fut = batch.get("future")
            per_h = [horizon] * B
            if fut is not None and fut.shape[0] == B:
                for b in range(B):
                    fin = torch.isfinite(fut[b])
                    if fin.any():
                        per_h[b] = int(fin.nonzero().max().item()) + 1
            for b, t in enumerate(texts):
                h = min(per_h[b], horizon)
                vals = parse_forecast_values(t, h)
                if vals is not None:
                    qp[b, :, :h] = torch.from_numpy(vals).to(qp)[None, :].expand(Q, h)
            return {"text": texts, "quantile_preds": qp}
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate_forecast(self, batch: dict, horizon: int, max_new_tokens: int = 256,
                          ensemble_k: int = 1, ensemble_temperature: float = 0.7,
                          fb_mask_generated: bool = False, **gen_kwargs):
        """Generate the reasoning+conclusion text first, then feed it back into chronos for the quantile forecast.

        Returns {"text": list[str], "quantile_preds": (B, Q, ~horizon)}.
        Ablations 2/3 are not autoregressive: chronos_only predicts with chronos directly;
        forecast_prompt_only runs a plain-prompt forward pass and feeds its hidden states back
        (text is returned as empty-string placeholders).

        Inference-time options (only active on the FULL autoregressive path):
        - ``ensemble_k>1``: K-sample feedback ensemble -- besides the greedy pass, sample K-1 more
          conclusions, feed each back, and average the members' quantile curves point-wise
          (vincentization, keeps Q monotone). text still returns the greedy pass.
        - ``fb_mask_generated=True``: diagnostic switch -- the feedback masks the entire generated
          segment and only consumes the [soft prompt + prompt] prefix hidden states (= running the
          prompt-only (PO) configuration with FULL weights).
        """
        cfg = self.config
        n_samples = (len(batch["n_channels"]) if batch.get("n_channels") is not None
                     else batch["context"].shape[0])
        if getattr(cfg, "chronos_only", False):
            return {"text": [""] * n_samples,
                    "quantile_preds": self._forecast_chronos_only(batch, horizon)}
        if getattr(cfg, "llm_only", False):
            return self._generate_forecast_llm_only(batch, horizon, max_new_tokens, **gen_kwargs)
        was_training = self.training
        self.eval()
        try:
            if getattr(cfg, "forecast_prompt_only", False):
                soft = self._encode_history_to_soft_prompt(
                    batch["context"], batch["true_lengths"], batch.get("n_channels"),
                    batch.get("chunk_pos"), allow_upsample=False)
                out, new_attn = self._run_llm(
                    batch["input_ids"], batch["attention_mask"], None, soft)
                qp = self._forecast_from_hidden(
                    batch, self._fb_hidden(out.hidden_states[-1]), new_attn, horizon)
                return {"text": [""] * n_samples, "quantile_preds": qp}

            soft = self._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch.get("n_channels"),
                batch.get("chunk_pos"), allow_upsample=False)  # no upsampling in the forecasting branch
            embeds = self.llm.get_input_embeddings()(batch["input_ids"])
            new_e, _, new_a = self._splice_soft_prompt(
                embeds, None, batch["attention_mask"], batch["input_ids"], soft, side="left"
            )
            prefix_len = new_e.shape[1]
            pad_id = self.tokenizer.pad_token_id

            def _one_pass(pass_kwargs):
                gen = self.llm.generate(
                    inputs_embeds=new_e, attention_mask=new_a, max_new_tokens=max_new_tokens,
                    use_cache=True, return_dict_in_generate=True, **pass_kwargs,
                )
                seq = gen.sequences  # (B, gen_len), newly generated tokens only
                # Re-run a forward pass over "prefix + generation" to get the last hidden states
                # (generate uses the cache, so the hidden states must be recomputed). Left-padded
                # prefix followed by the generation: the real content is contiguous; trailing pad
                # after an early EOS in the generated segment is excluded via the mask.
                gen_e = self.llm.get_input_embeddings()(seq)
                full_e = torch.cat([new_e, gen_e], dim=1)
                gen_a = (seq != pad_id).long() if pad_id is not None else new_a.new_ones(seq.shape)
                full_a = torch.cat([new_a, gen_a], dim=1)
                out = self.llm(inputs_embeds=full_e, attention_mask=full_a,
                               output_hidden_states=True, use_cache=False)
                fb_mask = None
                if fb_mask_generated:
                    # Diagnostic: mask the entire generated segment (True = masked); the prefix keeps the pad mask.
                    fb_mask = torch.ones(full_a.shape, dtype=torch.bool, device=full_a.device)
                    fb_mask[:, :prefix_len] = new_a <= 0
                elif getattr(cfg, "feedback_scope", "all") == "conclusion":
                    # Ablation 1: feed back only the conclusion of the generated segment (after </think>).
                    fb_mask = self._conclusion_mask_generate(seq, prefix_len, full_a, pad_id)
                qp = self._forecast_from_hidden(
                    batch, self._fb_hidden(out.hidden_states[-1], all_hidden=out.hidden_states),
                    full_a, horizon, fb_mask=fb_mask)
                return seq, qp

            seq, qp = _one_pass(self._gen_defaults(gen_kwargs))
            text = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
            if ensemble_k > 1:
                samp = self._gen_defaults({"do_sample": True,
                                           "temperature": float(ensemble_temperature),
                                           "top_p": 0.9})
                qps = [qp] + [_one_pass(samp)[1] for _ in range(ensemble_k - 1)]
                qp = torch.stack(qps, 0).mean(0)  # the mean of monotone curves is still monotone
            return {"text": text, "quantile_preds": qp}
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate_forecast_teacher_forced(self, batch: dict, horizon: int):
        """Control evaluation: no autoregressive generation; run the LLM on the batch's teacher-forced
        **ground-truth reasoning/conclusion** (training-rendered input_ids, right-padded) to get the
        hidden states -> feed back into chronos -> quantile forecast.

        Running infer -> eval once with this and once with ``generate_forecast`` (self-generated
        reasoning) measures the exposure bias directly (distribution gap between teacher-forced
        training and self-generated inference). The hidden-state path is identical to the training
        ``forward_forecast`` (only the loss is skipped). Returns {"quantile_preds": ...}.
        """
        cfg = self.config
        if getattr(cfg, "llm_only", False):
            raise RuntimeError("llm_only has no teacher-forced forecast evaluation (the forecast values come from "
                               "the generated text itself; feeding the ground-truth text and parsing it would "
                               "trivially echo the supervised span) -- use --mode generate / "
                               "--eval_forecast_teacher_forced 0")
        if getattr(cfg, "chronos_only", False):
            return {"quantile_preds": self._forecast_chronos_only(batch, horizon)}
        was_training = self.training
        self.eval()
        try:
            soft = self._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch.get("n_channels"),
                batch.get("chunk_pos"), allow_upsample=False)  # no upsampling in the forecasting branch
            out, new_attn = self._run_llm(batch["input_ids"], batch["attention_mask"], None, soft)
            # Ablation 1: input_ids contain the true reasoning+conclusion (right-padded) and only
            # the conclusion span is fed back; ablation 3 (prompt_only): input_ids are the plain
            # prompt and the whole sequence is fed back (fb_mask=None).
            fb_mask = None
            if (not getattr(cfg, "forecast_prompt_only", False)
                    and getattr(cfg, "feedback_scope", "all") == "conclusion"):
                fb_mask = self._conclusion_feedback_mask(batch["input_ids"], soft, new_attn)
            qp = self._forecast_from_hidden(
                batch, self._fb_hidden(out.hidden_states[-1]), new_attn, horizon, fb_mask=fb_mask)
            return {"quantile_preds": qp}
        finally:
            if was_training:
                self.train()

    def rollout_forecast_rl(self, batch: dict, horizon: int, group_size: int = 6,
                            temperature: float = 1.0, max_new_tokens: int = 256):
        """GRPO rollout (FULL feedback path only; chronos_only / prompt_only have no autoregressive generation and do not apply).

        **Sample** ``group_size`` groups of reasoning+conclusion for the whole batch and return:
        - ``prefix_e, prefix_a``: left-padded inference-prefix embeds/mask (shared by the G groups, reused by rl_logp);
        - ``gen_ids`` list[G] of (B, T): generated tokens of each group;
        - ``preds`` list[G] of (Σn_targets, Q, H): that group's forecast after feedback into chronos (no_grad, for the CRPS reward);
        - ``texts`` list[G] of list[str]: generated text (for the ROI / magnitude rewards).

        **Entirely under no_grad** -- the grad-carrying forward for log_prob is split out into
        ``rl_logp()`` and back-propagated group by group in grpo_step, so the G computation graphs
        never reside in memory at once. Same layout as ``generate_forecast`` so that rollout ==
        deployment.
        """
        pad_id = self.tokenizer.pad_token_id
        # **Entirely under no_grad** (no graph retained, avoiding OOM from G simultaneous graphs):
        # G groups of generation + feedback forecast. The grad forward for log_prob is split out
        # into rl_logp() and back-propagated per group by grpo_step (only one graph alive).
        with torch.no_grad():
            soft = self._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch.get("n_channels"),
                batch.get("chunk_pos"), allow_upsample=False)
            embeds0 = self.llm.get_input_embeddings()(batch["input_ids"])
            new_e, _, new_a = self._splice_soft_prompt(
                embeds0, None, batch["attention_mask"], batch["input_ids"], soft, side="left")
            gen_ids, preds, texts = [], [], []
            for _ in range(group_size):
                gen = self.llm.generate(
                    inputs_embeds=new_e, attention_mask=new_a, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=temperature,
                    return_dict_in_generate=True, use_cache=True, **self._gen_defaults({}))
                seq = gen.sequences  # (B, T) newly generated tokens only (inputs_embeds => no prefix)
                gen_ids.append(seq)
                texts.append(self.tokenizer.batch_decode(seq, skip_special_tokens=True))
                # Feedback forecast (no_grad, for the CRPS reward): forward over prefix + generation to get the hidden states.
                gen_e = self.llm.get_input_embeddings()(seq)
                full_e = torch.cat([new_e, gen_e], dim=1)
                gen_a = (seq != pad_id).long() if pad_id is not None else new_a.new_ones(seq.shape)
                full_a = torch.cat([new_a, gen_a], dim=1)
                out = self.llm(inputs_embeds=full_e, attention_mask=full_a,
                               output_hidden_states=True, use_cache=False)
                preds.append(self._forecast_from_hidden(
                    batch, self._fb_hidden(out.hidden_states[-1], all_hidden=out.hidden_states),
                    full_a, horizon))
        return new_e, new_a, gen_ids, preds, texts   # prefix(e,a) + list[G]×3

    def rollout_understanding_rl(self, batch: dict, group_size: int = 8,
                                 temperature: float = 1.0, max_new_tokens: int = 2048,
                                 top_p: float = 1.0, merge_groups: bool = True):
        """GRPO rollout (understanding branch): **sample** ``group_size`` groups of answer text for the whole batch.

        Returns ``(prefix_e, prefix_a, gen_ids list[G] of (B,T), texts list[G] of list[str])``.
        Same layout as ``generate_understanding`` (left-padded splice, default encode parameters)
        so that rollout == deployment; sampling uses the **true policy distribution** with
        top_p=1.0 / top_k=0 (truncated sampling would make logp mismatch the actual sampling
        distribution and distort the policy gradient). **Entirely under no_grad** -- the
        grad-carrying forward for log_prob is split out into ``rl_logp()`` and back-propagated
        group by group in the training loop (only one computation graph alive at any time).

        ``merge_groups=True`` (default): concatenate the G groups into one (G·B) batch and call
        generate **once** -- incremental decoding is bound by weight-read bandwidth, so ×G
        parallelism is almost free; rows are
        sampled independently => **distributionally equivalent** to per-group generation. The cost
        is ×G KV/state and prefix embeds (batch size ~64, comfortable on a large GPU); the fla
        TileLang kernels JIT-compile once for every new batch-size value. False falls back to
        a per-group loop (memory safety net)."""
        with torch.no_grad():
            soft = self._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch.get("n_channels"),
                batch.get("chunk_pos"))
            embeds0 = self.llm.get_input_embeddings()(batch["input_ids"])
            new_e, _, new_a = self._splice_soft_prompt(
                embeds0, None, batch["attention_mask"], batch["input_ids"], soft, side="left")
            B = new_e.shape[0]
            if merge_groups:
                big = self.llm.generate(
                    inputs_embeds=new_e.repeat(group_size, 1, 1),   # [group 0: B rows; group 1: B rows; ...]
                    attention_mask=new_a.repeat(group_size, 1),
                    max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                    top_p=top_p, top_k=0, use_cache=True, **self._gen_defaults({}))
                out = big.view(group_size, B, -1)                   # (G, B, T), globally padded to a common length
                gen_ids = [out[g] for g in range(group_size)]
                texts = [self.tokenizer.batch_decode(out[g], skip_special_tokens=True)
                         for g in range(group_size)]
            else:
                gen_ids, texts = [], []
                for _ in range(group_size):
                    gen = self.llm.generate(
                        inputs_embeds=new_e, attention_mask=new_a, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                        use_cache=True, **self._gen_defaults({}))
                    gen_ids.append(gen)   # (B, T) newly generated tokens only (inputs_embeds => no prefix)
                    texts.append(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return new_e, new_a, gen_ids, texts

    def rl_logp(self, prefix_e, prefix_a, seq, logits_chunk: int = 0,
                length_norm: bool = True):
        """log_prob of one group's generated sequences (grad flows through the LLM policy = LoRA). The prefix is detached.
        Called by grpo_step once per group followed by backward, so only one computation graph is
        alive (G-fold memory saving).

        ``length_norm=True`` (default, original GRPO convention): per-sequence mean token logp,
        i.e. divided by the sequence's own length; ``False`` (Dr.GRPO convention): return the
        **summed** logp -- a per-sequence 1/|o| penalises long wrong trajectories less and
        systematically encourages verbosity, so Dr.GRPO normalises by a sample-independent constant
        instead (the caller divides by max_new_tokens). The forecasting branch does not pass this
        argument and is unchanged.

        With ``logits_chunk>0`` the log_softmax+gather is computed in chunks along the sequence
        dimension (understanding-branch generations reach thousands of tokens, and a full (B,T,V)
        float32 log_softmax alone takes 30GB+; chunking lowers the peak to (B,chunk,V) and is
        **bit-identical** to the unchunked version -- softmax runs along the vocabulary dimension,
        positions are independent). 0 = original behaviour (forecasting branch unchanged)."""
        pad_id = self.tokenizer.pad_token_id
        gen_e = self.llm.get_input_embeddings()(seq)
        full_e = torch.cat([prefix_e.detach(), gen_e], dim=1)
        gen_a = (seq != pad_id).long() if pad_id is not None else prefix_a.new_ones(seq.shape)
        full_a = torch.cat([prefix_a, gen_a], dim=1)
        out = self.llm(inputs_embeds=full_e, attention_mask=full_a, use_cache=False)  # no hidden states requested, saves memory
        P, T = prefix_e.shape[1], seq.shape[1]
        gen_logits = out.logits[:, P - 1:P - 1 + T, :]           # logits[i] predicts full[i+1] => aligned with seq
        if logits_chunk and logits_chunk > 0:
            parts = []
            for s in range(0, T, logits_chunk):
                sl = gen_logits[:, s:s + logits_chunk, :]
                lp = torch.log_softmax(sl.float(), dim=-1)
                parts.append(lp.gather(-1, seq[:, s:s + logits_chunk].unsqueeze(-1)).squeeze(-1))
            tok_logp = torch.cat(parts, dim=1)                   # (B, T)
        else:
            logp_all = torch.log_softmax(gen_logits.float(), dim=-1)
            tok_logp = logp_all.gather(-1, seq.unsqueeze(-1)).squeeze(-1)  # (B, T)
        mask = gen_a.to(tok_logp.dtype)
        s = (tok_logp * mask).sum(-1)                                    # (B,) sum
        return s / mask.sum(-1).clamp(min=1.0) if length_norm else s

    def ss_forecast_loss(self, batch: dict, max_new_tokens: int = 256) -> dict:
        """Scheduled-sampling forecast loss: feed the model's **self-generated** conclusion (instead of
        the teacher-forced true conclusion) back into chronos to obtain the pred/roi losses, training
        **chronos+feedback to adapt to the generated distribution** (LLM under no_grad): chronos,
        which otherwise only sees teacher-forced hidden states, sees generated hidden states during
        training. The dataset must be in
        inference_mode (inference prefix with reasoning=True). Isomorphic to the second half of
        forward_forecast; only the hidden states come from generation rather than the true conclusion."""
        context = batch["context"]
        cfg = self.config
        pad_id = self.tokenizer.pad_token_id
        future = batch["future"]
        fl = future.shape[-1]
        # 1. Generate the conclusion and take the hidden states (LLM fully under no_grad: SS only trains chronos+feedback)
        with torch.no_grad():
            soft = self._encode_history_to_soft_prompt(
                context, batch["true_lengths"], batch.get("n_channels"),
                batch.get("chunk_pos"), allow_upsample=False)
            embeds0 = self.llm.get_input_embeddings()(batch["input_ids"])
            new_e, _, new_a = self._splice_soft_prompt(
                embeds0, None, batch["attention_mask"], batch["input_ids"], soft, side="left")
            gen = self.llm.generate(
                inputs_embeds=new_e, attention_mask=new_a, max_new_tokens=max_new_tokens,
                do_sample=False, return_dict_in_generate=True, use_cache=True, **self._gen_defaults({}))
            seq = gen.sequences
            gen_e = self.llm.get_input_embeddings()(seq)
            full_e = torch.cat([new_e, gen_e], dim=1)
            gen_a = (seq != pad_id).long() if pad_id is not None else new_a.new_ones(seq.shape)
            full_a = torch.cat([new_a, gen_a], dim=1)
            out = self.llm(inputs_embeds=full_e, attention_mask=full_a,
                           output_hidden_states=True, use_cache=False)
            hidden = self._fb_hidden(out.hidden_states[-1], all_hidden=out.hidden_states)
        # 2. Generated hidden -> feedback -> cross_states -> chronos.forecast_losses (grad reaches chronos+feedback)
        c_llm = self.feedback_qformer(
            hidden.to(self.feedback_qformer.input_proj.weight.dtype),
            src_key_padding_mask=(full_a <= 0)).to(self.chronos.dtype)
        group_ids, target_idx, future_covariates = self._forecast_layout(batch, context)
        cross_states = c_llm[group_ids]
        cross_mask = torch.ones(cross_states.shape[0], cross_states.shape[1], device=cross_states.device)
        res = self.chronos.forecast_losses(
            context=context, future_target=future, num_output_patches=self._num_output_patches(fl),
            context_mask=batch.get("context_mask"), group_ids=group_ids,
            cross_states=cross_states, cross_states_mask=cross_mask,
            roi_mask=batch.get("roi_mask"), future_covariates=future_covariates, target_idx=target_idx)
        pred_loss, roi_loss = res["pred_loss"], res["roi_loss"]
        total = cfg.pred_loss_weight * pred_loss
        if roi_loss is not None:
            total = total + cfg.roi_loss_weight * roi_loss
        return {"loss": total, "pred_loss": pred_loss.detach(),
                "roi_loss": (roi_loss.detach() if roi_loss is not None else None)}

    def tf_forecast_loss(self, batch: dict) -> dict:
        """Frozen-LLM forecast loss: teacher-text hidden states (LLM under no_grad) are fed back to
        train chronos+feedback.

        Isomorphic to ``ss_forecast_loss``; only the hidden states come from a teacher-forced forward
        pass (no generation). The dataset must use the training rendering (inference_mode=False;
        input_ids contain the true reasoning/conclusion)."""
        context = batch["context"]
        cfg = self.config
        future = batch["future"]
        fl = future.shape[-1]
        with torch.no_grad():
            soft = self._encode_history_to_soft_prompt(
                context, batch["true_lengths"], batch.get("n_channels"),
                batch.get("chunk_pos"), allow_upsample=False)
            out, new_a = self._run_llm(batch["input_ids"], batch["attention_mask"], None, soft)
            hidden = self._fb_hidden(out.hidden_states[-1])   # (B, L', llm_hidden), no_grad
        c_llm = self.feedback_qformer(
            hidden.to(self.feedback_qformer.input_proj.weight.dtype),
            src_key_padding_mask=(new_a <= 0)).to(self.chronos.dtype)
        group_ids, target_idx, future_covariates = self._forecast_layout(batch, context)
        cross_states = c_llm[group_ids]
        cross_mask = torch.ones(cross_states.shape[0], cross_states.shape[1], device=cross_states.device)
        res = self.chronos.forecast_losses(
            context=context, future_target=future, num_output_patches=self._num_output_patches(fl),
            context_mask=batch.get("context_mask"), group_ids=group_ids,
            cross_states=cross_states, cross_states_mask=cross_mask,
            roi_mask=batch.get("roi_mask"), future_covariates=future_covariates, target_idx=target_idx)
        pred_loss, roi_loss = res["pred_loss"], res["roi_loss"]
        total = cfg.pred_loss_weight * pred_loss
        if roi_loss is not None:
            total = total + cfg.roi_loss_weight * roi_loss
        return {"loss": total, "pred_loss": pred_loss.detach(),
                "roi_loss": (roi_loss.detach() if roi_loss is not None else None)}

    # ------------------------------------------------- HF Trainer integration
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False}
            )
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()
        self._encode_grad_ckpt = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()
        self._encode_grad_ckpt = False

    def trainable_parameter_summary(self) -> str:
        groups = {
            "llm(lora+saved)": self.llm,
            "chronos2": self.chronos,
            "history_qformer": self.history_qformer,
            "feedback_qformer": self.feedback_qformer,
        }
        lines = []
        total_t = 0
        for name, mod in groups.items():
            if mod is None:  # llm_only: chronos / the two Q-formers do not exist
                continue
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            a = sum(p.numel() for p in mod.parameters())
            total_t += t
            lines.append(f"  {name:22s} trainable {t/1e6:8.2f}M / total {a/1e6:8.2f}M")
        lines.append(f"  {'TOTAL trainable':22s} {total_t/1e6:8.2f}M")
        return "\n".join(lines)


def freeze_ts_modules(model) -> int:
    """Freeze chronos2 + both Q-formers (including their modules_to_save copies), leaving only the LLM LoRA trainable.

    Used by train.py ``--freeze_ts``; the mirror image of ``--freeze_llm``, with the same
    name-matching rule. Returns the number of frozen tensors.
    """
    n_frozen = 0
    for n, prm in model.named_parameters():
        if ("chronos" in n) or ("history_qformer" in n) or ("feedback_qformer" in n):
            if prm.requires_grad:
                prm.requires_grad_(False)
                n_frozen += 1
    return n_frozen


def add_lora(model: "ChronosLLM", r: int = 8, alpha: int = 32, dropout: float = 0.1,
             train_embed_tokens: bool = False):
    """Training assembly: attach a single LoRA to the whole ChronosLLM.

    - LoRA is injected into the LLM only (the ``target_modules`` regex is anchored on the ``^llm.``
      prefix, avoiding the out_proj of chronos / qformer).
    - chronos2 and both Q-formers are made trainable and saved with the adapter via
      ``modules_to_save`` (lm_head excluded).
    - ``train_embed_tokens=True`` additionally adds embed_tokens to ``modules_to_save`` (to train
      the <ts_*> input rows).

    Returns the PeftModel produced by ``get_peft_model``.
    """
    from peft import LoraConfig, get_peft_model

    if getattr(model, "chronos", None) is None:
        # llm_only: no chronos / Q-formers => modules_to_save is at most the optional embed_tokens;
        # trainable parameters = LLM LoRA (same LoRA config / lr group as the composite), base LLM frozen.
        mts = ["embed_tokens"] if train_embed_tokens else None
    else:
        mts = list(TRAINABLE_MODULES) + (["embed_tokens"] if train_embed_tokens else [])
        # This small from-scratch projection only exists when history_stats_token is on -> it must be
        # in modules_to_save to be trained and saved with the adapter.
        if getattr(model, "history_stats_proj", None) is not None:
            mts.append("history_stats_proj")
    lora = LoraConfig(
        task_type=None,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LLM_LORA_TARGET_REGEX,
        modules_to_save=mts,
    )
    return get_peft_model(model, lora)


def _add_ts_special_tokens(tokenizer) -> int:
    """Add the two time-series boundary markers <ts_start>/<ts_end> to the tokenizer; returns the number added.

    The soft prompt is spliced directly after <ts_start> during forward (embedding vectors are
    inserted, not placeholder tokens), so no <ts_pad>-style placeholder tokens are needed.
    """
    return tokenizer.add_special_tokens(
        {"additional_special_tokens": [TS_START, TS_END]}
    )
