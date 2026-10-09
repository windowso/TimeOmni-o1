"""Q-former modules.

``QFormer`` (BLIP-2 style: learned queries run self-attn + cross-attn into a source
sequence + FFN) is adapted from the TimeOmni-v2 ``models/qformer.py``. On top of it we add
``SlidingWindowQFormer``: it slides **non-overlapping** windows over the chronos2 patch
sequence, compresses each window with the same underlying ``QFormer`` into k query tokens,
and concatenates them in temporal order into an order-preserving soft-prompt sequence. The
window size grows monotonically with the history length (longer history -> larger window ->
stronger downsampling), and the total number of soft tokens follows a **log-linear budget**
(``token_min/token_max/token_pc_ref``) clamped to ``[token_min, token_max]`` (see
``SlidingWindowQFormer.plan``).
(The old ``window_ladder``/``target_windows`` ladder scheme is defunct and kept only for
backward compatibility -- see the class docstring.)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class _QFormerBlock(nn.Module):
    """One Q-former block: self-attn over queries, cross-attn into source, FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ffn_mult: float = 4.0,
    ):
        super().__init__()
        self.self_attn_ln = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_ln_q = nn.LayerNorm(hidden_dim)
        self.cross_attn_ln_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        ffn_hidden = int(round(hidden_dim * ffn_mult))
        self.ffn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: Tensor,
        kv: Tensor,
        kv_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q_norm = self.self_attn_ln(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        queries = queries + sa_out

        q_norm = self.cross_attn_ln_q(queries)
        kv_norm = self.cross_attn_ln_kv(kv)
        ca_out, _ = self.cross_attn(
            q_norm,
            kv_norm,
            kv_norm,
            key_padding_mask=kv_key_padding_mask,
            need_weights=False,
        )
        queries = queries + ca_out

        queries = queries + self.ffn(self.ffn_ln(queries))
        return queries


class QFormer(nn.Module):
    """BLIP-2-style Q-former: learned queries cross-attend to a sequence of source tokens.

    Args:
        in_dim: Source feature dim (e.g. ts encoder hidden, or LLM hidden).
        out_dim: Output (query) hidden dim. Should match the dim of the consumer module
            so cross-attn KV needs no further projection.
        num_query_tokens: Number of learned query tokens.
        num_heads: Attention heads inside the Q-former.
        num_layers: Number of stacked Q-former blocks.
        dropout: Attention/FFN dropout.
        ffn_mult: FFN expansion ratio.
        hidden_dim: Internal working width (BLIP-2 style: all self-attn/cross-attn/FFN run
            at this width, and a final Linear projects to out_dim). None = in_dim -- the KV
            source only carries in_dim dimensions of information, so compressing at the
            source width is enough; compared with working at out_dim internally (= LLM
            hidden 4096) this shrinks the parameter count by ~(out/in)^2.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_query_tokens: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        ffn_mult: float = 4.0,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if num_query_tokens <= 0:
            raise ValueError(f"num_query_tokens must be positive, got {num_query_tokens}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        hidden = in_dim if hidden_dim is None else int(hidden_dim)
        if hidden <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden}) must be divisible by num_heads ({num_heads})"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden
        self.num_query_tokens = num_query_tokens

        self.input_proj = nn.Linear(in_dim, hidden)
        self.input_ln = nn.LayerNorm(hidden)

        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _QFormerBlock(
                    hidden_dim=hidden,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compress ``src`` into ``num_query_tokens`` learned query vectors.

        Args:
            src: (B, T, in_dim) source token embeddings.
            src_key_padding_mask: (B, T) bool, True = padded position (ignored).

        Returns:
            (B, num_query_tokens, out_dim) query outputs.
        """
        if src.dim() != 3:
            raise ValueError(f"src must be (B, T, D), got shape {tuple(src.shape)}")
        if src.size(-1) != self.in_dim:
            raise ValueError(
                f"src last dim ({src.size(-1)}) does not match in_dim ({self.in_dim})"
            )

        kv = self.input_ln(self.input_proj(src))

        if src_key_padding_mask is not None:
            if src_key_padding_mask.shape != src.shape[:2]:
                raise ValueError(
                    "src_key_padding_mask shape must equal src.shape[:2]:"
                    f" {tuple(src_key_padding_mask.shape)} != {tuple(src.shape[:2])}"
                )
            kpm = src_key_padding_mask.to(dtype=torch.bool, device=kv.device)
            # If a row is fully padded, MHA would NaN. Flip one position to valid
            # to keep gradients finite; the consumer should not use that row anyway.
            all_padded = kpm.all(dim=1)
            if all_padded.any():
                kpm = kpm.clone()
                kpm[all_padded, 0] = False
        else:
            kpm = None

        queries = self.query_tokens.expand(src.size(0), -1, -1).to(dtype=kv.dtype)
        for block in self.blocks:
            queries = block(queries, kv, kv_key_padding_mask=kpm)

        return self.out_proj(self.output_ln(queries))


DEFAULT_WINDOW_LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _sinusoidal_pe_at(pos: Tensor, dim: int, dtype) -> Tensor:
    """Sin/cos encoding ``(P, dim)`` evaluated at arbitrary (possibly non-integer) positions
    ``pos (P,)``; parameter-free and usable at any length.

    The KV of the global-query branch covers the whole history (up to tens of thousands of
    patches), which exceeds the ``time_pos_embedding`` table size (= largest ladder window),
    so a parameter-free sinusoidal encoding provides temporal-order information over the full
    length. After window sampling, a patch's index in the concatenated sequence is no longer
    proportional to real time, so the global-branch PE must use externally supplied real
    positions; ``_sinusoidal_pe(L) == _sinusoidal_pe_at(arange(L))`` holds element-wise.
    """
    pos = pos.to(torch.float32).unsqueeze(1)
    half = torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * half / dim)
    pe = torch.zeros(pos.shape[0], dim, device=pos.device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : dim // 2]
    return pe.to(dtype)


def _sinusoidal_pe(length: int, dim: int, device, dtype) -> Tensor:
    """Standard sin/cos absolute positional encoding (length, dim) (= evaluated at indices 0..L-1)."""
    return _sinusoidal_pe_at(
        torch.arange(length, device=device, dtype=torch.float32), dim, dtype)


class SlidingWindowQFormer(nn.Module):
    """Order-preserving sliding-window Q-former (ladder windows + non-overlapping + bounded
    budget + minimum window count).

    Design: given a source sequence ``(B, P, in_dim)`` (P **all-valid** patches; the caller
    guarantees no padding),

    1. **Log-linear token budget**: ``plan(P, C)`` first computes the total soft-token budget
       (including the G global tokens) as ``T = clamp(token_min + token_b*log2(P*C),
       token_min, token_max)``, then derives the number of local windows ``nw`` and the
       window size ``W = ceil(P/nw)`` (see the ``plan`` docstring). Larger P*C -> larger T,
       larger W (stronger downsampling), but the token count is capped at ``token_max``.
    2. **Non-overlapping**: stride == W; windows ``[i*W, (i+1)*W)`` do not overlap and cover
       P completely.
    3. **Bounded budget**: the output token count ``= G + nw*k`` is clamped by ``token_max``
       (logarithmic growth, weakly dependent on history length).

    .. note:: **Legacy (defunct) parameters**: ``target_windows`` / ``min_windows`` /
       ``window_ladder`` are leftovers of the old "ladder window" scheme. They no longer take
       part in window planning and are kept only so that old checkpoint configs still load --
       ``target_windows`` / ``min_windows`` are **never read** in ``plan()`` (only the
       ``target_windows >= min_windows`` validity check in ``__init__`` remains);
       the only remaining effect of ``window_ladder`` is that ``max(ladder)`` sizes the
       intra-window time-position embedding table (see ``_time_pos_size``). To change the
       token budget use ``token_min/token_max/token_pc_ref``; tuning ``target_windows`` does
       not change any output.

    Each window is compressed by the same underlying ``QFormer`` into ``queries_per_window``
    (k) tokens, concatenated in temporal order into ``(B, nw*k, out_dim)``. Short histories
    (small P) get W=1 -> every patch is its own window (no downsampling, finest granularity);
    long histories (large P) get a larger W and coarser windows (downsampling).

    **Global-query branch** (enabled when ``global_queries=G>0``): windows never interact
    inside this module, and chronos2 chunks never interact either, so cross-window global
    structure (whole-range trend / large-scale periodicity) could originally only be pieced
    together by the LLM from local summaries -- and cannot be recovered if the window
    compression dropped it. The global branch uses G independent learned queries that
    cross-attend to the **whole** ``C*P`` patch sequence (linear in KV length, no KV
    self-attention, affordable for long histories). The KV carries the shared channel
    embedding plus a sinusoidal time-position encoding (usable at full length, not bounded by
    the ``time_pos_embedding`` table size), and the output is placed **before** the local
    tokens: ``[G global | nw*k local]``. With G=0 the module is exactly equivalent to having no
    such branch.

    Args:
        in_dim: Source dim (chronos2 d_model=768).
        out_dim: Output dim (should match the LLM hidden size, e.g. 4096).
        queries_per_window: Number of query tokens (k) each window is compressed into.
        target_windows: **[Defunct, backward compatibility only]** upper bound on the window
            count in the old ladder scheme; ``plan()`` no longer reads it, only the
            ``target_windows >= min_windows`` construction-time check remains. The token
            budget is now controlled by ``token_max``.
        min_windows: **[Defunct, backward compatibility only]** lower bound on the window
            count in the old ladder scheme; ``plan()`` no longer reads it.
        window_ladder: **[Mostly defunct, backward compatibility only]** old ladder of
            candidate window sizes; its only remaining effect is that ``max(window_ladder)``
            sizes the intra-window time-position embedding table (no longer used to pick the
            window size).
        global_queries: Number of global query tokens (G); 0 = disable the global branch.
        num_heads/num_layers/dropout: Passed through to the underlying QFormer.
        hidden_dim: Internal working width of the underlying QFormer (both the local and the
            global branch); None = in_dim (chronos d_model, the information bottleneck width
            of the KV source). The output is still out_dim (linear projection at the end of
            the QFormer); compared with working at out_dim internally this shrinks the
            parameter count by ~(out/in)^2.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        queries_per_window: int = 4,
        target_windows: int = 16,
        min_windows: int = 4,
        window_ladder: tuple = DEFAULT_WINDOW_LADDER,
        global_queries: int = 0,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_channels: int = 64,
        intra_window_pos: bool = True,
        hidden_dim: Optional[int] = None,
        token_min: int = 40,
        token_max: int = 200,
        token_pc_ref: int = 32768,
    ):
        super().__init__()
        if queries_per_window <= 0:
            raise ValueError(f"queries_per_window must be positive, got {queries_per_window}")
        if min_windows <= 0 or target_windows < min_windows:
            raise ValueError(
                f"need 0 < min_windows <= target_windows, got {min_windows}/{target_windows}"
            )
        if global_queries < 0:
            raise ValueError(f"global_queries must be >= 0, got {global_queries}")
        ladder = tuple(sorted({int(w) for w in window_ladder if int(w) > 0}))
        if not ladder:
            raise ValueError("window_ladder must contain at least one positive integer")
        self.queries_per_window = queries_per_window
        self.target_windows = target_windows
        self.min_windows = min_windows
        self.window_ladder = ladder
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_channels = max_channels
        self.intra_window_pos = intra_window_pos
        self.global_queries = global_queries
        # Log-linear token budget: T = clamp(token_min + token_b*log2(P*C), token_min, token_max).
        # token_pc_ref = the P*C anchor at which T reaches token_max (log2 needs >1, hence the floor of 2).
        self.token_min = int(token_min)
        self.token_max = int(token_max)
        self.token_pc_ref = max(2, int(token_pc_ref))
        self.token_b = (self.token_max - self.token_min) / math.log2(self.token_pc_ref)
        self.qformer = QFormer(
            in_dim=in_dim,
            out_dim=out_dim,
            num_query_tokens=queries_per_window,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        if global_queries > 0:
            self.global_qformer = QFormer(
                in_dim=in_dim,
                out_dim=out_dim,
                num_query_tokens=global_queries,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                hidden_dim=hidden_dim,
            )
        # Multi-channel: channel-id embedding (required, tells apart which channel each of the
        # C*W KV entries inside a window comes from).
        self.channel_embedding = nn.Embedding(max_channels, in_dim)
        nn.init.normal_(self.channel_embedding.weight, std=0.02)
        # Intra-window relative time-position embedding (configurable, on by default; table size =
        # largest ladder window, out-of-range positions are clamped to reuse the last row).
        self._time_pos_size = max(ladder)
        if intra_window_pos:
            self.time_pos_embedding = nn.Embedding(self._time_pos_size, in_dim)
            nn.init.normal_(self.time_pos_embedding.weight, std=0.02)

    def plan(self, length: int, channels: int = 1) -> tuple[int, int]:
        """Given the number of valid patches ``length`` and the channel count ``channels``,
        return (num_windows, window_size).

        The window count is derived from the log-linear token budget:
        1. T = clamp(token_min + token_b*log2(P*C), token_min, token_max) (total tokens, incl. the G global ones).
        2. Local window target nw_target = round((T - G) / k) (G=global_queries, k=queries_per_window).
        3. nw = min(nw_target, P) (nw <= P: for tiny P this naturally falls to a soft lower bound, tokens < token_min).
        4. W = ceil(P/nw); then recompute nw = ceil(P/W) -- the ceil guarantees (nw-1)*W < P <= nw*W
           (complete coverage, no empty window), at the cost of a +-1 quantisation jitter in nw for
           small P; tokens are always <= token_max (the upper bound is never exceeded).

        Note: this method does **not** read ``target_windows`` / ``min_windows`` / ``window_ladder``
        (leftovers of the old ladder scheme, see the class docstring); the window size W comes
        directly from the formula above and is no longer picked from the ``window_ladder``.
        """
        P = max(int(length), 1)
        C = max(int(channels), 1)
        T = self.token_min + self.token_b * math.log2(P * C)
        T = int(round(min(max(T, self.token_min), self.token_max)))
        nw_target = max(1, int(round((T - self.global_queries) / self.queries_per_window)))
        nw = min(nw_target, P)
        W = math.ceil(P / nw)
        nw = math.ceil(P / W)
        return nw, W

    def num_windows_for_length(self, length: int, channels: int = 1) -> int:
        return self.plan(length, channels)[0]

    def num_tokens_for_length(self, length: int, channels: int = 1) -> int:
        """Total soft-prompt token count = G global + nw*k local (the dynamic-batching budget uses this)."""
        return self.plan(length, channels)[0] * self.queries_per_window + self.global_queries

    def forward(self, src: Tensor, patch_pos: Tensor | None = None) -> Tensor:
        """Args:
            src: ``(B, C, P, in_dim)`` multi-channel source (C channels, P time patches, **all
                valid**), or ``(B, P, in_dim)`` (treated as C=1). Usually called per sample with B=1.
            patch_pos: ``(P,)`` real time position of each patch (in patch units, may be
                non-integer); None = use indices 0..P-1 (element-wise identical to the behaviour
                before window sampling). Only affects the global-branch PE.
        Returns:
            ``(B, G + num_windows * queries_per_window, out_dim)``; the first G entries are the
            global tokens (absent when ``global_queries=0``). The window count ``nw`` is decided
            by the **log-linear token budget** of ``plan(P, C)`` (derived from
            ``T = clamp(token_min + b*log2(P*C), token_min, token_max)``) and **varies with both
            P and C** (C enters the budget through ``log2(P*C)``); the token count grows
            logarithmically and is capped at ``token_max`` (not a fixed budget). Each time
            window takes the patches of that window x all C channels (``C*w`` of them) as KV,
            with the channel-id embedding and (optionally) the intra-window relative
            time-position embedding added.
        """
        if src.dim() == 3:
            src = src.unsqueeze(1)  # (B,P,D) -> (B,1,P,D), degenerate C=1 entry point
        if src.dim() != 4:
            raise ValueError(f"src must be (B,C,P,D) or (B,P,D), got {tuple(src.shape)}")
        B, C, P, D = src.shape
        if C > self.max_channels:
            raise ValueError(f"channels C={C} exceeds max_channels={self.max_channels}")
        nw, w = self.plan(P, C)

        # Channel-id embedding: added to every (channel, patch) (broadcast along time).
        ch_emb = self.channel_embedding(torch.arange(C, device=src.device))  # (C, D)
        src = src + ch_emb[None, :, None, :]

        # Global branch: G queries cross-attend to the whole C*P patch sequence (KV carries a
        # sinusoidal time PE for full-length ordering information; the channel embedding is already
        # added to src). No padding: the caller guarantees all P patches are valid.
        global_out = None
        if self.global_queries > 0:
            if patch_pos is None:
                pe = _sinusoidal_pe(P, D, src.device, src.dtype)
            else:
                if patch_pos.numel() != P:
                    raise ValueError(f"patch_pos length {patch_pos.numel()} != number of patches {P}")
                pe = _sinusoidal_pe_at(patch_pos.to(src.device), D, src.dtype)
            g_kv = (src + pe[None, None, :, :]).reshape(B, C * P, D)
            global_out = self.global_qformer(g_kv)  # (B, G, out_dim)

        # Cut non-overlapping time windows; each window takes all C channels -> flattened (C*w) as KV.
        # The last window is right-padded with zeros + masked.
        win_list = []
        mask_list = []
        for i in range(nw):
            s = i * w
            e = min(P, s + w)
            if s >= P:  # defensive: already avoided by the floor; fully padded window
                chunk = src.new_zeros(B, C, w, D)
                cmask = torch.ones(B, C * w, dtype=torch.bool, device=src.device)
            else:
                chunk = src[:, :, s:e, :]  # (B, C, cur, D)
                cur = e - s
                if self.intra_window_pos:
                    idx = torch.arange(cur, device=src.device).clamp_(max=self._time_pos_size - 1)
                    chunk = chunk + self.time_pos_embedding(idx)[None, None, :, :]
                valid = torch.zeros(B, C, cur, dtype=torch.bool, device=src.device)
                if cur < w:
                    pad_n = w - cur
                    chunk = torch.cat([chunk, src.new_zeros(B, C, pad_n, D)], dim=2)
                    valid = torch.cat(
                        [valid, torch.ones(B, C, pad_n, dtype=torch.bool, device=src.device)], dim=2
                    )
                cmask = valid.reshape(B, C * w)  # flatten channel x intra-window time
            win_list.append(chunk.reshape(B, C * w, D))
            mask_list.append(cmask)

        # Run (B*nw, C*w, D) through the QFormer in one go, then restore the temporally
        # concatenated (B, nw*k, out_dim).
        windows = torch.stack(win_list, dim=1).reshape(B * nw, C * w, D)
        masks = torch.stack(mask_list, dim=1).reshape(B * nw, C * w)
        out = self.qformer(windows, src_key_padding_mask=masks)  # (B*nw, k, out_dim)
        out = out.reshape(B, nw * self.queries_per_window, self.out_dim)
        if global_out is not None:
            out = torch.cat([global_out, out], dim=1)  # [G global | nw*k local]
        return out


class SlidingWindowPooler(SlidingWindowQFormer):
    """Architecture ablation A1: a learned-query-free control that is **token-for-token
    aligned** with :class:`SlidingWindowQFormer`.

    It keeps exactly the same token-budget planning (inherits ``plan()``), channel-id
    embedding, intra-window time PE and the sinusoidal PE of the global branch -- **the only
    difference is the compression operator**: "k learned queries cross-attending to the C x w
    patches of a window" is replaced by "split the window into k equal segments along
    **time** and take a masked mean over (C x within-segment time) per segment"; the global
    branch is likewise replaced by "split the whole range into G equal time segments and take
    the mean". Output token count, order and position are element-wise identical to the
    Q-former arm; only the learned queries + attention are removed (params ~60M -> ~6M).

    This ablation therefore isolates the learned-query cross-attention, not compression itself.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Drop the learned-query / attention blocks and replace them with two linear projections
        # (one local, one global, mirroring the original two branches).
        del self.qformer
        if self.global_queries > 0:
            del self.global_qformer
        self.local_ln = nn.LayerNorm(self.in_dim)
        self.local_proj = nn.Linear(self.in_dim, self.out_dim)
        if self.global_queries > 0:
            self.global_ln = nn.LayerNorm(self.in_dim)
            self.global_proj = nn.Linear(self.in_dim, self.out_dim)

    @staticmethod
    def _seg_mean(x: Tensor, n_seg: int) -> Tensor:
        """Split ``(B, C, T, D)`` into ``n_seg`` equal segments along time and average each
        segment over (C, within-segment time) -> ``(B, n_seg, D)``.

        When T < n_seg the ``linspace`` boundaries are used and empty segments fall back to the
        whole-range mean (no NaN is produced).
        """
        B, C, T, D = x.shape
        flat = x.mean(dim=1)  # (B, T, D): average channels first (channel identity is already carried additively by channel_embedding)
        whole = flat.mean(dim=1)  # (B, D) fallback
        bounds = torch.linspace(0, T, n_seg + 1, device=x.device)
        segs = []
        for i in range(n_seg):
            s, e = int(bounds[i].item()), int(bounds[i + 1].item())
            segs.append(flat[:, s:e].mean(dim=1) if e > s else whole)
        return torch.stack(segs, dim=1)  # (B, n_seg, D)

    def forward(self, src: Tensor, patch_pos: Tensor | None = None) -> Tensor:
        if src.dim() == 3:
            src = src.unsqueeze(1)
        if src.dim() != 4:
            raise ValueError(f"src must be (B,C,P,D) or (B,P,D), got {tuple(src.shape)}")
        B, C, P, D = src.shape
        if C > self.max_channels:
            raise ValueError(f"channels C={C} exceeds max_channels={self.max_channels}")
        nw, w = self.plan(P, C)

        ch_emb = self.channel_embedding(torch.arange(C, device=src.device))
        src = src + ch_emb[None, :, None, :]

        global_out = None
        if self.global_queries > 0:
            if patch_pos is None:
                pe = _sinusoidal_pe(P, D, src.device, src.dtype)
            else:
                if patch_pos.numel() != P:
                    raise ValueError(f"patch_pos length {patch_pos.numel()} != number of patches {P}")
                pe = _sinusoidal_pe_at(patch_pos.to(src.device), D, src.dtype)
            g = self._seg_mean(src + pe[None, None, :, :], self.global_queries)
            global_out = self.global_proj(self.global_ln(g).to(self.global_proj.weight.dtype))

        outs = []
        for i in range(nw):
            s = i * w
            e = min(P, s + w)
            if s >= P:  # defensive: same fully-padded-window fallback as the Q-former arm
                outs.append(src.new_zeros(B, self.queries_per_window, D))
                continue
            chunk = src[:, :, s:e, :]
            if self.intra_window_pos:
                idx = torch.arange(e - s, device=src.device).clamp_(max=self._time_pos_size - 1)
                chunk = chunk + self.time_pos_embedding(idx)[None, None, :, :]
            outs.append(self._seg_mean(chunk, self.queries_per_window))
        local = torch.cat(outs, dim=1)  # (B, nw*k, D)
        out = self.local_proj(self.local_ln(local).to(self.local_proj.weight.dtype))
        if global_out is not None:
            out = torch.cat([global_out, out], dim=1)
        return out


class SegmentMeanPooler(nn.Module):
    """Architecture ablation B1: a learned-query-free control for the feedback Q-former.

    Splits the LLM hidden states ``(B, L, in_dim)`` into ``M`` equal time segments **according
    to each sample's own valid length**, takes a masked mean per segment, and linearly projects
    to ``out_dim`` -> ``(B, M, out_dim)``. The token count M and the injection mechanism
    (per-layer gated cross-attn) are identical to the Q-former arm; the only difference is the
    compression operator.

    .. note:: This is not the same as the "single-vector / mean-pool **injection**" ablation --
       that one changes the injection mechanism (compress to one vector and inject once); here
       the injection side is untouched and only the compressor is swapped.

    The attribute name ``input_proj`` matches :class:`QFormer` (callers read
    ``input_proj.weight.dtype`` to obtain the dtype).
    """

    def __init__(self, in_dim: int, out_dim: int, num_query_tokens: int):
        super().__init__()
        if num_query_tokens <= 0:
            raise ValueError(f"num_query_tokens must be positive, got {num_query_tokens}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_query_tokens = num_query_tokens
        self.input_ln = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, out_dim)

    def forward(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src.dim() != 3:
            raise ValueError(f"src must be (B, T, D), got shape {tuple(src.shape)}")
        if src.size(-1) != self.in_dim:
            raise ValueError(f"src last dim ({src.size(-1)}) does not match in_dim ({self.in_dim})")
        B, L, D = src.shape
        M = self.num_query_tokens
        if src_key_padding_mask is None:
            valid = src.new_ones(B, L, dtype=torch.bool)
        else:
            valid = ~src_key_padding_mask.to(dtype=torch.bool, device=src.device)
        # Per sample, split into M equal segments by **valid-position index**: the j-th valid
        # position -> segment floor(j*M/n_valid).
        pos = (valid.cumsum(dim=1) - 1).clamp_(min=0)          # (B, L) valid-position index
        n_valid = valid.sum(dim=1, keepdim=True).clamp_(min=1)  # (B, 1)
        seg = (pos * M // n_valid).clamp_(max=M - 1)            # (B, L)
        w = valid.to(src.dtype)
        num = src.new_zeros(B, M, D).scatter_add_(
            1, seg.unsqueeze(-1).expand(-1, -1, D), src * w.unsqueeze(-1))
        den = src.new_zeros(B, M).scatter_add_(1, seg, w).clamp_(min=1e-6)
        pooled = num / den.unsqueeze(-1)
        # Fully padded rows (which the caller ignores on the QFormer side too) and empty segments:
        # after the den floor they become zero vectors, no NaN is produced.
        return self.input_proj(self.input_ln(pooled).to(self.input_proj.weight.dtype))
