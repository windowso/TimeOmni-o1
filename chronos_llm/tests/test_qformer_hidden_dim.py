"""Pure-CPU unit test for the Q-former's internal low width (hidden_dim) + output projection.

Structure (BLIP-2 style): all self-attn / cross-attn / FFN run at hidden_dim width (default =
in_dim, i.e. the information-bottleneck width of the KV source), with a single Linear at the end
projecting to out_dim. With out_dim = LLM hidden (4096), running the interior at 4096 as well makes
attn/FFN parameters explode quadratically with the width;
compressing at low width shrinks parameters by ~(out/in)^2 with the output dimension unchanged.

Five properties are verified:
1. QFormer(hidden_dim=H): internal blocks/query width = H, out_proj is always Linear(H, out_dim),
   output shape (B,k,out_dim), finite values, gradients fully connected; the padding-mask path does
   not regress.
2. Parameter savings: with hidden << out_dim the parameter count is far below the full-width
   interior (< 1/3).
3. Default hidden_dim = in_dim; invalid values (not divisible by num_heads / non-positive) raise
   ValueError.
4. SlidingWindowQFormer(hidden_dim=H) forwards to both the local and the global branch; the
   token-budget interface (plan/num_tokens_for_length) is unaffected; multi-channel / single-channel
   forward works; gradients connected.
5. ChronosLLMConfig.qformer_hidden_dim defaults to 768 (= chronos d_model) and survives a round trip.
"""

import torch
import torch.nn as nn

from chronos_llm.models.qformer import QFormer, SlidingWindowQFormer


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    IN, H, OUT, K = 32, 32, 128, 4

    # ---------------- 1. basic QFormer: internal width H, output projected to OUT ----------------
    q = QFormer(in_dim=IN, out_dim=OUT, num_query_tokens=K, num_heads=4,
                num_layers=2, hidden_dim=H)
    assert q.query_tokens.shape[-1] == H, "query tokens must be at hidden_dim width"
    for blk in q.blocks:
        assert blk.ffn[0].in_features == H, "block FFN must be at hidden_dim width"
        assert blk.self_attn.embed_dim == H, "self_attn must be at hidden_dim width"
    assert isinstance(q.out_proj, nn.Linear), "output projection must always be Linear (no compatibility branch)"
    assert (q.out_proj.in_features, q.out_proj.out_features) == (H, OUT)
    src = torch.randn(2, 50, IN)
    out = q(src)
    assert out.shape == (2, K, OUT), out.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad.abs().sum().item() for p in q.parameters() if p.grad is not None]
    assert len(grads) == len(list(q.parameters())) and sum(grads) > 0, "gradients not connected"

    # padding-mask path does not regress (fully padded rows are flipped to avoid NaN)
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[1] = True
    assert torch.isfinite(q(src, src_key_padding_mask=mask)).all()

    # ---------------- 2. parameter savings ----------------
    q_wide = QFormer(in_dim=IN, out_dim=OUT, num_query_tokens=K, num_heads=4,
                     num_layers=2, hidden_dim=OUT)
    ratio = n_params(q) / n_params(q_wide)
    print(f"narrow={n_params(q):,} wide={n_params(q_wide):,} ratio={ratio:.3f}")
    assert ratio < 1 / 3, f"hidden_dim={H} should shrink parameters substantially, got ratio={ratio:.3f}"

    # ---------------- 3. default hidden_dim = in_dim; invalid values raise ----------------
    q_default = QFormer(in_dim=IN, out_dim=OUT, num_query_tokens=K, num_heads=4, num_layers=2)
    assert q_default.hidden_dim == IN, "default internal width must be in_dim"
    assert q_default.query_tokens.shape[-1] == IN
    for bad in (30, 0, -8):
        try:
            QFormer(in_dim=IN, out_dim=OUT, num_query_tokens=K, num_heads=4,
                    num_layers=1, hidden_dim=bad)
            raise AssertionError(f"hidden_dim={bad} must raise ValueError")
        except ValueError:
            pass

    # ---------------- 4. SlidingWindowQFormer pass-through ----------------
    G = 8
    common = dict(in_dim=IN, out_dim=OUT, queries_per_window=K,
                  target_windows=16, min_windows=4, num_heads=4, num_layers=1)
    swq = SlidingWindowQFormer(global_queries=G, hidden_dim=H, **common)
    swq_wide = SlidingWindowQFormer(global_queries=G, hidden_dim=OUT, **common)
    for sub in (swq.qformer, swq.global_qformer):
        assert sub.blocks[0].ffn[0].in_features == H, "branch is not operating at hidden_dim width"
    assert n_params(swq) < n_params(swq_wide) / 3

    # token-budget interface and window planning are unaffected by hidden_dim
    for P in [1, 5, 100, 15360]:
        assert swq.plan(P) == swq_wide.plan(P)
        assert swq.num_tokens_for_length(P) == swq_wide.num_tokens_for_length(P)

    # multi-channel + single-channel forward shape/finiteness; patch_pos path
    for P in [3, 17, 256]:
        src4 = torch.randn(2, 3, P, IN)
        out = swq(src4)
        assert out.shape == (2, swq.num_tokens_for_length(P, 3), OUT), out.shape
        assert torch.isfinite(out).all()
    src3 = torch.randn(1, 7, IN)
    with torch.no_grad():
        assert torch.allclose(swq(src3), swq(src3.unsqueeze(1)), atol=0)
    pos = torch.tensor([0.0, 3.5, 10.0, 100.0, 1000.0, 5000.0, 9999.0])
    assert torch.isfinite(swq(src3, patch_pos=pos)).all()

    # gradients connected (both the local and the global branch receive them)
    out = swq(torch.randn(1, 2, 50, IN))
    out.sum().backward()
    for name, sub in [("qformer", swq.qformer), ("global_qformer", swq.global_qformer)]:
        gnorm = sum(p.grad.abs().sum().item() for p in sub.parameters()
                    if p.grad is not None)
        assert gnorm > 0, f"{name} received no gradient"

    # ---------------- 5. config default 768 ----------------
    from chronos_llm.models.chronos_llm_model import ChronosLLMConfig
    cfg = ChronosLLMConfig()
    assert cfg.qformer_hidden_dim == 768, "config default qformer_hidden_dim must be 768"
    cfg2 = ChronosLLMConfig(qformer_hidden_dim=512)
    assert ChronosLLMConfig.from_dict(cfg2.to_dict()).qformer_hidden_dim == 512

    print("QFORMER HIDDEN_DIM CHECKS PASSED")


if __name__ == "__main__":
    main()
