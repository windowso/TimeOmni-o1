"""Pure-CPU unit test of the SlidingWindowQFormer log-linear token budget (no chronos / LLM).

Token budget T = clamp(token_min + b*log2(P*C), token_min, token_max); the window count is
derived from the budget: nw_target = round((T - G) / k) -> W = ceil(P/nw) -> nw = ceil(P/W)
(full coverage, no empty window). Checks:
1. Upper bound + coverage + forward consistency: for any (P,C), tokens <= token_max;
   (nw-1)*W < P <= nw*W; forward output shape == num_tokens_for_length(P,C), values finite.
2. Coarse monotonicity: for well-separated P*C, larger => tokens non-decreasing (locally there is
   +-k quantisation jitter, hence only well-separated points are compared).
3. C sensitivity: fixed P, larger C => tokens non-decreasing and strictly larger at the endpoints
   (the newly introduced C dependence).
4. Soft lower bound: P=1 gives a single window => tokens == G + 1*k < token_min.
5. Global branch: [G | nw*k] order, the local segment is not perturbed by the global branch,
   time sensitivity, gradient connectivity.
"""
import math

import torch

from chronos_llm.models.qformer import SlidingWindowQFormer


def main():
    torch.manual_seed(0)
    k = 4
    tmin, tmax, pcref = 20, 100, 4096
    swq = SlidingWindowQFormer(
        in_dim=32, out_dim=64, queries_per_window=k,
        token_min=tmin, token_max=tmax, token_pc_ref=pcref,
        num_heads=4, num_layers=1,
    )
    G = swq.global_queries  # default 0

    # 1. Upper bound / coverage / forward shape consistency
    print(f"{'P':>8} {'C':>4} {'nw':>4} {'W':>7} {'tokens':>7}")
    for P in [1, 2, 3, 4, 8, 16, 17, 31, 64, 256, 1024, 4096, 65536]:
        for C in [1, 4]:
            nw, W = swq.plan(P, C)
            tokens = swq.num_tokens_for_length(P, C)
            if C == 1:
                print(f"{P:>8} {C:>4} {nw:>4} {W:>7} {tokens:>7}")
            assert tokens == nw * k + G
            assert tokens <= tmax, f"P={P},C={C}: tokens {tokens} > upper bound {tmax}"
            assert (nw - 1) * W < P <= nw * W, f"P={P},C={C}: coverage violated (nw-1)*W<P<=nw*W"
            out = swq(torch.randn(1, C, P, 32))
            assert out.shape == (1, tokens, 64), (P, C, out.shape)
            assert torch.isfinite(out).all()

    # 2. Coarse monotonicity (well-separated P*C, C=1)
    toks = [swq.num_tokens_for_length(P, 1) for P in [1, 16, 64, 256, 1024, 4096, 65536]]
    assert toks == sorted(toks), f"tokens not monotone: {toks}"
    assert toks[-1] == tmax, f"P*C >= pc_ref should clamp to token_max: {toks[-1]} != {tmax}"

    # 3. C sensitivity (fixed P=256, growing C -> tokens non-decreasing, strictly larger at the endpoint)
    cs = [swq.num_tokens_for_length(256, C) for C in [1, 8, 64]]
    assert cs == sorted(cs) and cs[-1] > cs[0], f"tokens did not grow with the channel count: {cs}"

    # 4. Soft lower bound (P=1: a single window -> G + 1*k < token_min)
    assert swq.num_tokens_for_length(1, 1) == G + 1 * k
    assert swq.num_tokens_for_length(1, 1) < tmin

    # 5. Global query branch
    G2 = 8
    common = dict(in_dim=32, out_dim=64, queries_per_window=k,
                  token_min=40, token_max=120, token_pc_ref=4096,
                  num_heads=4, num_layers=1)
    swg = SlidingWindowQFormer(global_queries=G2, **common)
    ref = SlidingWindowQFormer(global_queries=0, **common)
    ref.qformer.load_state_dict(swg.qformer.state_dict())
    ref.channel_embedding.load_state_dict(swg.channel_embedding.state_dict())
    ref.time_pos_embedding.load_state_dict(swg.time_pos_embedding.state_dict())
    # plan now depends on G (local budget = T-G); force ref to use exactly the same window plan
    # as swg, so that only "the local qformer values are not perturbed by the global branch" is
    # tested.
    ref.plan = swg.plan

    for P in [1, 5, 100, 4096]:
        nw, _ = swg.plan(P, 1)
        assert swg.num_tokens_for_length(P, 1) == nw * k + G2

    with torch.no_grad():
        for P in [3, 17, 256]:
            src = torch.randn(2, 3, P, 32)  # B=2, C=3 multi-channel
            out_g = swg(src)
            assert out_g.shape == (2, swg.num_tokens_for_length(P, 3), 64), out_g.shape
            assert torch.isfinite(out_g).all()
            assert torch.allclose(out_g[:, G2:], ref(src), atol=1e-6), f"P={P}: local segment perturbed by the global branch"
        # C=1 / 3-D input compatibility
        src3 = torch.randn(1, 7, 32)
        assert torch.allclose(swg(src3), swg(src3.unsqueeze(1)), atol=0)
        # Global tokens are time-sensitive (flipping should change the global segment)
        src = torch.randn(1, 1, 64, 32)
        assert not torch.allclose(swg(src)[:, :G2], swg(src.flip(2))[:, :G2], atol=1e-4)

    # Gradient reaches global_qformer
    out = swg(torch.randn(1, 2, 50, 32))
    out[:, :G2].sum().backward()
    gnorm = sum(p.grad.abs().sum().item() for p in swg.global_qformer.parameters()
                if p.grad is not None)
    assert gnorm > 0, "global_qformer received no gradient"

    print("BUDGET CHECKS PASSED")


if __name__ == "__main__":
    main()
