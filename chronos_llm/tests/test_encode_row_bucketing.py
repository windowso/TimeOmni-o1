"""CPU unit test of group-bucketed chronos sliding-window encoding (real chronos-2 + tiny LLM).

GroupSelfAttention's mask/score is (T, R, R) and grows quadratically with the total folded row
count R, while groups never interact anyway (block-diagonal mask) -- bucketing whole groups makes
the cost linear in R. Checks:
1. Bucketed output (encode_max_rows small enough to force several buckets) is **value-identical**
   to a single large-batch encode (mixed multi-channel samples + C=1 single channel + bucket size
   that does not divide evenly).
2. It really splits into several encode calls, each bucket ~ the cap (exceeding it by at most C-1 rows).
3. Gradients flow through the bucketed path (chronos parameters get non-zero gradients).
"""
import math

import torch

from chronos_llm.tests.test_pretrained_peft import _build_tiny_base


def _soft(model, ctx, tl, nch, max_rows):
    model.encode_max_rows = max_rows
    return model._encode_history_to_soft_prompt(ctx, tl, nch)


def _spy_encode(model, calls):
    orig = model.chronos.encode

    def spy(*a, **k):
        rows = (k["context"] if "context" in k else a[0]).shape[0]
        calls.append(rows)
        return orig(*a, **k)

    model.chronos.encode = spy
    return orig


def test_bucketed_equals_single():
    model, _ = _build_tiny_base()
    model.eval()
    model.chronos_window = 32  # small chunks -> many chunks, many rows
    L = 300
    torch.manual_seed(1)
    ctx = torch.randn(4, L)            # sample 0: C=3 full length; sample 1: C=1 with 200 valid (left NaN pad)
    ctx[3, : L - 200] = float("nan")
    tl = torch.tensor([L, L, L, 200])
    nch = torch.tensor([3, 1])
    # Rows: sample 0 = 3ch x ceil(300/32)=10 chunks = 30 rows (10 groups x 3); sample 1 = 7 rows (7 groups x 1).
    with torch.no_grad():
        ref = _soft(model, ctx, tl, nch, 0)        # single large batch (37 rows)
        calls = []
        orig = _spy_encode(model, calls)
        b4 = _soft(model, ctx, tl, nch, 4)         # force several buckets
        model.chronos.encode = orig
        b7 = _soft(model, ctx, tl, nch, 7)         # non-dividing bucket size
    assert len(calls) > 1, "encode_max_rows=4 should split into several buckets"
    assert max(calls) <= 4 + 2, f"a bucket may exceed the cap by at most C-1=2 rows: {calls}"  # mr + (C-1)
    for a, b in zip(ref, b4):
        assert torch.allclose(a, b, atol=1e-5), f"bucketed != single: max|delta|={(a - b).abs().max()}"
    for a, b in zip(ref, b7):
        assert torch.allclose(a, b, atol=1e-5), f"bucketed(7) != single: max|delta|={(a - b).abs().max()}"
    print(f"bucketed == single encode OK (bucket rows {calls}, max|delta|="
          f"{max((a - b).abs().max().item() for a, b in zip(ref, b4)):.2e})")


def test_single_channel_compat():
    """C=1 / n_channels=None compatibility: bucketed equals single."""
    model, _ = _build_tiny_base()
    model.eval()
    model.chronos_window = 16
    torch.manual_seed(2)
    ctx = torch.randn(2, 100)  # 2 single-channel samples, ceil(100/16)=7 chunks each
    tl = torch.tensor([100, 100])
    with torch.no_grad():
        ref = _soft(model, ctx, tl, None, 0)
        bkt = _soft(model, ctx, tl, None, 3)
    for a, b in zip(ref, bkt):
        assert torch.allclose(a, b, atol=1e-5)
    print("C=1 / n_channels=None bucketing compatibility OK")


def test_gradient_through_buckets():
    model, _ = _build_tiny_base()
    model.train()
    model.chronos_window = 32
    model.encode_max_rows = 4
    ctx = torch.randn(2, 200)
    soft = model._encode_history_to_soft_prompt(ctx, torch.tensor([200, 200]), None)
    loss = sum(s.float().pow(2).mean() for s in soft)
    loss.backward()
    probe = model.chronos.input_patch_embedding
    g = [p.grad for p in probe.parameters() if p.grad is not None]
    assert g and any(x.abs().sum() > 0 for x in g), "chronos gradient not connected through the bucketed path"
    print("gradient through bucketed path OK")


if __name__ == "__main__":
    test_bucketed_equals_single()
    test_single_channel_compat()
    test_gradient_through_buckets()
    print("ALL OK")
