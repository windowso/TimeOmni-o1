"""CPU unit test for per-bucket gradient checkpointing of the chronos encode (real chronos-2 +
tiny LLM).

Without checkpointing the backward pass must keep the forward activations of every bucket, so
activation memory grows linearly with the total patch count of the sample (a 210k-patch sample needs
20GB+ for the encode activations alone).
`gradient_checkpointing_enable` also switches on per-bucket / per-sample checkpointing, which brings
the peak down to a single bucket. Verified:
1. Soft-prompt outputs and **chronos/qformer parameter gradients match value-for-value** with
   checkpointing on and off (dropout set to 0).
2. checkpoint is actually invoked (once per bucket + once per sample), and the eval / no_grad paths
   bypass it.
3. gradient_checkpointing_enable/disable correctly toggle `_encode_grad_ckpt`.
"""
import torch
import torch.utils.checkpoint as torch_ckpt

from chronos_llm.tests.test_pretrained_peft import _build_tiny_base


def _zero_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def _inputs():
    torch.manual_seed(7)
    L = 300
    ctx = torch.randn(4, L)            # sample 0: C=3 full length; sample 1: C=1 with 200 valid (left NaN pad)
    ctx[3, : L - 200] = float("nan")
    tl = torch.tensor([L, L, L, 200])
    nch = torch.tensor([3, 1])
    return ctx, tl, nch


def _run(model, use_ckpt: bool):
    """Run one forward+backward of encode->soft prompt; return (soft outputs, dict of parameter gradients)."""
    model.zero_grad(set_to_none=True)
    model._encode_grad_ckpt = use_ckpt
    ctx, tl, nch = _inputs()
    soft = model._encode_history_to_soft_prompt(ctx, tl, nch)
    loss = sum(s.float().pow(2).mean() for s in soft)
    loss.backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None and ("chronos" in n or "history_qformer" in n)
    }
    return [s.detach().clone() for s in soft], grads


def test_ckpt_matches_plain():
    model, _ = _build_tiny_base()
    model.train()
    _zero_dropout(model)
    model.chronos_window = 32   # small chunks -> many chunks, many rows
    model.encode_max_rows = 4   # force multiple buckets

    soft_ref, grads_ref = _run(model, use_ckpt=False)
    soft_ckpt, grads_ckpt = _run(model, use_ckpt=True)

    assert grads_ref, "reference path has no gradients -- test fixture is broken"
    for a, b in zip(soft_ref, soft_ckpt):
        assert torch.allclose(a, b, atol=1e-6), f"outputs differ: max|delta|={(a - b).abs().max()}"
    assert grads_ref.keys() == grads_ckpt.keys(), "the two paths have different sets of parameters with gradients"
    worst = 0.0
    for k in grads_ref:
        d = (grads_ref[k] - grads_ckpt[k]).abs().max().item()
        worst = max(worst, d)
        assert torch.allclose(grads_ref[k], grads_ckpt[k], atol=1e-5), f"{k} gradient differs: max|delta|={d}"
    print(f"checkpoint on/off outputs and gradients match value-for-value OK ({len(grads_ref)} parameters, max|delta|={worst:.2e})")


def test_ckpt_actually_engaged():
    """In training mode checkpoint is really invoked (bucket count + sample count times); eval/no_grad do not invoke it."""
    model, _ = _build_tiny_base()
    model.train()
    _zero_dropout(model)
    model.chronos_window = 32
    model.encode_max_rows = 4
    model._encode_grad_ckpt = True
    ctx, tl, nch = _inputs()

    calls = []
    orig = torch_ckpt.checkpoint

    def spy(fn, *a, **k):
        calls.append(fn)
        return orig(fn, *a, **k)

    torch_ckpt.checkpoint = spy
    try:
        model._encode_history_to_soft_prompt(ctx, tl, nch)
        n_train = len(calls)
        assert n_train > 2, f"unexpected number of checkpoint calls in training mode: {n_train}"

        calls.clear()
        with torch.no_grad():   # no_grad: must not use checkpoint
            model._encode_history_to_soft_prompt(ctx, tl, nch)
        assert not calls, "checkpoint must not be called under no_grad"

        model.eval()            # eval: must not use checkpoint
        model._encode_history_to_soft_prompt(ctx, tl, nch)
        assert not calls, "checkpoint must not be called in eval mode"
    finally:
        torch_ckpt.checkpoint = orig
    print(f"checkpoint active in training mode ({n_train} calls), bypassed in eval/no_grad OK")


def test_enable_disable_wiring():
    model, _ = _build_tiny_base()
    assert model._encode_grad_ckpt is False, "must be off by default"
    model.gradient_checkpointing_enable()
    assert model._encode_grad_ckpt is True, "enable did not switch on encode checkpointing"
    model.gradient_checkpointing_disable()
    assert model._encode_grad_ckpt is False, "disable did not switch off encode checkpointing"
    print("gradient_checkpointing_enable/disable wiring OK")


if __name__ == "__main__":
    test_ckpt_matches_plain()
    test_ckpt_actually_engaged()
    test_enable_disable_wiring()
    print("ALL OK")
