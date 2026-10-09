"""Checks for the gated cross-attn injected into TimesFM-3.0 (the arm of the backbone ablation that
swaps the TSFM).

The chronos-side counterparts are test_cross_attn_identity + test_zero_out_proj_floor:

1) the weights load correctly (nothing is missing apart from the newly added cross_attn);
2) with cross_states=None the output is **identical value for value** to the original TimesFM3Torch
   (the modification does not touch the pretrained capability);
3) with cross_states supplied but gate=0 it is still identical (tanh(0)=0 => identity);
4) with gate=1.0 and a zero-initialised out_proj it is still identical (the identity fallback that
   production training relies on);
5) with gate!=0 and a non-zero out_proj the output really does change (the feedback path is open);
6) no crossing over between samples: changing the cross_states of one sample in the batch leaves the
   other sample's output identical value for value (the variate-axis flattening uses
   repeat_interleave rather than repeat, and this is the check that guards it);
7) the patch representations from encode have the right shape and can serve as the Q-former's KV;
8) set_cross_attn_layers really restricts the injected layer subset;
9) gradient connectivity: with gate!=0 the cross_attn parameters receive finite gradients;
10) B=1 / C=1 compatibility (the minimal-configuration regression case).
"""

import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO, os.path.join(_REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from timesfm3.torch.model import TimesFM3Torch  # noqa: E402

from chronos_llm.models.cross_attn_timesfm import load_timesfm3_with_cross_attn  # noqa: E402

CKPT = os.environ.get("TIMESFM3_PATH", "checkpoints/TimesFM3.0")


def _zero_out_proj(model) -> int:
    """Zero the cross-attn out_proj -- the same matching rule as train.py::_apply_zero_cross_out_proj."""
    n = 0
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if ".cross_attn.attn.out_proj." in ("." + name):
                prm.zero_()
                n += 1
    return n


def _set_all_gates(model, val: float) -> int:
    n = 0
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if name.endswith(".cross_attn.gate"):
                prm.fill_(val)
                n += 1
    return n


def main() -> None:
    torch.manual_seed(0)

    base = TimesFM3Torch.from_pretrained(CKPT).eval()
    xattn = load_timesfm3_with_cross_attn(CKPT).eval()
    d_model = xattn.transformer_config.transformer.model_dims
    n_layers = len(xattn.transformer_stack.layers)
    print(f"[load] d_model={d_model} layers={n_layers}")

    B, C, L, H = 2, 3, 256, 32
    context = torch.randn(B, C, L)
    M = 16
    cross = torch.randn(B, M, d_model)
    cross_mask = torch.ones(B, M)

    # ---- 1/2: identity with cross_states=None ----
    with torch.no_grad():
        out_base = base.decode(target=context, horizon=H)
        out_none = xattn.forecast(context, H)
    d_none = (out_base - out_none).abs().max().item()
    print(f"[1] max|base - xattn(cross=None)| = {d_none:.3e}")
    assert d_none == 0.0, "with cross_states=None the output must match the original TimesFM3 value for value"

    # ---- 3: gate=0 (the default) + cross_states supplied, still identity ----
    with torch.no_grad():
        out_g0 = xattn.forecast(context, H, cross_states=cross, cross_states_mask=cross_mask)
    d_g0 = (out_base - out_g0).abs().max().item()
    print(f"[2] gate=0 with cross_states supplied: max|delta| = {d_g0:.3e}")
    assert d_g0 == 0.0, "with gate=0, tanh(0)=0, so it must be the identity"

    # ---- 4: gate=1.0 + zero-initialised out_proj, still identity (the production identity fallback) ----
    n_gate = _set_all_gates(xattn, 1.0)
    n_op = _zero_out_proj(xattn)
    with torch.no_grad():
        out_zop = xattn.forecast(context, H, cross_states=cross, cross_states_mask=cross_mask)
    d_zop = (out_base - out_zop).abs().max().item()
    print(f"[3] gate=1.0 + zero out_proj ({n_gate} gates / {n_op} out_proj tensors): max|delta| = {d_zop:.3e}")
    assert d_zop == 0.0, "the zero-out_proj identity fallback has been broken"

    # ---- 5: non-zero out_proj => the feedback really changes the output ----
    with torch.no_grad():
        for name, prm in xattn.named_parameters():
            if ".cross_attn.attn.out_proj.weight" in ("." + name):
                prm.normal_(0, 0.02)
    with torch.no_grad():
        out_on = xattn.forecast(context, H, cross_states=cross, cross_states_mask=cross_mask)
    d_on = (out_base - out_on).abs().max().item()
    print(f"[4] gate=1.0 + non-zero out_proj: max|delta| = {d_on:.3e}")
    assert d_on > 1e-4, "the feedback path is not working"

    # ---- 6: no crossing over between samples ----
    cross_b = cross.clone()
    cross_b[1] = torch.randn(M, d_model)          # change sample 1 only
    with torch.no_grad():
        out_b = xattn.forecast(context, H, cross_states=cross_b, cross_states_mask=cross_mask)
    d_keep = (out_on[0] - out_b[0]).abs().max().item()
    d_change = (out_on[1] - out_b[1]).abs().max().item()
    print(f"[5] changing only sample 1's cross: sample 0 max|delta|={d_keep:.3e} (should be 0), sample 1 max|delta|={d_change:.3e} (should be >0)")
    assert d_keep == 0.0, "cross_states crossed over between samples -- check the expansion order of repeat_interleave"
    assert d_change > 1e-6, "cross_states changed but the output did not"

    # ---- 7: encode shape ----
    hid = xattn.encode(context)
    n_ctx_patch = -(-L // xattn.input_patch_len)
    print(f"[6] encode -> {tuple(hid.shape)} (expected ({B}, {C}, {n_ctx_patch}, {d_model}))")
    assert hid.shape == (B, C, n_ctx_patch, d_model), hid.shape
    assert torch.isfinite(hid).all()

    # ---- 8: injected layer subset ----
    ids = xattn.set_cross_attn_layers("last6")
    assert ids == list(range(n_layers - 6, n_layers)), ids
    with torch.no_grad():
        out_last6 = xattn.forecast(context, H, cross_states=cross, cross_states_mask=cross_mask)
    d_sub = (out_on - out_last6).abs().max().item()
    print(f"[7] set_cross_attn_layers('last6') -> {ids[0]}..{ids[-1]}, max|delta| vs injecting everywhere = {d_sub:.3e} (should be >0)")
    assert d_sub > 1e-6, "restricting the injected layers did not change the output"
    xattn.set_cross_attn_layers("all")

    # ---- 9: gradient connectivity ----
    xattn.train()
    out_g = xattn.forecast(context, H, cross_states=cross, cross_states_mask=cross_mask)
    out_g.float().square().mean().backward()
    gates = [p for n, p in xattn.named_parameters() if n.endswith(".cross_attn.gate")]
    xattn_ps = [p for n, p in xattn.named_parameters() if ".cross_attn." in ("." + n)]
    n_gate_grad = sum(1 for p in gates if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs() > 0)
    n_x_grad = sum(1 for p in xattn_ps if p.grad is not None and torch.isfinite(p.grad).all())
    print(f"[8] gradients: {n_gate_grad}/{len(gates)} gates have a non-zero gradient, {n_x_grad}/{len(xattn_ps)} cross_attn parameters have a finite gradient")
    assert n_gate_grad == len(gates), "the gates did not receive gradients"
    assert n_x_grad == len(xattn_ps), "some cross_attn parameters did not receive gradients"
    xattn.zero_grad(set_to_none=True)
    xattn.eval()

    # ---- 10: the minimal B=1 / C=1 configuration ----
    ctx1 = torch.randn(1, 1, 128)
    cross1 = torch.randn(1, M, d_model)
    with torch.no_grad():
        o_base1 = base.decode(target=ctx1, horizon=H)
        o_x1 = xattn.forecast(ctx1, H)                                   # no feedback
        o_x1c = xattn.forecast(ctx1, H, cross_states=cross1,
                               cross_states_mask=torch.ones(1, M))        # with feedback
    d1 = (o_base1 - o_x1).abs().max().item()
    d1c = (o_base1 - o_x1c).abs().max().item()
    h1 = xattn.encode(ctx1)
    print(f"[9] B=1/C=1: no feedback max|delta|={d1:.3e} (should be 0), with feedback max|delta|={d1c:.3e} (should be >0), encode={tuple(h1.shape)}")
    assert d1 == 0.0 and d1c > 1e-4
    assert h1.shape == (1, 1, -(-128 // xattn.input_patch_len), d_model)

    print("\nTIMESFM CROSS-ATTN CHECKS PASSED")


if __name__ == "__main__":
    main()
