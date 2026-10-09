"""Verify the "floor-preserving" property of zero-initializing the cross-attn out_proj:

Even with gate=1.0 (fully open) and cross_states supplied, as long as cross_attn.attn.out_proj is
zeroed, the feedback branch outputs exactly 0 => chronos predictions equal the original zero-shot
(cross_states=None) value for value. This is the core guarantee of the
--zero_cross_out_proj + --gate_init 1.0 combination: the starting point is the zero-shot floor, and
the fully open gate merely lets out_proj receive full gradients to learn from, without disturbing
the starting point.
"""
import os

import torch

from chronos_llm.tests.test_cross_attn_identity import load_base
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn

CKPT = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")


def main():
    torch.manual_seed(0)
    base = load_base()
    xattn = load_chronos2_with_cross_attn(CKPT)

    # Replicate the two steps of train.py: warm-start gate at 1.0 + zero out_proj.
    n_gate = n_op = 0
    with torch.no_grad():
        for n, p in xattn.named_parameters():
            if n.endswith(".cross_attn.gate"):
                p.fill_(1.0); n_gate += 1
            if ".cross_attn.attn.out_proj." in ("." + n):
                p.zero_(); n_op += 1
    print(f"tensors with gate set to 1.0={n_gate}, out_proj tensors zeroed={n_op}")
    assert n_gate > 0 and n_op > 0, "gate and out_proj parameters should be matched (check the naming)"

    B, L, n_out, M = 2, 160, 4, 16
    context = torch.randn(B, L)
    cross = torch.randn(B, M, xattn.config.d_model)
    cross_mask = torch.ones(B, M)
    with torch.no_grad():
        out_base = base(context=context, num_output_patches=n_out)
        out_x = xattn(context=context, num_output_patches=n_out,
                      cross_states=cross, cross_states_mask=cross_mask)
    diff = (out_base.quantile_preds - out_x.quantile_preds).abs().max().item()
    print(f"max|zero-shot - xattn(cross, gate=1, out_proj=0)| = {diff:.3e}")
    assert diff < 1e-4, "with out_proj=0 the feedback contribution must be exactly 0 => identical to zero-shot value for value"

    # Control: with a non-zero (random) out_proj + gate=1 the prediction should change clearly
    # (proving that out_proj is indeed what controls the contribution).
    with torch.no_grad():
        torch.nn.init.normal_(xattn.encoder.block[0].cross_attn.attn.out_proj.weight, std=0.1)
        out_x2 = xattn(context=context, num_output_patches=n_out,
                       cross_states=cross, cross_states_mask=cross_mask)
    diff2 = (out_base.quantile_preds - out_x2.quantile_preds).abs().max().item()
    print(f"max|zero-shot - xattn(out_proj!=0)| = {diff2:.3e} (should be >1e-3)")
    assert diff2 > 1e-3, "with a non-zero out_proj the feedback should change the prediction"
    print("ZERO-OUT-PROJ FLOOR CHECK PASSED")


if __name__ == "__main__":
    main()
