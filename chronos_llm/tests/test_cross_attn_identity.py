"""Identity checks for the gated cross-attention injected into Chronos-2:

1) Chronos2WithCrossAttn loads the original checkpoint correctly (bypassing the broken from_pretrained).
2) With cross_states=None it matches the original Chronos2Model value-for-value (the modification
   does not break the pretrained capability).
3) With cross_states provided but gate=0 it still matches (tanh(0)=0 -> identity).
4) Setting one layer's gate to non-zero and providing cross_states changes the forecast (the
   feedback path is really effective).
"""

import os

import safetensors.torch as st
import torch

from chronos.chronos2 import Chronos2Model
from chronos.chronos2.config import Chronos2CoreConfig
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn

CKPT = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")


def load_base() -> Chronos2Model:
    # likewise bypass the broken from_pretrained: construct directly + manual load_state_dict.
    cfg = Chronos2CoreConfig.from_pretrained(CKPT)
    m = Chronos2Model(cfg)
    m.load_state_dict(st.load_file(CKPT + "/model.safetensors"), strict=False)
    return m.eval()


def main():
    torch.manual_seed(0)
    base = load_base()
    xattn = load_chronos2_with_cross_attn(CKPT)

    B, L = 2, 160
    n_out = 4
    context = torch.randn(B, L)

    with torch.no_grad():
        out_base = base(context=context, num_output_patches=n_out)
        out_x_none = xattn(context=context, num_output_patches=n_out)
    diff_none = (out_base.quantile_preds - out_x_none.quantile_preds).abs().max().item()
    print(f"[1/2] max|base - xattn(cross=None)| = {diff_none:.3e}")
    assert diff_none < 1e-4, "with cross_states=None the output must match the original model"

    d_model = xattn.config.d_model
    M = 16
    cross = torch.randn(B, M, d_model)
    cross_mask = torch.ones(B, M)

    with torch.no_grad():
        out_x_gate0 = xattn(
            context=context, num_output_patches=n_out,
            cross_states=cross, cross_states_mask=cross_mask,
        )
    diff_gate0 = (out_base.quantile_preds - out_x_gate0.quantile_preds).abs().max().item()
    print(f"[3] max|base - xattn(cross, gate=0)| = {diff_gate0:.3e}")
    assert diff_gate0 < 1e-4, "with gate=0 the output must be identical even when cross_states is provided"

    with torch.no_grad():
        xattn.encoder.block[0].cross_attn.gate.fill_(1.0)
        out_x_gate1 = xattn(
            context=context, num_output_patches=n_out,
            cross_states=cross, cross_states_mask=cross_mask,
        )
    diff_gate1 = (out_base.quantile_preds - out_x_gate1.quantile_preds).abs().max().item()
    print(f"[4] max|base - xattn(cross, gate=1 @layer0)| = {diff_gate1:.3e}")
    assert diff_gate1 > 1e-3, "after opening the gate the forecast must change with cross_states"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
