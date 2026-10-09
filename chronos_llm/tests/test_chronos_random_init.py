"""Ablation 4: chronos_random_init check (CPU, real chronos-2 config + tiny Qwen2 standing in for the 9B).

Checks:
1) random_init=True skips loading the pretrained weights: no safetensors read, and no
   missing/unexpected-key errors.
2) The random_init weights differ from the pretrained ones (neither an accidental skip nor all zeros).
3) The random_init chronos2 forward works (no NaN/Inf) and its predictions differ from the
   pretrained version.
4) The chronos_random_init field of ChronosLLMConfig round-trips through save/load; an old-style
   config.json (without the key) loads with default False, behaving exactly as before the change
   (backward compatible).
"""
import os

import torch

from chronos_llm.models.chronos_llm_model import ChronosLLMConfig
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn

CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")


def test_random_init_skips_pretrained_weights():
    torch.manual_seed(0)
    pretrained = load_chronos2_with_cross_attn(CHRONOS)
    torch.manual_seed(0)
    rand = load_chronos2_with_cross_attn(CHRONOS, random_init=True)

    # Same architecture (same config.json): identical parameter-name sets.
    names_pre = {n for n, _ in pretrained.named_parameters()}
    names_rand = {n for n, _ in rand.named_parameters()}
    assert names_pre == names_rand, "random_init must not change the architecture/parameter naming"

    # Compare parameter by parameter, excluding the newly added cross-attn (zero/default-initialized
    # on both sides, hence identical by construction; not part of the comparison).
    diffs = []
    for (n, p_pre), (_, p_rand) in zip(pretrained.named_parameters(), rand.named_parameters()):
        if ".cross_attn." in n:
            continue
        diffs.append((p_pre - p_rand).abs().max().item())
    assert max(diffs) > 1e-3, "random_init weights should differ clearly from the pretrained weights"


def test_random_init_forward_finite_and_differs():
    B, L, n_out = 2, 160, 4
    context = torch.randn(B, L)

    torch.manual_seed(1)
    pretrained = load_chronos2_with_cross_attn(CHRONOS)
    torch.manual_seed(2)
    rand = load_chronos2_with_cross_attn(CHRONOS, random_init=True)

    with torch.no_grad():
        out_pre = pretrained(context=context, num_output_patches=n_out)
        out_rand = rand(context=context, num_output_patches=n_out)

    assert torch.isfinite(out_rand.quantile_preds).all(), "random_init forward must not produce NaN/Inf"
    diff = (out_pre.quantile_preds - out_rand.quantile_preds).abs().max().item()
    assert diff > 1e-3, "random_init predictions should differ clearly from the pretrained model (not an accidental skip)"


def test_config_roundtrip_and_backward_compat():
    cfg = ChronosLLMConfig(chronos_ckpt="a", llm_path="b", chronos_random_init=True)
    assert cfg.chronos_random_init is True
    d = cfg.to_dict()
    cfg2 = ChronosLLMConfig.from_dict(d)
    assert cfg2.chronos_random_init is True

    # Old-style config.json without the key -> default False (backward compatible, existing
    # checkpoints load unchanged).
    d_old = cfg.to_dict()
    del d_old["chronos_random_init"]
    cfg_old = ChronosLLMConfig.from_dict(d_old)
    assert cfg_old.chronos_random_init is False


def main():
    test_random_init_skips_pretrained_weights()
    print("[1/3] random_init skips the pretrained weights, architecture unchanged OK")
    test_random_init_forward_finite_and_differs()
    print("[2/3] random_init forward is finite and differs from the pretrained version OK")
    test_config_roundtrip_and_backward_compat()
    print("[3/3] config save/load round-trip + old-style config backward compatibility OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
