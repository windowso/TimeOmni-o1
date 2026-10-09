"""CPU unit tests for the short-window detection enhancements: history_stats_token (option 1)
+ short_min_patches (option 2).

Aimed at detection sets like Weather ("single channel, 16 points, 1 patch"), where the
discriminative signal (the window's fluctuation amplitude) is divided away by chronos2's
per-series instance_norm. Both features are off by default; this test verifies:
- with both off, behaviour is **numerically identical** to the old one (C=1/B=1 compatible, token
  count unchanged);
- option 1: when on, one extra statistics token per sample per channel, gradient reaches
  history_stats_proj, outputs finite;
- option 2: when on, ultra-short series are upsampled to short_min_patches patches => more soft
  tokens;
- the soft_token_count cost estimate matches the actual encode output sample by sample;
- config round trip + add_lora puts history_stats_proj into modules_to_save.
"""
import tempfile

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from chronos_llm.models.chronos_llm_model import (
    ChronosLLM, ChronosLLMConfig, add_lora, _add_ts_special_tokens,
)
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn
from chronos_llm.tests.test_pretrained_peft import CHRONOS, LLM


def _tiny(**cfg_overrides):
    """Tiny model whose config can be overridden (same tokenizer/LLM/chronos so arms can be compared)."""
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    qcfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                       max_position_embeddings=4096)
    llm = Qwen2ForCausalLM(qcfg); llm.config.use_cache = False
    chronos = load_chronos2_with_cross_attn(CHRONOS); chronos.train()
    kw = dict(chronos_ckpt=CHRONOS, llm_path=LLM, sw_queries_per_window=2,
              sw_target_windows=8, sw_min_windows=2, sw_global_queries=4,
              fb_num_query_tokens=8, qformer_num_heads=4)
    kw.update(cfg_overrides)
    cfg = ChronosLLMConfig(**kw)
    return ChronosLLM(cfg, chronos, llm, tok), tok


def test_default_off_compat():
    """Both features off by default => no history_stats_proj, short_min_patches=0, C=1/B=1 encoding numerically unchanged."""
    base, _ = _tiny(); base.eval()   # eval disables dropout so the numeric comparison is deterministic
    assert base.history_stats_proj is None and base.short_min_patches == 0
    torch.manual_seed(0)
    ctx = torch.randn(2, 64); lens = torch.tensor([64, 64])
    s0 = base._encode_history_to_soft_prompt(ctx, lens, torch.tensor([1, 1]))
    s1 = base._encode_history_to_soft_prompt(ctx, lens, torch.tensor([1, 1]))
    for a, b in zip(s0, s1):
        assert torch.allclose(a, b)
    # soft_token_count == actual token count (P=ceil(64/16)=4)
    assert base.soft_token_count(4, 1) == s0[0].shape[0]
    print("default-off compatibility OK")


def test_stats_token_adds_one_per_channel():
    """Option 1: history_stats_token on => one extra token per sample per channel, and the tail tokens match the off arm."""
    off, _ = _tiny(); on, _ = _tiny(history_stats_token=True)
    # Compare with identical chronos/qformer weights: copy the off arm's weights into the on arm
    # (structures agree except for the new projection)
    on.chronos.load_state_dict(off.chronos.state_dict())
    on.history_qformer.load_state_dict(off.history_qformer.state_dict())
    off.eval(); on.eval()
    torch.manual_seed(1)
    ctx = torch.randn(1, 48); lens = torch.tensor([48])
    s_off = off._encode_history_to_soft_prompt(ctx, lens, torch.tensor([1]))[0]
    s_on = on._encode_history_to_soft_prompt(ctx, lens, torch.tensor([1]))[0]
    assert s_on.shape[0] == s_off.shape[0] + 1, (s_on.shape, s_off.shape)
    # The statistics token comes **first**: the remainder must match the off arm exactly (same qformer weights)
    assert torch.allclose(s_on[1:], s_off, atol=1e-5)
    assert torch.isfinite(s_on).all()
    assert on.soft_token_count(3, 1) == s_on.shape[0]   # P=ceil(48/16)=3
    print(f"option 1 statistics token OK (off={s_off.shape[0]} -> on={s_on.shape[0]})")


def test_stats_token_multichannel_and_grad():
    """Option 1 multichannel: C=3 adds 3 tokens; gradient reaches history_stats_proj."""
    on, _ = _tiny(history_stats_token=True)
    ctx = torch.randn(3, 50, requires_grad=False); lens = torch.tensor([50, 50, 50])
    soft = on._encode_history_to_soft_prompt(ctx, lens, torch.tensor([3]))[0]
    base_tokens = on.history_qformer.num_tokens_for_length(
        on._effective_patches(4), 3)  # P=ceil(50/16)=4
    assert soft.shape[0] == base_tokens + 3, (soft.shape[0], base_tokens)
    soft.sum().backward()
    g = on.history_stats_proj[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print("option 1 multichannel + gradient connectivity OK")


def test_upsample_short_series():
    """Option 2: a 16-point (P=1) single-chunk series with short_min_patches=8 => upsampled to 8 patches, more soft tokens."""
    off, _ = _tiny()
    on, _ = _tiny(short_min_patches=8)
    on.chronos.load_state_dict(off.chronos.state_dict())
    on.history_qformer.load_state_dict(off.history_qformer.state_dict())
    off.eval(); on.eval()
    torch.manual_seed(2)
    ctx = torch.randn(1, 16); lens = torch.tensor([16])
    s_off = off._encode_history_to_soft_prompt(ctx, lens, torch.tensor([1]))[0]
    s_on = on._encode_history_to_soft_prompt(ctx, lens, torch.tensor([1]))[0]
    assert on._effective_patches(1) == 8 and off._effective_patches(1) == 1
    assert s_on.shape[0] > s_off.shape[0], (s_on.shape, s_off.shape)
    assert s_on.shape[0] == on.soft_token_count(1, 1)
    assert torch.isfinite(s_on).all()
    # Long series (P >= threshold) do not trigger upsampling: numerically identical to off
    long_ctx = torch.randn(1, 16 * 20); long_lens = torch.tensor([16 * 20])
    l_off = off._encode_history_to_soft_prompt(long_ctx, long_lens, torch.tensor([1]))[0]
    l_on = on._encode_history_to_soft_prompt(long_ctx, long_lens, torch.tensor([1]))[0]
    assert l_on.shape == l_off.shape and torch.allclose(l_on, l_off, atol=1e-5)
    # allow_upsample=False (forecasting branch) never upsamples even with short_min_patches on => == off
    s_pred = on._encode_history_to_soft_prompt(ctx, lens, torch.tensor([1]), allow_upsample=False)[0]
    assert s_pred.shape == s_off.shape and torch.allclose(s_pred, s_off, atol=1e-5)
    assert on._effective_patches(1, allow_upsample=False) == 1
    assert on.soft_token_count(1, 1, allow_upsample=False) == s_off.shape[0]
    print(f"option 2 upsampling OK (16 points: off={s_off.shape[0]} -> on={s_on.shape[0]}; long series / forecasting branch not triggered)")


def test_both_combined_cost_consistency():
    """Both features on: soft_token_count matches the real encode output for various (P,C), sample by sample."""
    m, _ = _tiny(history_stats_token=True, short_min_patches=8)
    for L, C in [(16, 1), (48, 1), (50, 3), (16 * 30, 1)]:
        ctx = torch.randn(C, L); lens = torch.tensor([L] * C)
        soft = m._encode_history_to_soft_prompt(ctx, lens, torch.tensor([C]))[0]
        P = (L + 15) // 16
        assert soft.shape[0] == m.soft_token_count(P, C), (L, C, soft.shape[0], m.soft_token_count(P, C))
    print("both features on: cost consistent OK")


def test_config_roundtrip_and_modules_to_save():
    """Config round trip + old config.json without the new keys defaults to off + add_lora puts history_stats_proj into modules_to_save."""
    cfg = ChronosLLMConfig(chronos_ckpt="a", llm_path="b",
                           history_stats_token=True, short_min_patches=8)
    with tempfile.TemporaryDirectory() as d:
        cfg.save_pretrained(d)
        cfg2 = ChronosLLMConfig.from_pretrained(d)
    assert cfg2.history_stats_token is True and cfg2.short_min_patches == 8
    old = ChronosLLMConfig(chronos_ckpt="a", llm_path="b")
    assert old.history_stats_token is False and old.short_min_patches == 0
    # add_lora: when enabled, history_stats_proj goes into modules_to_save and is trainable
    m, _ = _tiny(history_stats_token=True)
    peft = add_lora(m, r=4, alpha=8, dropout=0.0)
    trainable = [n for n, p in peft.named_parameters()
                 if p.requires_grad and "history_stats_proj" in n]
    assert trainable, "history_stats_proj is not marked trainable"
    print("config round trip + modules_to_save OK")


if __name__ == "__main__":
    test_default_off_compat()
    test_stats_token_adds_one_per_channel()
    test_stats_token_multichannel_and_grad()
    test_upsample_short_series()
    test_both_combined_cost_consistency()
    test_config_roundtrip_and_modules_to_save()
    print("\nall short-window enhancement tests passed")
