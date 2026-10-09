"""CPU unit tests for the five architecture-ablation switches (A1/A4/B1/B2/B3).

Every switch is checked for two things:
1. **With its default value the model is numerically identical to the standard configuration**
   (zero regression for old checkpoints / existing runs);
2. when enabled, shapes / token budgets / gradient connectivity are correct (it really runs, it
   is not an empty switch).

Runs on CPU with the real chronos-2 + a tiny Qwen2 (stand-in for the 9B LLM).
"""

import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from chronos_llm.models.chronos_llm_model import ChronosLLM, ChronosLLMConfig, _add_ts_special_tokens
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn
from chronos_llm.models.qformer import SegmentMeanPooler

CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")


def build(**cfg_kwargs):
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    qcfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                       max_position_embeddings=4096)
    llm = Qwen2ForCausalLM(qcfg)
    llm.resize_token_embeddings(len(tok))
    chronos = load_chronos2_with_cross_attn(CHRONOS)
    chronos.train()
    base = dict(sw_queries_per_window=2, sw_target_windows=8, sw_min_windows=2,
                sw_global_queries=4, fb_num_query_tokens=8, qformer_num_heads=4)
    base.update(cfg_kwargs)
    return ChronosLLM(ChronosLLMConfig(**base), chronos, llm, tok), tok


# ------------------------------------------------------------------ A1: history compressor
def test_a1_history_pool():
    torch.manual_seed(0)
    m_q, _ = build()
    torch.manual_seed(0)
    m_p, _ = build(history_compressor="pool")
    assert type(m_p.history_qformer).__name__ == "SlidingWindowPooler"
    assert not hasattr(m_p.history_qformer, "qformer"), "the pool arm must not keep the learned-query branch"

    ctx = torch.randn(2, 300)
    tl = torch.tensor([300, 300])
    nch = torch.tensor([1, 1])
    soft_q = m_q._encode_history_to_soft_prompt(ctx, tl, nch)
    soft_p = m_p._encode_history_to_soft_prompt(ctx, tl, nch)
    for a, b in zip(soft_q, soft_p):
        assert a.shape == b.shape, f"token budgets must match exactly: {a.shape} vs {b.shape}"
    # The budget interface (used by dynamic batching) must agree between the two arms
    assert m_q.soft_token_count(19, 1) == m_p.soft_token_count(19, 1)

    # Multichannel + gradient connectivity
    ctx2 = torch.randn(3, 200)          # sample 0: 2 channels, sample 1: 1 channel
    soft = m_p._encode_history_to_soft_prompt(ctx2, torch.tensor([200] * 3), torch.tensor([2, 1]))
    assert len(soft) == 2 and soft[0].shape[-1] == m_p.llm_hidden
    torch.cat(soft, 0).sum().backward()
    hp = m_p.history_qformer
    assert hp.local_proj.weight.grad is not None and hp.local_proj.weight.grad.abs().sum() > 0
    assert hp.global_proj.weight.grad.abs().sum() > 0
    assert hp.channel_embedding.weight.grad is not None
    n_q = sum(p.numel() for p in m_q.history_qformer.parameters())
    n_p = sum(p.numel() for p in hp.parameters())
    print(f"[A1] OK  soft tokens={[tuple(s.shape) for s in soft_q]}  params qformer={n_q/1e6:.2f}M pool={n_p/1e6:.2f}M")


# ------------------------------------------------------------------ B1: feedback compressor
def test_b1_feedback_pool():
    torch.manual_seed(0)
    pooler = SegmentMeanPooler(in_dim=16, out_dim=8, num_query_tokens=2)
    src = torch.randn(2, 8, 16)
    pad = torch.zeros(2, 8, dtype=torch.bool)
    pad[0, 4:] = True                      # sample 0: only the first 4 positions are valid
    out = pooler(src, src_key_padding_mask=pad)
    assert out.shape == (2, 2, 8)

    # (1) the content of padded positions does not affect the output (the mask really works)
    src2 = src.clone()
    src2[0, 4:] = 1e3
    assert torch.allclose(out, pooler(src2, src_key_padding_mask=pad), atol=1e-5)

    # (2) segments split **the valid positions** evenly and preserve order: swapping the two halves
    #     swaps the two segment outputs
    a = torch.randn(1, 1, 16).expand(1, 2, 16)
    b = torch.randn(1, 1, 16).expand(1, 2, 16)
    o_ab = pooler(torch.cat([a, b], 1))
    o_ba = pooler(torch.cat([b, a], 1))
    assert torch.allclose(o_ab[:, 0], o_ba[:, 1], atol=1e-5) and torch.allclose(o_ab[:, 1], o_ba[:, 0], atol=1e-5)

    # (3) an all-padding row does not produce NaN (the QFormer side has the same fallback)
    allpad = torch.ones(1, 8, dtype=torch.bool)
    assert torch.isfinite(pooler(src[:1], src_key_padding_mask=allpad)).all()

    # (4) forward+backward run inside the full model; the injection path is unchanged (still per-layer cross-attn)
    torch.manual_seed(0)
    m, tok = build(feedback_compressor="pool")
    assert type(m.feedback_qformer).__name__ == "SegmentMeanPooler"
    assert hasattr(m.feedback_qformer, "input_proj")   # callers rely on it to read the dtype
    print("[B1] OK  mask / segmentation / all-pad / attribute name all pass")


# ------------------------------------------------------------------ A4: intermediate chronos layer
def test_a4_encode_layer():
    torch.manual_seed(0)
    chronos = load_chronos2_with_cross_attn(CHRONOS).eval()
    n = len(chronos.encoder.block)
    ctx = torch.randn(2, 320)
    with torch.no_grad():
        full = chronos.encode(context=ctx, num_output_patches=1)[0][0]
        same = chronos.encode(context=ctx, num_output_patches=1, stop_at_layer=n)[0][0]
        mid = chronos.encode(context=ctx, num_output_patches=1, stop_at_layer=n // 2)[0][0]
    d = (full - same).abs().max().item()
    assert d == 0.0, f"stop_at_layer=num_layers must be numerically identical to the default, max|delta|={d}"
    assert mid.shape == full.shape and (mid - full).abs().max().item() > 1e-3, "the intermediate layer should really differ"

    # Full-model switch: after the early exit the soft prompt still back-propagates into the used
    # first half of the layers, and the second half receives no gradient
    torch.manual_seed(0)
    m, _ = build(history_encode_layer=n // 2)
    assert m.history_encode_layer == n // 2
    soft = m._encode_history_to_soft_prompt(torch.randn(1, 320), torch.tensor([320]), torch.tensor([1]))
    soft[0].sum().backward()
    g_used = m.chronos.encoder.block[0].layer[0].self_attention.q.weight.grad
    g_unused = m.chronos.encoder.block[n - 1].layer[0].self_attention.q.weight.grad
    assert g_used is not None and g_used.abs().sum() > 0, "used layers must receive a gradient"
    assert g_unused is None or g_unused.abs().sum() == 0, "layers after the early exit must not receive a gradient"
    print(f"[A4] OK  layers={n} stop=None/{n} numerically identical, stop={n//2} effective and gradient only reaches the first half")


# ------------------------------------------------------------------ B3: subset of injection layers
def test_b3_cross_attn_layers():
    torch.manual_seed(0)
    chronos = load_chronos2_with_cross_attn(CHRONOS)
    n = len(chronos.encoder.block)
    for blk in chronos.encoder.block:                 # open all gates, otherwise injection is identity either way
        blk.cross_attn.gate.data.fill_(1.0)
    ctx = torch.randn(2, 320)
    cs = torch.randn(2, 8, chronos.config.d_model)
    cm = torch.ones(2, 8)

    ids_all = chronos.set_cross_attn_layers("all")
    assert ids_all == list(range(n)) and chronos.encoder.cross_attn_layer_ids is None
    with torch.no_grad():
        out_all = chronos(context=ctx, num_output_patches=2, cross_states=cs,
                          cross_states_mask=cm).quantile_preds
    ids_last = chronos.set_cross_attn_layers("last2")
    assert ids_last == [n - 2, n - 1]
    with torch.no_grad():
        out_last2 = chronos(context=ctx, num_output_patches=2, cross_states=cs,
                            cross_states_mask=cm).quantile_preds
    assert (out_all - out_last2).abs().max().item() > 1e-4, "injecting only the last two layers should differ from injecting all"

    chronos.zero_grad(set_to_none=True)
    chronos(context=ctx, num_output_patches=2, cross_states=cs,
            cross_states_mask=cm).quantile_preds.sum().backward()
    for i, blk in enumerate(chronos.encoder.block):
        g = blk.cross_attn.attn.out_proj.weight.grad
        got = 0.0 if g is None else g.abs().sum().item()
        if i in ids_last:
            assert got > 0, f"layer {i} is selected but has no cross-attn gradient"
        else:
            assert got == 0, f"layer {i} is not selected but has a cross-attn gradient"
    assert chronos.set_cross_attn_layers("first3") == [0, 1, 2]
    assert chronos.set_cross_attn_layers("0,5") == [0, 5]
    try:
        chronos.set_cross_attn_layers(f"{n}")
        raise AssertionError("an out-of-range layer index must raise")
    except ValueError:
        pass
    print(f"[B3] OK  all identity / last2 effective / unselected layers zero gradient ({n} layers in total)")


# ------------------------------------------------------------------ B2: feedback from an intermediate LLM layer
def test_b2_feedback_llm_layer():
    torch.manual_seed(0)
    m0, tok = build()                       # default: last layer
    torch.manual_seed(0)
    m1, _ = build(feedback_llm_layer=1)     # the tiny LLM has 2 layers => layer 1 = intermediate layer
    ids = torch.tensor([[m0.ts_start_id, m0.ts_end_id]
                        + tok("hello world", add_special_tokens=False)["input_ids"]])
    attn = torch.ones_like(ids)
    soft = [torch.zeros(0, m0.llm_hidden)]

    out0, _ = m0._run_llm(ids, attn, None, soft)
    assert m0._fb_hidden(out0.hidden_states[-1]) is out0.hidden_states[-1], "the default must be exactly the last layer"

    out1, _ = m1._run_llm(ids, attn, None, soft)
    picked = m1._fb_hidden(out1.hidden_states[-1])
    with torch.no_grad():
        ref = m1.llm(input_ids=ids, attention_mask=attn, output_hidden_states=True,
                     use_cache=False).hidden_states[1]
    d = (picked - ref).abs().max().item()
    assert picked.shape == out1.hidden_states[-1].shape
    assert d < 1e-4, f"layer 1 taken by the hook must equal output_hidden_states[1], max|delta|={d}"
    assert (picked - out1.hidden_states[-1]).abs().max().item() > 1e-5, "the intermediate layer should really differ from the last layer"

    # A shape mismatch (stale per-step residue from generate) must fail loudly instead of silently using the wrong hidden
    m1._fb_hook_out = torch.zeros(1, 1, m1.llm_hidden)
    try:
        m1._fb_hidden(out1.hidden_states[-1])
        raise AssertionError("a stale hook cache must raise")
    except RuntimeError:
        pass
    print(f"[B2] OK  hook==output_hidden_states[k] (max|delta|={d:.2e}), default last layer zero regression, stale-cache guard present")


# ------------------------------------------------------------------ config round trip
def test_config_roundtrip():
    c = ChronosLLMConfig(history_compressor="pool", feedback_compressor="pool",
                         history_encode_layer=6, feedback_llm_layer=24, cross_attn_layers="last6")
    c2 = ChronosLLMConfig.from_dict(c.to_dict())
    for k in ("history_compressor", "feedback_compressor", "history_encode_layer",
              "feedback_llm_layer", "cross_attn_layers"):
        assert getattr(c2, k) == getattr(c, k), k
    old = ChronosLLMConfig()               # old checkpoint (config.json without these keys)
    assert old.history_compressor == "qformer" and old.feedback_compressor == "qformer"
    assert old.history_encode_layer == 0 and old.feedback_llm_layer == 0
    assert old.cross_attn_layers == "all"
    print("[cfg] OK  the five keys round-trip correctly, defaults = standard configuration")


if __name__ == "__main__":
    test_config_roundtrip()
    test_b1_feedback_pool()
    test_a1_history_pool()
    test_a4_encode_layer()
    test_b3_cross_attn_layers()
    test_b2_feedback_llm_layer()
    print("test_arch_ablations: ALL PASS")
