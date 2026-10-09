"""Does B2 (`--feedback_llm_layer`) still receive gradients under **gradient checkpointing**?

Why this deserves its own test: feeding back an intermediate LLM layer relies on a forward hook,
and the hook fires **inside** ``decoder_layer.__call__`` -- during real training that layer is
wrapped by gradient checkpointing. With the reentrant checkpoint variant (whose first forward runs
under ``torch.no_grad()``) the tensor captured by the hook has no grad_fn, and **the gradient of
the feedback branch is silently cut before the LLM**: the B2 arm appears to train, but the LLM
never receives the pred_loss gradient again, invalidating the ablation without any error.

This repository's ``gradient_checkpointing_enable`` explicitly passes ``use_reentrant=False`` (the
non-reentrant variant runs in grad mode and merely discards intermediate activations for
recomputation), so the gradient should flow -- but this chain is too easily flipped by an upstream
default, hence a regression guard.
"""

import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from chronos_llm.models.chronos_llm_model import ChronosLLM, ChronosLLMConfig, _add_ts_special_tokens
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn

CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")


def build(fb_layer: int):
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    qcfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                       max_position_embeddings=4096)
    llm = Qwen2ForCausalLM(qcfg)
    llm.resize_token_embeddings(len(tok))
    chronos = load_chronos2_with_cross_attn(CHRONOS)
    chronos.train()
    cfg = ChronosLLMConfig(sw_queries_per_window=2, sw_global_queries=4, fb_num_query_tokens=8,
                           qformer_num_heads=4, feedback_llm_layer=fb_layer)
    return ChronosLLM(cfg, chronos, llm, tok), tok


def run(fb_layer: int, grad_ckpt: bool) -> dict:
    torch.manual_seed(0)
    m, tok = build(fb_layer)
    m.train()
    if grad_ckpt:
        m.gradient_checkpointing_enable()
    ids = torch.tensor([[m.ts_start_id, m.ts_end_id]
                        + tok("the series rises then falls sharply", add_special_tokens=False)["input_ids"]])
    attn = torch.ones_like(ids)
    soft = [torch.zeros(0, m.llm_hidden)]

    out, new_a = m._run_llm(ids, attn, None, soft)
    hidden = m._fb_hidden(out.hidden_states[-1])
    c = m.feedback_qformer(hidden.to(m.feedback_qformer.input_proj.weight.dtype),
                           src_key_padding_mask=(new_a <= 0))
    c.sum().backward()

    # Probe a parameter of a layer **before** the hooked one: it only receives a gradient if the
    # gradient really flows through the tensor captured by the hook.
    layers = m._llm_decoder_layers()
    probe = layers[max(0, fb_layer - 1)].self_attn.q_proj.weight
    later = layers[-1].self_attn.q_proj.weight   # layers after the intermediate one: this path must not reach them
    return {
        "has_grad_fn": hidden.grad_fn is not None,
        "probe_grad": 0.0 if probe.grad is None else float(probe.grad.abs().sum()),
        "later_grad": 0.0 if later.grad is None else float(later.grad.abs().sum()),
        "fb_grad": float(m.feedback_qformer.input_proj.weight.grad.abs().sum()),
    }


def main():
    for path, name in ((CHRONOS, "CHRONOS2_PATH"), (LLM, "LLM_PATH")):
        if not os.path.exists(path):
            print(f"SKIP: {path} not found (set {name} to the checkpoint directory)")
            return
    for gc in (False, True):
        r = run(fb_layer=2, grad_ckpt=gc)
        tag = "grad_ckpt=ON " if gc else "grad_ckpt=OFF"
        print(f"[{tag}] hidden.grad_fn={r['has_grad_fn']}  LLM param grad before layer 2={r['probe_grad']:.4e}  "
              f"last-layer grad={r['later_grad']:.4e}  feedback projection grad={r['fb_grad']:.4e}")
        assert r["has_grad_fn"], f"{tag}: the intermediate-layer tensor captured by the hook has no grad_fn => feedback gradient is cut"
        assert r["probe_grad"] > 0, f"{tag}: the gradient did not flow back through the hook into the LLM (the B2 arm would silently fail)"
        assert r["later_grad"] == 0, f"{tag}: layers after the intermediate one must not receive the feedback gradient"
        assert r["fb_grad"] > 0, f"{tag}: the feedback compressor itself has no gradient"
    print("test_fb_layer_grad_ckpt: ALL PASS (gradient flows through the hook path with use_reentrant=False)")


if __name__ == "__main__":
    main()
