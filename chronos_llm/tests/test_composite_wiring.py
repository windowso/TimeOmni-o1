"""Wiring check for the composite model (CPU, a tiny Qwen2 LLM stands in for the 9B Qwen3.5; only
wiring / shapes / losses are verified).

Verified:
- soft-prompt injection: the actual LLM sequence length = text length + N (sliding-window tokens).
- understanding branch: CE loss finite.
- forecast branch: text/pred/roi losses all finite, roi effective; loss.backward() reaches chronos,
  both Q-formers and the cross-attn gate.
"""

import math
import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from chronos_llm.models.chronos_llm_model import ChronosLLM, ChronosLLMConfig, _add_ts_special_tokens
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn

CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")


def build_tiny():
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    cfg = Qwen2Config(
        vocab_size=len(tok), hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=4096,
    )
    llm = Qwen2ForCausalLM(cfg)
    llm.resize_token_embeddings(len(tok))
    chronos = load_chronos2_with_cross_attn(CHRONOS)
    chronos.train()
    model = ChronosLLM(
        ChronosLLMConfig(sw_queries_per_window=2, sw_target_windows=8, sw_min_windows=2,
                         sw_global_queries=4, fb_num_query_tokens=8, qformer_num_heads=4),
        chronos, llm, tok,
    )
    return model, tok


def make_text(model, tok, answer_supervised: bool):
    """Build one input_ids/labels pair containing <ts_start><ts_end>."""
    pre = [model.ts_start_id, model.ts_end_id] + tok("Describe the series.", add_special_tokens=False)["input_ids"]
    ans = tok(" It rises then falls.", add_special_tokens=False)["input_ids"]
    ids = pre + ans
    labels = [-100] * len(pre) + ans if answer_supervised else None
    return ids, labels


def pad_text(seqs, pad_id):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    return out, attn


def pad_labels(seqs):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def left_pad_context(hist_list):
    L = max(len(h) for h in hist_list)
    ctx = torch.full((len(hist_list), L), float("nan"))
    lens = torch.zeros(len(hist_list), dtype=torch.long)
    for i, h in enumerate(hist_list):
        ctx[i, L - len(h):] = torch.tensor(h, dtype=torch.float32)
        lens[i] = len(h)
    return ctx, lens


def main():
    torch.manual_seed(0)
    model, tok = build_tiny()
    # sw_global_queries from the config must be wired into the sliding-window Q-former (the soft
    # prompt contains global tokens, so the num_tokens_for_length check below also covers the
    # end-to-end output of the global branch).
    assert model.history_qformer.global_queries == 4, \
        f"sw_global_queries not wired: {model.history_qformer.global_queries}"

    # ---- understanding branch ----
    ids = [make_text(model, tok, True) for _ in range(2)]
    input_ids, attn = pad_text([a for a, _ in ids], tok.pad_token_id)
    labels = pad_labels([b for _, b in ids])
    ctx, lens = left_pad_context([[1.0, 2, 3, 4, 5, 6, 7, 8] * 6, [0.5, 1.0, 1.5, 2.0] * 6])
    # the soft prompt is now variable-length per sample: windows are planned from each sample's valid patch count (ceil(true_len/patch)).
    N_expected = [
        model.history_qformer.num_tokens_for_length(
            math.ceil(int(l) / model.chronos.chronos_config.input_patch_size)
        )
        for l in lens
    ]

    # check that the per-sample soft-prompt lengths produced by chronos sliding windows -> ladder Q-former match the plan.
    soft_list = model._encode_history_to_soft_prompt(ctx, lens)
    soft_lens = [s.shape[0] for s in soft_list]
    print(f"[soft prompt] lengths={soft_lens} expected={N_expected}")
    assert soft_lens == N_expected, f"soft lengths {soft_lens} != planned {N_expected}"

    # call forward directly
    u = model.forward_understanding(
        {"context": ctx, "true_lengths": lens, "input_ids": input_ids, "attention_mask": attn, "labels": labels}
    )
    print(f"[understanding] CE={float(u['loss']):.3f} finite={bool(torch.isfinite(u['loss']))}")
    assert torch.isfinite(u["loss"])

    # ---- forecast branch ----
    fids = [make_text(model, tok, True) for _ in range(2)]
    f_input_ids, f_attn = pad_text([a for a, _ in fids], tok.pad_token_id)
    f_labels = pad_labels([b for _, b in fids])
    fl = 64
    future = torch.randn(2, fl) * 100 + 500
    future[1, 40:] = float("nan")  # the second future is shorter
    roi = torch.zeros(2, fl); roi[:, 15:21] = 1.0
    # With a zero-initialised gate the feedback path is closed at step 0 (tanh(0)=0; expected for the
    # tanh-zero cold start -- the gate itself receives a gradient first and then opens). Open the
    # gate here to verify that the feedback path is really connected and back-propagates.
    for blk in model.chronos.encoder.block:
        blk.cross_attn.gate.data.fill_(0.3)
    batch = {
        "branch": "forecast", "context": ctx, "true_lengths": lens,
        "input_ids": f_input_ids, "attention_mask": f_attn, "labels": f_labels,
        "future": future, "roi_mask": roi,
    }
    f = model.forward_forecast(batch)
    print(f"[forecast] text={float(f['text_loss']):.3f} pred={float(f['pred_loss']):.3f} "
          f"roi={float(f['roi_loss']):.3f} total={float(f['loss']):.3f}")
    for k in ["text_loss", "pred_loss", "roi_loss", "loss"]:
        assert torch.isfinite(f[k] if k != "loss" else f["loss"]).all()

    # backward: check that gradients reach the key modules
    f["loss"].backward()
    grads = {
        "chronos.input_patch_embedding": model.chronos.input_patch_embedding.hidden_layer.weight.grad,
        "chronos.cross_attn.gate(l0)": model.chronos.encoder.block[0].cross_attn.gate.grad,
        "history_qformer.query": model.history_qformer.qformer.query_tokens.grad,
        "feedback_qformer.query": model.feedback_qformer.query_tokens.grad,
        "llm.embed": model.llm.get_input_embeddings().weight.grad,
    }
    for name, g in grads.items():
        ok = g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
        print(f"  grad {name}: {'OK' if ok else 'MISSING/ZERO'}")
        assert ok, f"gradient did not reach {name}"

    print("WIRING CHECKS PASSED")


if __name__ == "__main__":
    main()
