"""Full-pipeline wiring check for the TimesFM-3.0 backbone (CPU, a tiny Qwen2 stands in for the 9B
LLM) -- the arm of the backbone ablation that swaps the TSFM.

Same structure and same assertions as ``test_composite_wiring``, with the TSFM replaced by
TimesFM-3.0: it verifies that **the whole pipeline still runs after the backbone swap** -- the soft
prompt length is planned from the new patch_size, both the understanding and the forecasting branch
losses are finite, and the backward pass reaches TimesFM's patch embedding, the injected cross-attn
gate, both Q-formers and the LLM.

It also guards two things that only a backbone swap can expose:
- ``input_patch_size`` must really be read off the backbone (32 for TimesFM, 16 for Chronos-2) --
  hard-coding 16 would make the soft-token budget off by a factor of two without raising anything;
- ``ChronosLLM`` takes the Q-former's in_dim from ``chronos.config.d_model`` (1280, not 768).
"""

import math
import os
import sys

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO, os.path.join(_REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from chronos_llm.models.chronos_llm_model import (  # noqa: E402
    ChronosLLM,
    ChronosLLMConfig,
    _add_ts_special_tokens,
)
from chronos_llm.models.timesfm_backbone import load_timesfm3_backbone  # noqa: E402

TIMESFM = os.environ.get("TIMESFM3_PATH", "checkpoints/TimesFM3.0")
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
    backbone = load_timesfm3_backbone(TIMESFM)
    backbone.train()
    model = ChronosLLM(
        ChronosLLMConfig(sw_queries_per_window=2, sw_target_windows=8, sw_min_windows=2,
                         sw_global_queries=4, fb_num_query_tokens=8, qformer_num_heads=4,
                         tsfm_backbone="timesfm3"),
        backbone, llm, tok,
    )
    return model, tok


def make_text(model, tok):
    pre = [model.ts_start_id, model.ts_end_id] + tok("Describe the series.", add_special_tokens=False)["input_ids"]
    ans = tok(" It rises then falls.", add_special_tokens=False)["input_ids"]
    return pre + ans, [-100] * len(pre) + ans


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

    # The backbone attributes must genuinely come from TimesFM and not from chronos hard-coded values.
    assert model.input_patch_size == 32, f"input_patch_size should be taken from the backbone (32), got {model.input_patch_size}"
    assert model.history_qformer.qformer.input_proj.in_features == 1280, "the Q-former in_dim should be TimesFM's 1280"
    assert model.chronos_window == 8192
    print(f"[cfg] patch={model.input_patch_size} qformer_in={model.history_qformer.qformer.input_proj.in_features} "
          f"window={model.chronos_window} layers={len(model.chronos.encoder.block)}")

    # ---- understanding branch ----
    ids = [make_text(model, tok) for _ in range(2)]
    input_ids, attn = pad_text([a for a, _ in ids], tok.pad_token_id)
    labels = pad_labels([b for _, b in ids])
    ctx, lens = left_pad_context([[1.0, 2, 3, 4, 5, 6, 7, 8] * 24, [0.5, 1.0, 1.5, 2.0] * 24])

    N_expected = [
        model.history_qformer.num_tokens_for_length(
            math.ceil(int(l) / model.chronos.chronos_config.input_patch_size)
        )
        for l in lens
    ]
    soft_list = model._encode_history_to_soft_prompt(ctx, lens)
    soft_lens = [s.shape[0] for s in soft_list]
    print(f"[soft prompt] lengths={soft_lens} expected={N_expected}")
    assert soft_lens == N_expected, f"soft prompt lengths {soft_lens} != planned {N_expected}"
    assert all(s.shape[1] == model.llm_hidden for s in soft_list)

    u = model.forward_understanding(
        {"context": ctx, "true_lengths": lens, "input_ids": input_ids,
         "attention_mask": attn, "labels": labels}
    )
    print(f"[understanding] CE={float(u['loss']):.3f}")
    assert torch.isfinite(u["loss"])

    # ---- forecasting branch ----
    fids = [make_text(model, tok) for _ in range(2)]
    f_input_ids, f_attn = pad_text([a for a, _ in fids], tok.pad_token_id)
    f_labels = pad_labels([b for _, b in fids])
    fl = 64
    future = torch.randn(2, fl) * 100 + 500
    future[1, 40:] = float("nan")          # variable-length future
    roi = torch.zeros(2, fl); roi[:, 15:21] = 1.0
    for blk in model.chronos.encoder.block:
        blk.cross_attn.gate.data.fill_(0.3)   # open the feedback to check the path can backpropagate
    batch = {
        "branch": "forecast", "context": ctx, "true_lengths": lens,
        "input_ids": f_input_ids, "attention_mask": f_attn, "labels": f_labels,
        "future": future, "roi_mask": roi,
    }
    f = model.forward_forecast(batch)
    print(f"[forecast] text={float(f['text_loss']):.3f} pred={float(f['pred_loss']):.3f} "
          f"roi={float(f['roi_loss']):.3f} total={float(f['loss']):.3f}")
    for k in ["text_loss", "pred_loss", "roi_loss", "loss"]:
        assert torch.isfinite(f[k]).all(), f"{k} is not finite"

    # ---- the backward pass reaches the key modules ----
    f["loss"].backward()
    grads = {
        "timesfm.pre_transformer_resblock": model.chronos.tfm.pre_transformer_resblock.hidden_layer.weight.grad,
        "timesfm.output_head": model.chronos.tfm.output_head.weight.grad,
        "timesfm.cross_attn.gate(l0)": model.chronos.encoder.block[0].cross_attn.gate.grad,
        "history_qformer.query": model.history_qformer.qformer.query_tokens.grad,
        "feedback_qformer.query": model.feedback_qformer.query_tokens.grad,
        "llm.embed": model.llm.get_input_embeddings().weight.grad,
    }
    for name, g in grads.items():
        ok = g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
        print(f"  grad {name}: {'OK' if ok else 'MISSING/ZERO'}")
        assert ok, f"gradient did not reach {name}"

    print("\nTIMESFM WIRING CHECKS PASSED")


if __name__ == "__main__":
    main()
