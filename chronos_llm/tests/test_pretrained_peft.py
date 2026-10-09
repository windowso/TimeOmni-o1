"""Milestone: verification of the PreTrainedModel + training-side PEFT refactor (CPU, tiny Qwen2 standing in for the 9B).

Test functions and imports were added incrementally per task; __main__ runs all of them in order.
"""
import os
import tempfile

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from chronos_llm.models.chronos_llm_model import (
    ChronosLLM, ChronosLLMConfig, add_lora, _add_ts_special_tokens,
    LLM_LORA_TARGET_REGEX, TRAINABLE_MODULES,
)
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn

CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")


def _build_tiny_base():
    """Real Chronos-2 + tiny Qwen2: build a ChronosLLM(PreTrainedModel) without LoRA."""
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    qcfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                       max_position_embeddings=4096)
    llm = Qwen2ForCausalLM(qcfg)
    llm.config.use_cache = False
    chronos = load_chronos2_with_cross_attn(CHRONOS); chronos.train()
    cfg = ChronosLLMConfig(chronos_ckpt=CHRONOS, llm_path=LLM, sw_queries_per_window=2,
                           sw_target_windows=8, sw_min_windows=2, sw_global_queries=4,
                           fb_num_query_tokens=8, qformer_num_heads=4)
    return ChronosLLM(cfg, chronos, llm, tok), tok


def _toy_batch(model, tok, branch):
    pre = [model.ts_start_id, model.ts_end_id] + tok("Describe.", add_special_tokens=False)["input_ids"]
    ans = tok(" up then down.", add_special_tokens=False)["input_ids"]
    ids = pre + ans
    labels = [-100] * len(pre) + ans
    input_ids = torch.tensor([ids, ids])
    attn = torch.ones_like(input_ids)
    labels_t = torch.tensor([labels, labels])
    ctx = torch.full((2, 48), float("nan")); ctx[0] = torch.arange(48.0); ctx[1, 24:] = torch.arange(24.0)
    lens = torch.tensor([48, 24])
    batch = {"branch": branch, "context": ctx, "true_lengths": lens,
             "input_ids": input_ids, "attention_mask": attn, "labels": labels_t}
    if branch == "forecast":
        fut = torch.randn(2, 32) * 10 + 50; fut[1, 20:] = float("nan")
        batch["future"] = fut
        roi = torch.zeros(2, 32); roi[:, 5:9] = 1.0
        batch["roi_mask"] = roi
    return batch


def test_config_roundtrip():
    cfg = ChronosLLMConfig(chronos_ckpt="a", llm_path="b", sw_target_windows=8,
                           sw_global_queries=4, fb_num_query_tokens=8, qformer_num_heads=4,
                           sw_token_min=30, sw_token_max=150, sw_token_pc_ref=8192)
    with tempfile.TemporaryDirectory() as d:
        cfg.save_pretrained(d)
        cfg2 = ChronosLLMConfig.from_pretrained(d)
    assert cfg2.chronos_ckpt == "a" and cfg2.llm_path == "b"
    assert cfg2.sw_target_windows == 8 and cfg2.fb_num_query_tokens == 8
    assert cfg2.sw_global_queries == 4
    # Older checkpoints have no sw_global_queries key in config.json -> must default to 0 (no random global branch appears out of nowhere)
    assert ChronosLLMConfig(chronos_ckpt="a", llm_path="b").sw_global_queries == 0
    # round trip of the newer fields
    assert cfg2.sw_token_min == 30 and cfg2.sw_token_max == 150 and cfg2.sw_token_pc_ref == 8192
    # Older config.json without these keys -> defaults 40/200/32768 (old checkpoints load without error and use the new defaults)
    _old = ChronosLLMConfig(chronos_ckpt="a", llm_path="b")
    assert (_old.sw_token_min, _old.sw_token_max, _old.sw_token_pc_ref) == (40, 200, 32768)
    assert cfg2.model_type == "chronos_llm"
    print("config roundtrip OK")


def test_build_base_is_pretrained_and_forward():
    from transformers import PreTrainedModel
    model, tok = _build_tiny_base()
    assert isinstance(model, PreTrainedModel)
    u = model.forward_understanding(_toy_batch(model, tok, "understanding"))
    assert torch.isfinite(u["loss"]).all()
    f = model.forward_forecast(_toy_batch(model, tok, "forecast"))
    for k in ("text_loss", "pred_loss", "roi_loss", "loss"):
        assert torch.isfinite(f[k]).all()
    assert not hasattr(model, "save_trainable")
    print("build base + forward OK")


def test_add_lora_trainable_set_and_isolation():
    model, tok = _build_tiny_base()
    model = add_lora(model, r=4, alpha=8, dropout=0.0)

    named = dict(model.named_parameters())
    # 1) LoRA is injected into the llm, and only there (chronos contains no lora_)
    lora_keys = [k for k in named if "lora_" in k]
    assert lora_keys, "no LoRA injected"
    assert all(".llm." in k for k in lora_keys), "LoRA injected outside the llm"
    assert not any(".chronos." in k for k in lora_keys), "LoRA hit chronos (out_proj name clash not isolated)"

    # 2) chronos/qformer go through modules_to_save -> trainable copies exist
    def any_trainable(substr):
        return any(p.requires_grad for k, p in named.items() if substr in k)
    assert any_trainable("chronos") and any_trainable("history_qformer") and any_trainable("feedback_qformer")

    # 3) the base LLM body is frozen (llm weights that are neither lora nor modules_to_save are not trainable)
    base_llm = [p for k, p in named.items()
                if ".llm." in k and "lora_" not in k and "modules_to_save" not in k]
    assert base_llm and all(not p.requires_grad for p in base_llm), "base LLM not frozen"

    # 4) gradients from both branches reach chronos / both qformers / LoRA
    #    With a zero-initialised gate, tanh(0)=0 closes the feedback path (expected at cold start); open the gate first to verify the feedback path is connected.
    base = model.get_base_model()
    for blk in base.chronos.encoder.block:
        blk.cross_attn.gate.data.fill_(0.3)
    f = model(_toy_batch(base, tok, "forecast"))
    f["loss"].backward()
    def grad_ok(substr):
        return any(p.grad is not None and p.grad.abs().sum() > 0
                   for k, p in named.items() if substr in k and p.requires_grad)
    assert grad_ok("history_qformer") and grad_ok("feedback_qformer") and grad_ok("lora_")
    print("add_lora trainable-set + isolation + grad OK")


def test_save_load_roundtrip():
    from peft import PeftModel
    base, tok = _build_tiny_base()
    # NOTE: capture the clean base weights before add_lora (keys not yet rewritten by LoRA/wrappers), to load into the rebuilt base2.
    clean_sd = {k: v.clone() for k, v in base.state_dict().items()}
    model = add_lora(base, r=4, alpha=8, dropout=0.0)
    # Perturb the trainable parameters so the adapter stores non-trivial values (otherwise a broken round trip could still match by coincidence).
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    model.eval()
    batch = _toy_batch(model.get_base_model(), tok, "understanding")
    with torch.no_grad():
        ref = model(batch)["loss"].item()

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        assert os.path.exists(os.path.join(d, "adapter_model.safetensors"))
        base2, tok2 = _build_tiny_base()
        base2.load_state_dict(clean_sd, strict=True)   # clean onto clean, keys match exactly
        loaded = PeftModel.from_pretrained(base2, d)   # load LoRA + overwrite with the trained chronos/qformer
        loaded.eval()
        with torch.no_grad():
            got = loaded(_toy_batch(loaded.get_base_model(), tok2, "understanding"))["loss"].item()
    assert abs(ref - got) < 1e-4, f"round-trip values differ: {ref} vs {got}"
    print("save/load roundtrip OK")


def test_generate():
    model, tok = _build_tiny_base()
    model = add_lora(model, r=4, alpha=8, dropout=0.0)
    base = model.get_base_model()
    b_u = _toy_batch(base, tok, "understanding")
    texts = base.generate_understanding(b_u, max_new_tokens=4, do_sample=False)
    assert isinstance(texts, list) and len(texts) == 2 and all(isinstance(t, str) for t in texts)

    b_f = _toy_batch(base, tok, "forecast")
    out = base.generate_forecast(b_f, horizon=32, max_new_tokens=4, do_sample=False)
    assert isinstance(out["text"], list) and len(out["text"]) == 2
    qp = out["quantile_preds"]
    assert qp.dim() == 3 and qp.shape[0] == 2 and torch.isfinite(qp).all()
    print("generate understanding + forecast OK")


def test_from_pretrained_merge_roundtrip():
    """from_pretrained(dir, merge=True) -> a merged plain ChronosLLM whose forward equals the training-time PeftModel.

    from_config is mocked (to avoid loading the real 9B on CPU): it returns a tiny base loaded with the clean
    weights, equivalent to production from_config deterministically rebuilding the base from external checkpoints.
    """
    from unittest.mock import patch

    base, tok = _build_tiny_base()
    clean_sd = {k: v.clone() for k, v in base.state_dict().items()}
    model = add_lora(base, r=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    model.eval()
    with torch.no_grad():
        ref = model(_toy_batch(model.get_base_model(), tok, "understanding"))["loss"].item()

    def fake_from_config(config):
        b2, _ = _build_tiny_base()
        b2.load_state_dict(clean_sd, strict=True)
        return b2

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        model.get_base_model().config.save_pretrained(d)
        with patch.object(ChronosLLM, "from_config", staticmethod(fake_from_config)):
            loaded = ChronosLLM.from_pretrained(d, merge=True)
        assert isinstance(loaded, ChronosLLM), "merge=True should return a plain ChronosLLM"
        loaded.eval()
        with torch.no_grad():
            got = loaded(_toy_batch(loaded, tok, "understanding"))["loss"].item()
    assert abs(ref - got) < 1e-4, f"from_pretrained(merge) values differ: {ref} vs {got}"
    print("from_pretrained merge roundtrip OK")


if __name__ == "__main__":
    test_config_roundtrip()
    test_build_base_is_pretrained_and_forward()
    test_add_lora_trainable_set_and_isolation()
    test_save_load_roundtrip()
    test_generate()
    test_from_pretrained_merge_roundtrip()
    print("ALL OK")
