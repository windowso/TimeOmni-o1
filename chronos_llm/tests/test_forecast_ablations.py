"""Wiring checks for the three forecasting-branch ablations (CPU, tiny Qwen2 standing in for the 9B):

- **chronos_only (ablation 2)**: forward runs only chronos2(cross_states=None), loss=pred(+roi), text_loss==0,
  backward gradients reach only Chronos-2 and feedback_qformer gets none; generate/tf both predict with Chronos-2 directly.
- **feedback_scope=conclusion (ablation 1)**: the feedback takes only the conclusion segment after </think>, the loss is
  finite, backward connects feedback_qformer+chronos; _conclusion_feedback_mask lets only the conclusion segment through.
- **forecast_prompt_only (ablation 3)**: the dataset emits plain-prompt-only ids (labels all -100) => text_loss==0,
  the whole hidden sequence is fed back, the pred_loss gradient is connected; generate/tf predict through forward.
"""
import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.models.chronos_llm_model import ChronosLLM, ChronosLLMConfig, _add_ts_special_tokens
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn
from chronos_llm.tests.test_composite_wiring import CHRONOS, LLM, left_pad_context, pad_text, pad_labels

CORPUS = os.environ.get("FORECAST_PARQUET", "data/forecast/mmtr_forecast_corpus.parquet")


def build_model(**cfg_overrides):
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    qcfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                       max_position_embeddings=4096)
    llm = Qwen2ForCausalLM(qcfg)
    llm.resize_token_embeddings(len(tok))
    chronos = load_chronos2_with_cross_attn(CHRONOS); chronos.train()
    model = ChronosLLM(
        ChronosLLMConfig(sw_queries_per_window=2, sw_target_windows=8, sw_min_windows=2,
                         sw_global_queries=4, fb_num_query_tokens=8, qformer_num_heads=4,
                         **cfg_overrides),
        chronos, llm, tok)
    for blk in model.chronos.encoder.block:  # open the gate so the feedback path can back-propagate
        blk.cross_attn.gate.data.fill_(0.3)
    return model, tok


def _base_batch(model, tok, input_ids, labels):
    ctx, lens = left_pad_context([[1.0, 2, 3, 4, 5, 6, 7, 8] * 6, [0.5, 1.0, 1.5, 2.0] * 6])
    f_ids, f_attn = pad_text(input_ids, tok.pad_token_id)
    f_labels = pad_labels(labels)
    fl = 64
    future = torch.randn(2, fl) * 100 + 500
    future[1, 40:] = float("nan")
    roi = torch.zeros(2, fl); roi[:, 15:21] = 1.0
    return {"branch": "forecast", "context": ctx, "true_lengths": lens,
            "input_ids": f_ids, "attention_mask": f_attn, "labels": f_labels,
            "future": future, "roi_mask": roi}


def _has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_chronos_only():
    model, tok = build_model(chronos_only=True)
    pre = lambda: [model.ts_start_id, model.ts_end_id] + tok("desc", add_special_tokens=False)["input_ids"]
    ids = [pre() + tok(" up", add_special_tokens=False)["input_ids"] for _ in range(2)]
    labels = [[-100] * len(x) for x in ids]
    batch = _base_batch(model, tok, ids, labels)
    model.zero_grad()
    out = model.forward_forecast(batch)
    assert torch.isfinite(out["loss"]) and float(out["text_loss"]) == 0.0, "chronos_only text_loss should be 0"
    out["loss"].backward()
    assert _has_grad(model.chronos.input_patch_embedding), "chronos should have a gradient"
    assert not _has_grad(model.feedback_qformer), "feedback_qformer should have no gradient under chronos_only"
    qp = model.generate_forecast(batch, horizon=64)["quantile_preds"]
    tf = model.generate_forecast_teacher_forced(batch, horizon=64)["quantile_preds"]
    assert qp.shape[0] == 2 and tf.shape == qp.shape, "chronos_only generate/tf prediction shape is wrong"
    print(f"[chronos_only] loss={float(out['loss']):.3f} text=0 pred_shape={tuple(qp.shape)} gradient only in chronos OK")


def test_feedback_scope_conclusion():
    model, tok = build_model(feedback_scope="conclusion")
    think_end = model.think_end_id
    # Build the <think>reasoning</think>conclusion layout: pre + reasoning + </think> + conclusion
    ids, labels = [], []
    for _ in range(2):
        pre = [model.ts_start_id, model.ts_end_id] + tok("desc", add_special_tokens=False)["input_ids"]
        rea = tok(" reasoning here", add_special_tokens=False)["input_ids"]
        con = tok(" final conclusion", add_special_tokens=False)["input_ids"]
        seq = pre + rea + [think_end] + con
        ids.append(seq)
        labels.append([-100] * len(pre) + rea + [think_end] + con)  # supervise reasoning+conclusion
    batch = _base_batch(model, tok, ids, labels)
    # Unit-check the conclusion mask: only what follows </think> passes
    soft = model._encode_history_to_soft_prompt(batch["context"], batch["true_lengths"], allow_upsample=False)
    _, new_attn = model._run_llm(batch["input_ids"], batch["attention_mask"], batch["labels"], soft)
    fbm = model._conclusion_feedback_mask(batch["input_ids"], soft, new_attn)
    for b in range(2):
        n_soft = int(soft[b].shape[0])
        te = int((batch["input_ids"][b] == think_end).nonzero()[0, 0])
        c_start = te + 1 + n_soft
        assert fbm[b, :c_start].all(), "the conclusion mask should block everything before </think>"
        assert not fbm[b, c_start:int(new_attn[b].sum())].all(), "the conclusion segment should pass (mask=False)"
    model.zero_grad()
    out = model.forward_forecast(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert _has_grad(model.feedback_qformer) and _has_grad(model.chronos.input_patch_embedding)
    print(f"[conclusion] loss={float(out['loss']):.3f} mask located correctly, feedback gradient connected OK")


def test_forecast_prompt_only():
    model, tok = build_model(forecast_prompt_only=True)
    # Dataset side: plain-prompt-only ids + labels all -100
    ds = ForecastParquetDataset(CORPUS, tok, split=None, max_rows=4,
                                max_user_tokens=256, max_tokens=768, forecast_prompt_only=True)
    item = ds[0]
    assert all(l == -100 for l in item["labels"]), "prompt_only labels should all be -100"
    # Compared with non-prompt_only: input_ids are shorter (no answer/reasoning)
    ds_full = ForecastParquetDataset(CORPUS, tok, split=None, max_rows=4,
                                     max_user_tokens=256, max_tokens=768)
    assert len(item["input_ids"]) < len(ds_full[0]["input_ids"]), "prompt_only ids should be shorter than those with an answer"

    # forward: text_loss==0, the pred_loss gradient connects to feedback+chronos
    pre = [model.ts_start_id, model.ts_end_id] + tok("plain prompt facts", add_special_tokens=False)["input_ids"]
    ids = [list(pre), list(pre)]
    labels = [[-100] * len(pre), [-100] * len(pre)]
    batch = _base_batch(model, tok, ids, labels)
    model.zero_grad()
    out = model.forward_forecast(batch)
    assert torch.isfinite(out["loss"]) and float(out["text_loss"]) == 0.0, "prompt_only text_loss should be 0"
    out["loss"].backward()
    assert _has_grad(model.feedback_qformer) and _has_grad(model.chronos.input_patch_embedding)
    qp = model.generate_forecast(batch, horizon=64)["quantile_preds"]
    assert qp.shape[0] == 2
    print(f"[prompt_only] ids shorter than with answer={len(item['input_ids'])}<{len(ds_full[0]['input_ids'])} "
          f"text=0 feedback gradient connected pred_shape={tuple(qp.shape)} OK")


def main():
    torch.manual_seed(0)
    test_chronos_only()
    test_feedback_scope_conclusion()
    test_forecast_prompt_only()
    print("FORECAST ABLATIONS PASSED")


if __name__ == "__main__":
    main()
