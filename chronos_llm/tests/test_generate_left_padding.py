"""CPU unit test for the correctness of left padding in generate.

Batched generation with a decoder-only model must use left padding (the real content sits
contiguously at the right end, so the last-position logits belong to the last real token).
This test builds a B=2 batch with **both unequal ctx lengths and unequal text lengths** (mimicking
the collator's right-padded input) and asserts:
  batched generation (the short sample is padded inside the batch) == that sample generated alone with B=1 (no padding), row by row.
Under deterministic greedy decoding the two must match verbatim -- with right padding the short
sample's last position would be a pad token, generation would degrade, and this test would fail.
"""
import torch

from chronos_llm.tests.test_pretrained_peft import _build_tiny_base


def _ids(model, tok, user_text):
    # Same structure as _toy_batch: <ts_start><ts_end> + text (the soft prompt is spliced in after ts_start during forward).
    return [model.ts_start_id, model.ts_end_id] + tok(user_text, add_special_tokens=False)["input_ids"]


def _ctx(vals):
    t = torch.zeros(1, len(vals)); t[0] = torch.tensor(vals, dtype=torch.float32)
    return t


def _single_batch(model, tok, branch, ids, ctx1):
    """B=1: no text padding (attention mask all 1), ctx fully valid."""
    input_ids = torch.tensor([ids])
    return {
        "branch": branch, "context": ctx1, "true_lengths": torch.tensor([ctx1.shape[1]]),
        "input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
        "n_channels": torch.tensor([1]), "n_targets": torch.tensor([1]),
    }


def _padded_batch(model, tok, branch, ids_list, ctx_list):
    """B=N: text right-padded (mimicking the collator), ctx folded into (N, Lmax) with left NaN padding."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    Lmax = max(len(x) for x in ids_list)
    input_ids = torch.full((len(ids_list), Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), Lmax), dtype=torch.long)
    for i, x in enumerate(ids_list):
        input_ids[i, : len(x)] = torch.tensor(x); attn[i, : len(x)] = 1
    Cmax = max(c.shape[1] for c in ctx_list)
    ctx = torch.full((len(ctx_list), Cmax), float("nan"))
    lens = []
    for i, c in enumerate(ctx_list):
        ctx[i, Cmax - c.shape[1] :] = c[0]; lens.append(c.shape[1])
    return {
        "branch": branch, "context": ctx, "true_lengths": torch.tensor(lens),
        "input_ids": input_ids, "attention_mask": attn,
        "n_channels": torch.tensor([1] * len(ids_list)),
        "n_targets": torch.tensor([1] * len(ids_list)),
    }


def test_understanding_left_padding_batch_equals_single():
    torch.manual_seed(0)
    model, tok = _build_tiny_base(); model.eval()
    # Two samples: different text lengths + different ctx lengths -> padding is guaranteed inside the batch (text right pad + shorter soft prompt for the short sample).
    ids0 = _ids(model, tok, "Describe the overall trend in great detail please today.")
    ids1 = _ids(model, tok, "Trend?")
    ctx0 = _ctx([float(i) for i in range(48)])
    ctx1 = _ctx([float(i) for i in range(24)])

    batched = model.generate_understanding(
        _padded_batch(model, tok, "understanding", [ids0, ids1], [ctx0, ctx1]),
        max_new_tokens=8, do_sample=False,
    )
    s0 = model.generate_understanding(_single_batch(model, tok, "understanding", ids0, ctx0),
                                      max_new_tokens=8, do_sample=False)[0]
    s1 = model.generate_understanding(_single_batch(model, tok, "understanding", ids1, ctx1),
                                      max_new_tokens=8, do_sample=False)[0]
    assert batched[0] == s0, f"long sample batch != single:\n{batched[0]!r}\n{s0!r}"
    assert batched[1] == s1, f"short (padded) sample batch != single:\n{batched[1]!r}\n{s1!r}"
    print("understanding left pad: B=2 unequal-length batched generation == per-sample B=1 OK")


def test_forecast_left_padding_batch_equals_single():
    torch.manual_seed(0)
    model, tok = _build_tiny_base(); model.eval()
    ids0 = _ids(model, tok, "Predict the next steps with a careful long explanation here.")
    ids1 = _ids(model, tok, "Next?")
    ctx0 = _ctx([float(i) for i in range(48)])
    ctx1 = _ctx([float(i) for i in range(24)])

    out_b = model.generate_forecast(
        _padded_batch(model, tok, "forecast", [ids0, ids1], [ctx0, ctx1]),
        horizon=32, max_new_tokens=8, do_sample=False,
    )
    o0 = model.generate_forecast(_single_batch(model, tok, "forecast", ids0, ctx0),
                                 horizon=32, max_new_tokens=8, do_sample=False)
    o1 = model.generate_forecast(_single_batch(model, tok, "forecast", ids1, ctx1),
                                 horizon=32, max_new_tokens=8, do_sample=False)
    assert out_b["text"][0] == o0["text"][0], "long sample forecast text batch != single"
    assert out_b["text"][1] == o1["text"][0], "short (padded) sample forecast text batch != single"
    # Quantile forecasts: the padded short sample must also match the standalone run numerically (same soft prompt, same generated text -> same feedback).
    assert torch.allclose(out_b["quantile_preds"][0], o0["quantile_preds"][0], atol=1e-4), "long sample quantiles do not match"
    assert torch.allclose(out_b["quantile_preds"][1], o1["quantile_preds"][0], atol=1e-4), "short sample quantiles do not match"
    print("forecast left pad: B=2 text+quantiles batched == per-sample OK")


def test_generate_restores_train_mode():
    """generate_* called in training mode must restore train() afterwards (otherwise mid-training
    evaluation would permanently switch dropout off); called in eval mode it must stay eval. The
    teacher-forced path restores it as well."""
    model, tok = _build_tiny_base()
    ids = _ids(model, tok, "Trend?")
    ctx = _ctx([float(i) for i in range(24)])
    b_u = _single_batch(model, tok, "understanding", ids, ctx)
    b_f = _single_batch(model, tok, "forecast", ids, ctx)
    b_f["future"] = torch.randn(1, 8)

    model.train()
    model.generate_understanding(b_u, max_new_tokens=2, do_sample=False)
    assert model.training, "train mode should be restored after generate_understanding"
    model.generate_forecast(b_f, horizon=8, max_new_tokens=2, do_sample=False)
    assert model.training, "train mode should be restored after generate_forecast"
    model.generate_forecast_teacher_forced(b_f, horizon=8)
    assert model.training, "train mode should be restored after generate_forecast_teacher_forced"

    model.eval()
    model.generate_understanding(b_u, max_new_tokens=2, do_sample=False)
    assert not model.training, "a call in eval mode must not switch back to train"
    print("generate_* mode restoration OK")


if __name__ == "__main__":
    test_understanding_left_padding_batch_equals_single()
    test_forecast_left_padding_batch_equals_single()
    test_generate_restores_train_mode()
    print("ALL OK")
