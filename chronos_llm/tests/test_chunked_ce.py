"""CPU unit tests for the chunked CE over supervised positions (real Chronos-2 + tiny LLM).

The internal loss of `self.llm(labels=...)` materialises (sum of tokens, vocab) logits for every position,
and the fp32 upcast / gradient of the CE grows linearly with the token count (28GiB for a single batch at a
30000-token budget). `_run_llm` instead takes last_hidden_state from the
backbone and uses `_chunked_ce` (gather supervised positions + chunked lm_head + per-chunk checkpointing in
training mode). Verified to be **numerically equivalent** to the HF internal loss path:
1. loss equivalence (eval mode, multiple chunks forced).
2. hidden_states[-1] == last element of the original output_hidden_states (after the final norm).
3. gradients in training mode (checkpointing active) match the reference path value by value (dropout set to 0).
4. all labels -100 -> loss=0 with a connected graph (no nan).
"""
import torch

from chronos_llm.tests.test_pretrained_peft import _build_tiny_base, _toy_batch


def _zero_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def _ref_forward(model, batch, soft):
    """Reference path: HF internal loss (full-position logits CE) + output_hidden_states."""
    embeds = model.llm.get_input_embeddings()(batch["input_ids"])
    new_e, new_l, new_a = model._splice_soft_prompt(
        embeds, batch["labels"], batch["attention_mask"], batch["input_ids"], soft)
    out = model.llm(inputs_embeds=new_e, attention_mask=new_a, labels=new_l,
                    output_hidden_states=True, use_cache=False)
    return out.loss, out.hidden_states[-1]


def test_loss_and_hidden_match_reference():
    model, tok = _build_tiny_base()
    model.eval()
    batch = _toy_batch(model, tok, "understanding")
    with torch.no_grad():
        soft = model._encode_history_to_soft_prompt(
            batch["context"], batch["true_lengths"], batch.get("n_channels"))
        ref_loss, ref_hidden = _ref_forward(model, batch, soft)
        out, _ = model._run_llm(
            batch["input_ids"], batch["attention_mask"], batch["labels"], soft)
        # chunk=2 forces several chunks, verifying that the chunked accumulation equals a single pass
        new_loss_small_chunk = model._chunked_ce(ref_hidden, _spliced_labels(model, batch, soft), chunk=2)
    assert torch.allclose(out.loss, ref_loss, atol=1e-5), \
        f"chunked CE != HF internal loss: {out.loss.item()} vs {ref_loss.item()}"
    assert torch.allclose(new_loss_small_chunk, ref_loss, atol=1e-5), \
        f"chunk=2 multi-chunk accumulation != HF internal loss: {new_loss_small_chunk.item()} vs {ref_loss.item()}"
    assert torch.allclose(out.hidden_states[-1], ref_hidden, atol=1e-6), \
        "backbone last_hidden_state != last element of the original output_hidden_states"
    print(f"chunked CE == HF internal loss OK (delta={abs(out.loss.item() - ref_loss.item()):.2e}), hidden matches OK")


def _spliced_labels(model, batch, soft):
    embeds = model.llm.get_input_embeddings()(batch["input_ids"])
    _, new_l, _ = model._splice_soft_prompt(
        embeds, batch["labels"], batch["attention_mask"], batch["input_ids"], soft)
    return new_l


def _grads(model, via_ref: bool):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    tok = model.tokenizer
    batch = _toy_batch(model, tok, "understanding")
    soft = model._encode_history_to_soft_prompt(
        batch["context"], batch["true_lengths"], batch.get("n_channels"))
    if via_ref:
        loss, _ = _ref_forward(model, batch, soft)
    else:
        out, _ = model._run_llm(
            batch["input_ids"], batch["attention_mask"], batch["labels"], soft)
        loss = out.loss
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_train_grads_match_reference():
    model, _ = _build_tiny_base()
    model.train()
    _zero_dropout(model)
    g_ref = _grads(model, via_ref=True)
    g_new = _grads(model, via_ref=False)
    assert g_ref.keys() == g_new.keys(), "the two paths have different sets of parameters with gradients"
    worst = 0.0
    for k in g_ref:
        d = (g_ref[k] - g_new[k]).abs().max().item()
        worst = max(worst, d)
        assert torch.allclose(g_ref[k], g_new[k], atol=1e-5), f"{k} gradient mismatch: max|delta|={d}"
    print(f"training-mode (per-chunk checkpoint) gradients == reference path OK ({len(g_ref)} parameters, max|delta|={worst:.2e})")


def test_all_ignored_labels():
    model, tok = _build_tiny_base()
    model.train()
    batch = _toy_batch(model, tok, "understanding")
    batch["labels"] = torch.full_like(batch["labels"], -100)
    out = model.forward_understanding(batch)
    assert torch.isfinite(out["loss"]).all() and out["loss"].item() == 0.0, \
        f"all -100 should return 0: {out['loss']}"
    out["loss"].backward()   # graph connected, no crash
    print("all labels -100 -> loss=0 and backward works OK")


if __name__ == "__main__":
    test_loss_and_hidden_match_reference()
    test_train_grads_match_reference()
    test_all_ignored_labels()
    print("ALL OK")
