"""Wiring check for GRPO rollout (CPU, tiny Qwen2 standing in for the 9B):
rollout_forecast_rl samples G groups -> logps(G,B) have grad, preds (no_grad) have the right shape,
grad reaches only the LLM and not chronos.
"""
import torch

from chronos_llm.tests.test_forecast_ablations import _base_batch, _has_grad, build_model


def test_grpo_rollout():
    model, tok = build_model()          # FULL feedback path (default)
    model.eval()
    pre = lambda: [model.ts_start_id, model.ts_end_id] + tok("desc", add_special_tokens=False)["input_ids"]
    ids = [pre() + tok(" up", add_special_tokens=False)["input_ids"] for _ in range(2)]
    labels = [[-100] * len(x) for x in ids]     # inference layout, labels unused
    batch = _base_batch(model, tok, ids, labels)

    G, B, H = 3, 2, 16
    # rollout is entirely no_grad: returns prefix + per-group gen_ids/preds/texts
    prefix_e, prefix_a, gen_ids, preds, texts = model.rollout_forecast_rl(
        batch, horizon=H, group_size=G, temperature=1.0, max_new_tokens=4)
    assert len(gen_ids) == G and len(preds) == G and len(texts) == G
    assert prefix_e.dim() == 3 and prefix_e.shape[0] == B, prefix_e.shape
    assert gen_ids[0].shape[0] == B and gen_ids[0].dim() == 2
    assert preds[0].dim() == 3 and preds[0].shape[0] == B, preds[0].shape   # (sum targets=B, Q, H)
    assert not preds[0].requires_grad, "fed-back forecast should be no_grad (used for the reward)"
    assert len(texts[0]) == B

    # rl_logp: single-group grad forward; grad reaches only the LLM, not chronos/feedback_qformer
    # (GRPO trains only the LLM)
    model.zero_grad()
    logp = model.rl_logp(prefix_e, prefix_a, gen_ids[0])   # (B,)
    assert logp.shape == (B,) and logp.requires_grad, logp.shape
    (-logp.mean()).backward()
    assert _has_grad(model.llm), "LLM should have grad"
    assert not _has_grad(model.chronos), "chronos should have no grad (prefix detached)"
    assert not _has_grad(model.feedback_qformer), "feedback_qformer should have no grad (rollout no_grad)"
    print(f"GRPO rollout OK: gen_ids x{G} preds{tuple(preds[0].shape)} no_grad ok, "
          f"rl_logp{tuple(logp.shape)} grad ok, grad only on LLM ok")


def test_grpo_step():
    """grpo_step: rollout -> reward(-CRPS+ROI+mag) -> in-group advantage -> backward -> LLM update."""
    from chronos_llm.rl.grpo_trainer import compute_rewards, grpo_step

    model, tok = build_model()
    model.train()
    pre = lambda: [model.ts_start_id, model.ts_end_id] + tok("desc", add_special_tokens=False)["input_ids"]
    ids = [pre() + tok(" up", add_special_tokens=False)["input_ids"] for _ in range(2)]
    batch = _base_batch(model, tok, ids, [[-100] * len(x) for x in ids])
    # emit_meta protocol: the GRPO reward parses the true ROI/magnitude from gt_conclusion
    # (containing [a,b) + magnitude words)
    batch["meta"] = [
        {"gt_conclusion": "In [10, 16) rises to a peak. Overall, well below the series' usual highs."},
        {"gt_conclusion": "In [12, 18) a crest. Overall, well above the series' typical highs."},
    ]
    # Length not divisible by 16: chronos patch alignment => preds length (64) > future (62); checks
    # that compute_rewards trims to the common length without crashing.
    batch["future"] = batch["future"][:, :62]
    # freeze non-LLM parameters (GRPO trains only the LLM)
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("llm."))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    before = next(p for n, p in model.named_parameters() if n.startswith("llm.") and p.requires_grad).detach().clone()

    H = batch["future"].shape[-1]
    m = grpo_step(model, batch, opt, horizon=H, group_size=4, temperature=1.0, max_new_tokens=4)
    assert all(k in m for k in ("loss", "reward", "crps", "reward_std_in_group", "grad_norm")), m
    import math
    assert math.isfinite(m["loss"]) and math.isfinite(m["reward"]), m
    after = next(p for n, p in model.named_parameters() if n.startswith("llm.") and p.requires_grad)
    assert (after.detach() - before).abs().sum() > 0, "LLM parameters should be updated"
    print(f"GRPO step OK: { {k: round(v,3) for k,v in m.items()} }")


def test_ss_forecast_loss():
    """Scheduled sampling: generated conclusion fed back -> pred/roi loss; grad reaches
    chronos+feedback, not the LLM."""
    import math

    model, tok = build_model()          # build_model already sets gate=0.3 so the feedback path is differentiable
    model.train()
    pre = lambda: [model.ts_start_id, model.ts_end_id] + tok("desc", add_special_tokens=False)["input_ids"]
    ids = [pre() + tok(" up", add_special_tokens=False)["input_ids"] for _ in range(2)]
    batch = _base_batch(model, tok, ids, [[-100] * len(x) for x in ids])

    model.zero_grad()
    out = model.ss_forecast_loss(batch, max_new_tokens=4)
    assert math.isfinite(float(out["loss"])), out
    out["loss"].backward()
    assert _has_grad(model.chronos), "SS should train chronos"
    assert _has_grad(model.feedback_qformer), "SS should train feedback_qformer"
    assert not _has_grad(model.llm), "the LLM should be no_grad under SS (generation)"
    print(f"SS forecast loss OK: loss={float(out['loss']):.3f}, grad reaches chronos+feedback, not the LLM ok")


def test_tf_forecast_loss():
    """Frozen-LLM forecast loss: teacher-text hidden states (LLM no_grad) fed
    back -> pred/roi loss, grad reaches chronos+feedback, not the LLM -- same structure as
    ss_forecast_loss but without generation (training-rendered batch)."""
    import math

    model, tok = build_model()
    model.train()
    pre = lambda: [model.ts_start_id, model.ts_end_id] + tok("desc", add_special_tokens=False)["input_ids"]
    ans = tok(" up and down", add_special_tokens=False)["input_ids"]
    ids = [pre() + ans for _ in range(2)]
    labels = [[-100] * (len(x) - len(ans)) + ans for x in ids]   # training rendering: the answer segment is supervised
    batch = _base_batch(model, tok, ids, labels)

    model.zero_grad()
    out = model.tf_forecast_loss(batch)
    assert math.isfinite(float(out["loss"])), out
    out["loss"].backward()
    assert _has_grad(model.chronos), "tf_forecast_loss should train chronos"
    assert _has_grad(model.feedback_qformer), "tf_forecast_loss should train feedback_qformer"
    assert not _has_grad(model.llm), "the LLM should be no_grad in tf_forecast_loss"
    print(f"TF frozen-LLM loss OK: loss={float(out['loss']):.3f}, grad reaches chronos+feedback, not the LLM ok")


def test_ss_in_training():
    """In-training scheduled sampling (the SS branch of forward_forecast):
    ss_teacher_ratio=0 forces the generation branch -- loss finite; text CE still gives the LLM
    gradients (teacher forward), pred/roi through the generated hidden states give gradients only to
    chronos+feedback; ratio=1 (default) equals the old path value for value."""
    import math

    model, tok = build_model()
    model.train()
    pre = lambda: [model.ts_start_id, model.ts_end_id] + tok("desc", add_special_tokens=False)["input_ids"]
    ans = tok(" up and down", add_special_tokens=False)["input_ids"]
    ids = [pre() + ans for _ in range(2)]
    labels = [[-100] * (len(x) - len(ans)) + ans for x in ids]
    batch = _base_batch(model, tok, ids, labels)
    # inference prefix (SS generation start): the prefix without the answer segment + all-ones mask
    plen = len(ids[0]) - len(ans)
    batch["ss_prefix_ids"] = batch["input_ids"][:, :plen].clone()
    batch["ss_prefix_mask"] = torch.ones_like(batch["ss_prefix_ids"])

    # ratio=1.0 (default): SS not triggered, output identical to the old path value for value
    torch.manual_seed(0)
    model.ss_teacher_ratio = 1.0
    out_ref = model.forward_forecast({k: v for k, v in batch.items()
                                      if k not in ("ss_prefix_ids", "ss_prefix_mask")})
    torch.manual_seed(0)
    out_same = model.forward_forecast(batch)
    assert torch.allclose(out_ref["loss"], out_same["loss"]), "ratio=1 should equal the old path value for value"

    # ratio=0.0: must take the generation branch
    model.ss_teacher_ratio = 0.0
    model.ss_max_new_tokens = 4
    model.zero_grad()
    out = model.forward_forecast(batch)
    assert math.isfinite(float(out["loss"])), out
    out["loss"].backward()
    assert _has_grad(model.llm), "text CE of an SS batch should still give the LLM gradients"
    assert _has_grad(model.chronos), "pred loss of an SS batch should give chronos gradients"
    assert _has_grad(model.feedback_qformer), "an SS batch should give feedback_qformer gradients"

    # Generation micro-batches (ss_gen_chunk=1) are equivalent to the whole batch -- this checks the
    # chunk slicing/concatenation mechanism itself, so three chunk-unrelated sources of difference
    # must first be removed: (1) monkeypatch generation to a fixed output (the tiny random model's
    # logits are near ties, and batch-dimension float differences are enough to flip the greedy
    # argmax); (2) zero all Dropout (in train mode one whole-batch call vs. two chunk calls necessarily
    # consume different dropout random streams); (3) equal-length contexts -- with variable-length
    # samples the soft-token counts differ, the whole batch is left-padded while a single sample is
    # not, and the full forward passes no position_ids (deliberately mirroring the inference path
    # _one_pass, which is exactly what SS must align with), so the RoPE position shift with padding is
    # an inherent property of the inference protocol, not a chunk bug.
    import types as _types
    from chronos_llm.tests.test_composite_wiring import left_pad_context as _lpc
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    ctx_eq, lens_eq = _lpc([[1.0, 2, 3, 4] * 12, [0.5, 1.0, 1.5, 2.0] * 12])
    batch_eq = {**batch, "context": ctx_eq, "true_lengths": lens_eq}
    real_gen = model.llm.generate
    def _fixed_gen(*, inputs_embeds=None, attention_mask=None, **kw):
        return _types.SimpleNamespace(
            sequences=torch.full((inputs_embeds.shape[0], 4), 5, dtype=torch.long))
    # Reset the seed before each of the two calls: the sdpa dropout_p of chronos2's
    # GroupSelfAttention is a float argument (not an nn.Dropout module, so the zeroing above does not
    # cover it), and a different RNG start would change the upstream encode output -- unrelated to
    # the chunk logic (with the same seed the hidden diff is 0 and the loss diff ~1e-5).
    model.llm.generate = _fixed_gen
    torch.manual_seed(0)
    out_w = model.forward_forecast(batch_eq)
    model.ss_gen_chunk = 1
    torch.manual_seed(0)
    out_c = model.forward_forecast(batch_eq)
    model.llm.generate = real_gen
    del model.ss_gen_chunk
    assert torch.allclose(out_w["loss"], out_c["loss"], rtol=1e-3), \
        f"chunk=1 should be equivalent to the whole batch: {float(out_w['loss'])} vs {float(out_c['loss'])}"

    # The callback's ss_force_gen takes precedence over the local random draw: with force=False the
    # teacher path is taken even at ratio=0
    model.ss_force_gen = False
    torch.manual_seed(0)
    out_f = model.forward_forecast(batch)
    torch.manual_seed(0)
    model.ss_teacher_ratio = 1.0
    model.ss_force_gen = None
    out_t = model.forward_forecast(batch)
    assert torch.allclose(out_f["loss"], out_t["loss"]), "force=False should take the teacher path"
    print(f"SS in-training OK: ratio=1 equals old path ok, ratio=0 generation branch loss={float(out['loss']):.3f} "
          f"gradients connected ok, chunked micro-batches equivalent ok, force_gen precedence ok")


if __name__ == "__main__":
    test_grpo_rollout()
    test_grpo_step()
    test_ss_forecast_loss()
    test_tf_forecast_loss()
    test_ss_in_training()
    print("ALL GRPO ROLLOUT TESTS PASSED")
