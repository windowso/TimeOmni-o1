"""CPU verification of train.py's --init_from_checkpoint / --freeze_llm (tiny Qwen2 stands in for the 9B).

Scenario: continue training on an already trained forecasting-branch checkpoint -- the LLM is
frozen and no longer updated; only chronos2 + the two Q-formers adapt to the (optionally
self-generated) text feedback distribution. train.py reuses
ChronosLLM.from_pretrained(merge=False, is_trainable=True) for the "new optimizer, old weights"
continuation semantics (as opposed to the exact resume of --resume_from_checkpoint). Three things
must be verified: (1) the from_pretrained continuation load round-trips numerically;
(2) the freeze_llm rule (freeze llm.*/lora_*, keep chronos / the two Q-formers) hits the same set of
parameter names under the PEFT wrapper both at cold start and after a from_pretrained load;
(3) after one optimisation step the LLM/LoRA parameters are element-wise unchanged while
chronos/qformer really do update.
"""
import tempfile

import torch

from chronos_llm.models.chronos_llm_model import ChronosLLM, add_lora
from chronos_llm.tests.test_pretrained_peft import _build_tiny_base, _toy_batch


def _freeze_llm(model):
    """Mirror of train.py --freeze_llm's freezing rule (only chronos / the two Q-formers stay trainable)."""
    n_tr = 0
    for n, p in model.named_parameters():
        keep = ("chronos" in n) or ("feedback_qformer" in n) or ("history_qformer" in n)
        p.requires_grad_(keep)
        n_tr += int(keep)
    return n_tr


def test_freeze_llm_scope_and_grad():
    model, tok = _build_tiny_base()
    model = add_lora(model, r=4, alpha=8, dropout=0.0)
    base = model.get_base_model()
    for blk in base.chronos.encoder.block:
        blk.cross_attn.gate.data.fill_(0.3)  # open the gate so the feedback path is connected and gradients reach feedback_qformer

    named_before = {n: p.requires_grad for n, p in model.named_parameters()}
    n_tr = _freeze_llm(model)
    assert n_tr > 0

    for n, p in model.named_parameters():
        is_llm = ".llm." in ("." + n)
        if is_llm:
            assert not p.requires_grad, f"LLM/LoRA parameter not frozen: {n}"
        elif named_before[n]:
            # chronos/qformer parameters that were already trainable after add_lora (the active copy
            # of modules_to_save) must not be frozen back by freeze_llm. PEFT modules_to_save also
            # keeps a permanently frozen `.original_module.*` snapshot (forward never goes through it;
            # it is only the backup used by disable_adapter) -- it never takes part in training, so
            # freeze_llm's coarse substring match marking it True is harmless (no gradient to take),
            # and its before/after state is not asserted here.
            assert p.requires_grad, f"non-LLM trainable parameter unexpectedly frozen by freeze_llm: {n}"

    f = model(_toy_batch(base, tok, "forecast"))
    f["loss"].backward()
    lora_grad = any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
    assert not lora_grad, "LLM is frozen, LoRA parameters must not receive gradients"
    fb_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                  for n, p in model.named_parameters() if "feedback_qformer" in n and p.requires_grad)
    assert fb_grad, "with the LLM frozen, the feedback path (feedback_qformer) should still receive gradients"
    print("freeze_llm scope + grad isolation OK")


def test_init_from_checkpoint_continue_roundtrip_and_freeze():
    """save_pretrained -> from_pretrained(merge=False, is_trainable=True) numerical round-trip +
    freeze_llm hitting the same parameter names in the continuation scenario + one optimisation
    step moving only chronos/qformer."""
    from unittest.mock import patch

    base, tok = _build_tiny_base()
    clean_sd = {k: v.clone() for k, v in base.state_dict().items()}
    model = add_lora(base, r=4, alpha=8, dropout=0.0)
    b2 = model.get_base_model()
    for blk in b2.chronos.encoder.block:
        blk.cross_attn.gate.data.fill_(0.3)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    model.eval()
    # The forecast branch of _toy_batch builds the future with an unseeded torch.randn -- only the
    # result of a single call is deterministic; calling twice gives two different random targets.
    # Build once and reuse the same batch for both ref and got.
    batch = _toy_batch(model.get_base_model(), tok, "forecast")
    with torch.no_grad():
        ref = model(batch)["loss"].item()

    def fake_from_config(config):
        fresh, _ = _build_tiny_base()
        fresh.load_state_dict(clean_sd, strict=True)
        return fresh

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        model.get_base_model().config.save_pretrained(d)
        with patch.object(ChronosLLM, "from_config", staticmethod(fake_from_config)):
            loaded = ChronosLLM.from_pretrained(d, merge=False, is_trainable=True)
        from peft import PeftModel
        assert isinstance(loaded, PeftModel), "merge=False, is_trainable=True should return a trainable PeftModel"
        loaded_base = loaded.get_base_model()
        # The gate is part of the trainable chronos modules_to_save copy; save/load must bring back the
        # perturbed value as is (do not fill_ it again here, that would erase the perturbation and
        # deliberately manufacture a false positive that disagrees with ref).
        loaded.eval()
        with torch.no_grad():
            got = loaded(batch)["loss"].item()
        assert abs(ref - got) < 1e-4, f"init_from_checkpoint continuation load is numerically inconsistent: {ref} vs {got}"

        # Apply freeze_llm in the continuation scenario: must hit the same parameter names as the cold-start logic.
        n_tr = _freeze_llm(loaded)
        assert n_tr > 0
        llm_params = [p for n, p in loaded.named_parameters() if ".llm." in ("." + n)]
        assert llm_params and all(not p.requires_grad for p in llm_params)

        llm_before = {n: p.detach().clone() for n, p in loaded.named_parameters() if ".llm." in ("." + n)}
        chronos_before = {n: p.detach().clone() for n, p in loaded.named_parameters()
                          if p.requires_grad and ("chronos" in n or "feedback_qformer" in n
                                                   or "history_qformer" in n)}
        loaded.train()
        opt = torch.optim.SGD([p for p in loaded.parameters() if p.requires_grad], lr=0.1)
        out = loaded(_toy_batch(loaded_base, tok, "forecast"))
        out["loss"].backward()
        opt.step()

        for n, p in loaded.named_parameters():
            if n in llm_before:
                assert torch.equal(p.detach(), llm_before[n]), f"LLM parameter unexpectedly updated after one continuation step: {n}"
        moved = any(not torch.equal(p.detach(), chronos_before[n])
                    for n, p in loaded.named_parameters() if n in chronos_before)
        assert moved, "chronos/qformer parameters should have changed after one continuation step, but none did"
    print("init_from_checkpoint continue roundtrip + freeze + one-step update OK")


def test_init_from_checkpoint_gate_warmstart():
    """Init from a checkpoint whose feedback path was never trained (e.g. a pure
    understanding-branch product), then apply the gate warm start + out_proj zero init --
    train.py's _apply_gate_init/_apply_zero_cross_out_proj must hit every gate/out_proj on the
    PEFT-wrapped model without touching the other trained weights (an understanding checkpoint
    has gate == 0 and out_proj still randomly initialised, equivalent to a cold start)."""
    from unittest.mock import patch

    from chronos_llm.train import _apply_gate_init, _apply_zero_cross_out_proj

    base, tok = _build_tiny_base()
    clean_sd = {k: v.clone() for k, v in base.state_dict().items()}
    model = add_lora(base, r=4, alpha=8, dropout=0.0)
    # Simulate understanding training: chronos/qformer/LoRA get updated, but the gate (zero-initialised)
    # and the feedback path never received a gradient.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and not n.endswith(".cross_attn.gate"):
                p.add_(0.01 * torch.randn_like(p))
    for blk in model.get_base_model().chronos.encoder.block:
        assert float(blk.cross_attn.gate.abs().max()) == 0.0

    def fake_from_config(config):
        fresh, _ = _build_tiny_base()
        fresh.load_state_dict(clean_sd, strict=True)
        return fresh

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        model.get_base_model().config.save_pretrained(d)
        with patch.object(ChronosLLM, "from_config", staticmethod(fake_from_config)):
            loaded = ChronosLLM.from_pretrained(d, merge=False, is_trainable=True)

        probe = {n: p.detach().clone() for n, p in loaded.named_parameters()
                 if ("chronos" in n and ".cross_attn." not in n) or "history_qformer" in n}
        n_gate, prev_max = _apply_gate_init(loaded, 1.0)
        assert n_gate > 0 and prev_max == 0.0, f"gates should all start at 0: n={n_gate}, prev_max={prev_max}"
        n_op = _apply_zero_cross_out_proj(loaded)
        assert n_op > 0

        for n, p in loaded.named_parameters():
            if n.endswith(".cross_attn.gate"):
                assert torch.all(p.detach() == 1.0), f"gate not warm-started: {n}"
            if ".cross_attn.attn.out_proj." in ("." + n):
                assert float(p.detach().abs().max()) == 0.0, f"out_proj not zeroed: {n}"
            if n in probe:
                assert torch.equal(p.detach(), probe[n]), f"non-target weight unexpectedly modified: {n}"

        loaded.eval()
        with torch.no_grad():
            out = loaded(_toy_batch(loaded.get_base_model(), tok, "forecast"))
        assert torch.isfinite(out["loss"]), "forward loss should be finite after the gate warm start"
    print("init_from_checkpoint + gate warmstart + zero out_proj OK")


def test_init_merge_reopen_lineage():
    """Full chain of "merge, then open a new LoRA" (train.py --init_merge_reopen assembly + reload lineage):

    (1) _maybe_replay_merged_init replays ckpt_A's adapter onto the bare base; numerically ==
        from_pretrained(ckpt_A, merge=True); (2) the newly added LoRA has zero delta => forward is
        unchanged (the old delta is already baked into the base); (3) one training step moves only the
        new adapter; after saving ckpt_B (config carries init_merged_from=ckpt_A),
        from_pretrained(ckpt_B, merge=True) reloads to the same forward as inside the training
        process => the new adapter really is stacked on "original base + A delta" rather than on the
        original base; (4) nesting (merge_reopen on top of B) is rejected.
    fake_from_config only replaces "rebuild the bare base" (the real 9B cannot be loaded on CPU); the
    replay goes through the real code _maybe_replay_merged_init (the same function called at the end
    of the production from_config path)."""
    from unittest.mock import patch

    from chronos_llm.models.chronos_llm_model import (ChronosLLMConfig,
                                                      _maybe_replay_merged_init)

    base, tok = _build_tiny_base()
    clean_sd = {k: v.clone() for k, v in base.state_dict().items()}
    model = add_lora(base, r=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    b = model.get_base_model()
    for blk in b.chronos.encoder.block:
        blk.cross_attn.gate.data.fill_(0.3)  # open the gate: only then does the merge semantics show up in the forward

    def fake_from_config(config):
        fresh, _ = _build_tiny_base()
        fresh.load_state_dict(clean_sd, strict=True)
        fresh.config = config  # the production from_config builds the model from the passed config (lineage is saved with it); the fake mirrors that
        return _maybe_replay_merged_init(fresh, config)  # replay goes through the real code; only the rebuild is mocked

    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        model.save_pretrained(da)
        model.get_base_model().config.save_pretrained(da)
        batch = _toy_batch(model.get_base_model(), tok, "forecast")

        with patch.object(ChronosLLM, "from_config", staticmethod(fake_from_config)):
            # (1) replay == merge=True load (the same adapter reaches the same base via two paths)
            ref = ChronosLLM.from_pretrained(da, merge=True)
            cfg_a = ChronosLLMConfig.from_pretrained(da)
            cfg_a.init_merged_from = da
            merged = fake_from_config(cfg_a)  # simulate the from_config call of train.py merge_reopen
            for (n1, p1), (n2, p2) in zip(ref.state_dict().items(), merged.state_dict().items()):
                assert n1 == n2 and torch.equal(p1, p2), f"replayed weights differ from the merge load: {n1}"

            # (2) newly opened LoRA has zero delta => forward strictly unchanged
            merged.eval()
            with torch.no_grad():
                loss_merged = merged(batch)["loss"].item()
            m2 = add_lora(merged, r=4, alpha=8, dropout=0.0)
            m2.eval()
            with torch.no_grad():
                loss_reopen = m2(batch)["loss"].item()
            assert abs(loss_merged - loss_reopen) < 1e-5, \
                f"forward should be unchanged under a zero-delta new LoRA: {loss_merged} vs {loss_reopen}"

            # (3) one training step -> save ckpt_B (config carries the lineage) -> reload is numerically consistent
            m2.train()
            opt = torch.optim.SGD([p for p in m2.parameters() if p.requires_grad], lr=0.05)
            m2(_toy_batch(m2.get_base_model(), tok, "forecast"))["loss"].backward()
            opt.step()
            m2.eval()
            with torch.no_grad():
                loss_trained = m2(batch)["loss"].item()
            m2.save_pretrained(db)
            m2.get_base_model().config.save_pretrained(db)
            cfg_b = ChronosLLMConfig.from_pretrained(db)
            assert cfg_b.init_merged_from == da, "ckpt_B's config should carry the init_merged_from lineage"
            reloaded = ChronosLLM.from_pretrained(db, merge=True)
            reloaded.eval()
            with torch.no_grad():
                loss_reloaded = reloaded(batch)["loss"].item()
            assert abs(loss_trained - loss_reloaded) < 1e-4, \
                f"merge_reopen reload lineage broken: in-training {loss_trained} vs reloaded {loss_reloaded}"

            # (4) nesting rejected: B already carries init_merged_from, merge_reopen on it must raise
            cfg_nest = ChronosLLMConfig.from_pretrained(db)
            cfg_nest.init_merged_from = db
            try:
                fake_from_config(cfg_nest)
                raise AssertionError("nested init_merged_from should be rejected")
            except ValueError:
                pass
    print("init_merge_reopen lineage (replay/zero-delta/save-reload/nesting) OK")


def main():
    test_freeze_llm_scope_and_grad()
    test_init_from_checkpoint_continue_roundtrip_and_freeze()
    test_init_from_checkpoint_gate_warmstart()
    test_init_merge_reopen_lineage()
    print("ALL FREEZE_LLM CONTINUE TESTS PASSED")


if __name__ == "__main__":
    main()
