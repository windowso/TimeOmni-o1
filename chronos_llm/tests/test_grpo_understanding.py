"""GRPO understanding-branch wiring check (CPU, tiny Qwen2 standing in for the 9B model):
rollout_understanding_rl samples G groups of texts -> rl_logp is value-identical chunked vs
unchunked and grad reaches only the LLM -> answer reward extraction/scoring ->
grpo_understanding_step updates the LLM parameters in one step with finite metrics.
"""
import math
import os

import torch

from chronos_llm.rl.grpo_understanding import (compute_answer_rewards,
                                               grpo_understanding_step)
from chronos_llm.tests.test_forecast_ablations import _base_batch, _has_grad, build_model


def _batch(model, tok, with_meta=True):
    pre = lambda: [model.ts_start_id, model.ts_end_id] + tok("desc", add_special_tokens=False)["input_ids"]
    ids = [pre() + tok(" up", add_special_tokens=False)["input_ids"] for _ in range(2)]
    batch = _base_batch(model, tok, ids, [[-100] * len(x) for x in ids])
    if with_meta:
        batch["meta"] = [{"ground_truth": "Answer: up"}, {"ground_truth": "Answer: down"}]
    return batch


def test_understanding_rollout():
    model, tok = build_model()
    model.eval()
    batch = _batch(model, tok)
    G, B = 3, 2
    prefix_e, prefix_a, gen_ids, texts = model.rollout_understanding_rl(
        batch, group_size=G, temperature=1.0, max_new_tokens=5)   # default merge_groups=True
    assert len(gen_ids) == G and len(texts) == G
    assert prefix_e.dim() == 3 and prefix_e.shape[0] == B
    assert gen_ids[0].shape[0] == B and gen_ids[0].dim() == 2
    assert not prefix_e.requires_grad, "rollout should run entirely under no_grad"
    assert len(texts[0]) == B and all(isinstance(t, str) for t in texts[0])
    # Per-group fallback mode: same return structure as merged mode (distributional equivalence
    # follows from independent per-row sampling; values are not compared)
    _, _, gen_ids2, texts2 = model.rollout_understanding_rl(
        batch, group_size=2, temperature=1.0, max_new_tokens=5, merge_groups=False)
    assert len(gen_ids2) == 2 and gen_ids2[0].shape[0] == B and len(texts2[0]) == B

    # rl_logp chunked vs unchunked must be **value-identical** (softmax is over the vocab axis,
    # positions are independent; eval mode avoids dropout differences)
    lp_full = model.rl_logp(prefix_e, prefix_a, gen_ids[0], logits_chunk=0)
    lp_chunk = model.rl_logp(prefix_e, prefix_a, gen_ids[0], logits_chunk=2)
    assert lp_full.shape == (B,) and lp_full.requires_grad
    assert torch.allclose(lp_full, lp_chunk, rtol=1e-4, atol=1e-5), \
        f"chunked logp should equal the full-sequence value: {lp_full.tolist()} vs {lp_chunk.tolist()}"

    # grad reaches only the LLM, not chronos/history_qformer (GRPO trains the LLM only; prefix is
    # detached and rollout runs under no_grad)
    model.zero_grad()
    (-lp_chunk.mean()).backward()
    assert _has_grad(model.llm), "LLM should have grad"
    assert not _has_grad(model.chronos), "chronos should have no grad"
    assert not _has_grad(model.history_qformer), "history_qformer should have no grad"

    # Micro-batched logp (the memory-saving logp_micro_bs setting of the training entry point)
    # must be **value-identical** to the full batch: rows are independent and share the same left
    # padding => the forward of a sub-batch must equal the corresponding slice of the full result.
    lp_sub = model.rl_logp(prefix_e[0:1], prefix_a[0:1], gen_ids[0][0:1], logits_chunk=2)
    assert torch.allclose(lp_sub, lp_chunk[0:1], rtol=1e-4, atol=1e-5), \
        f"micro-batch logp should equal the full-batch slice: {lp_sub.tolist()} vs {lp_chunk[0:1].tolist()}"
    print("understanding rollout OK: G groups of text, chunked logp equivalent, grad only on LLM, micro-batch slice equivalent")


def test_answer_reward():
    """reward = exact match after extraction: both answer-first and CoT layouts, unmarked long
    text scores 0, an empty gt zeroes the whole column."""
    batch = {"meta": [{"ground_truth": "Answer: up"},
                      {"ground_truth": "Answer: down"},
                      {"ground_truth": ""}]}          # dirty data: extracted gt is empty => whole column 0 (no gradient)
    texts = [
        ["Answer: up\n\nBecause it rises.",           # correct (answer-first layout)
         "thinking...\n\nAnswer: up",                 # wrong (gt=down)
         "Answer: up"],                               # gt empty => 0
        ["nonsense " * 60,                            # unmarked, over-long => extraction empty => 0
         "some reasoning\n\nAnswer: down.",           # correct (trailing period is stripped by normalisation)
         "Answer: anything"],
    ]
    R = compute_answer_rewards(texts, batch)
    assert R.shape == (2, 3)
    assert R[0].tolist() == [1.0, 0.0, 0.0]
    assert R[1].tolist() == [0.0, 1.0, 0.0]
    print("answer reward OK: both layouts scored, unmarked long text / empty gt score 0")


def test_init_policy_merge_reopen():
    """init_merge_reopen option: merge the SFT LoRA into the LLM (in memory) -> record the
    lineage in config.init_merged_from -> open a brand-new zero-initialised LoRA. Persistence goes
    through lineage replay (_maybe_replay_merged_init, the same mechanism as the forecast-side
    --init_merge_reopen). Pins four things: the fresh B=0 starting point is value-identical to the
    continued-training arm; after perturbing the new adapter the save-reload round trip is
    value-identical (the new adapter really stacks back onto "base + SFT delta");
    _freeze_to_llm_lora leaves only the llm lora trainable; nested lineage is rejected.
    fake_from_config only replaces "rebuild the bare base"; replay runs the real code (same
    pattern as test_init_merge_reopen_lineage)."""
    import os
    import tempfile
    from unittest.mock import patch

    from chronos_llm.models.chronos_llm_model import (ChronosLLM, _maybe_replay_merged_init,
                                                      add_lora)
    from chronos_llm.rl.train_grpo import _freeze_to_llm_lora, _save_ckpt
    from chronos_llm.rl.train_grpo_understanding import init_policy
    from chronos_llm.tests.test_pretrained_peft import _build_tiny_base, _toy_batch

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
        fresh_b, _ = _build_tiny_base()
        fresh_b.load_state_dict(clean_sd, strict=True)
        fresh_b.config = config
        return _maybe_replay_merged_init(fresh_b, config)   # replay runs the real code; only the rebuild is mocked

    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        model.save_pretrained(da)
        model.get_base_model().config.save_pretrained(da)
        with patch.object(ChronosLLM, "from_config", staticmethod(fake_from_config)):
            fresh = init_policy(da, "cpu", init_merge_reopen=True)
            # lineage in config + fresh LoRA B all zero => starting policy value-identical to the continued arm
            assert fresh.get_base_model().config.init_merged_from == os.path.abspath(da)
            bs = [p for n, p in fresh.named_parameters() if "lora_B" in n]
            assert bs and all(float(p.abs().sum()) == 0.0 for p in bs), "fresh LoRA B should be zero-initialised"
            assert any("lora_A" in n and p.shape[0] == 4 for n, p in fresh.named_parameters())
            fresh.eval()
            with torch.no_grad():
                got = fresh(_toy_batch(fresh.get_base_model(), tok, "understanding"))["loss"].item()
            assert abs(ref - got) < 1e-4, f"init_merge_reopen start should be value-identical to the continued arm: {ref} vs {got}"

            # Perturb the new adapter (simulating training updates) -> save -> lineage reload; round trip value-identical
            with torch.no_grad():
                for p in fresh.parameters():
                    if p.requires_grad:
                        p.add_(0.01 * torch.randn_like(p))
            with torch.no_grad():
                got_in = fresh(_toy_batch(fresh.get_base_model(), tok, "understanding"))["loss"].item()
            _save_ckpt(fresh, db)
            loaded = ChronosLLM.from_pretrained(db, merge=True)
            loaded.eval()
            with torch.no_grad():
                got_rt = loaded(_toy_batch(loaded, tok, "understanding"))["loss"].item()
            assert abs(got_in - got_rt) < 1e-4, \
                f"lineage reload should be value-identical to in-process: {got_in} vs {got_rt}"

            # Nested lineage rejected: db's config already carries init_merged_from, so merge_reopen again must fail
            try:
                init_policy(db, "cpu", init_merge_reopen=True)
                raise AssertionError("nested lineage should be rejected")
            except ValueError:
                pass
        # After freezing only the llm lora remains trainable
        n_train = _freeze_to_llm_lora(fresh)
        assert n_train > 0
        for n, p in fresh.named_parameters():
            if p.requires_grad:
                assert "lora_" in n and "llm." in n, n
    print("init_policy merge_reopen OK: lineage in config, zero-init B start equivalent, "
          "save/lineage-reload round trip equivalent, nesting rejected, only llm lora trained")


def test_grpo_understanding_step():
    model, tok = build_model()
    model.train()
    batch = _batch(model, tok)
    # Freeze everything except the LLM (GRPO trains the LLM only)
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("llm."))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    before = next(p for n, p in model.named_parameters()
                  if n.startswith("llm.") and p.requires_grad).detach().clone()

    m = grpo_understanding_step(model, batch, opt, group_size=4, temperature=1.0,
                                max_new_tokens=4, logits_chunk=2, logp_micro_bs=1)
    for k in ("loss", "reward", "reward_std_in_group", "mixed_frac", "gen_len_mean", "grad_norm"):
        assert k in m and math.isfinite(m[k]), (k, m)
    assert m["gen_len_mean"] > 0
    after = next(p for n, p in model.named_parameters()
                 if n.startswith("llm.") and p.requires_grad)
    # A tiny random model's sampled group is most likely all wrong (reward all 0 => advantage all
    # 0 => no update); both that and "mixed group => update" are legal. Only verify that the
    # mechanism does not crash and the metric accounting is right.
    if m["reward_std_in_group"] > 0:
        assert (after.detach() - before).abs().sum() > 0, "LLM parameters should be updated when the group has variance"
    print(f"GRPO understanding step OK: { {k: round(v, 3) for k, v in m.items()} }")


def test_algo_variants():
    """Semantic sanity of the three algos on a deterministic stub:
    - all three have finite loss/metrics and do not crash;
    - rft has adv=R in {0,1} => loss = weighted sum of positive-sample NLL >= 0 (pure maximum
      likelihood, no negative-gradient term);
    - drgrpo uses the sum/constant-normalised logp path (the rl_logp length_norm=False branch is
      really exercised)."""
    from chronos_llm.tests.test_pretrained_peft import _build_tiny_base, _toy_batch

    for algo in ("grpo", "drgrpo", "rft"):
        torch.manual_seed(0)
        model, tok = _build_tiny_base()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("llm."))
        batch = _toy_batch(model, tok, "understanding")
        batch["meta"] = [{"ground_truth": "Answer: up"}, {"ground_truth": "Answer: down"}]
        B = batch["input_ids"].shape[0]

        def fake_rollout(b, **kw):
            with torch.no_grad():
                e = model.llm.get_input_embeddings()(b["input_ids"])
            gen = [torch.full((B, 3), 10, dtype=torch.long),
                   torch.full((B, 3), 20, dtype=torch.long)]
            texts = [["Answer: up", "Answer: nope"], ["Answer: nope", "Answer: nope"]]
            return e, b["attention_mask"], gen, texts       # R=[[1,0],[0,0]]

        model.rollout_understanding_rl = fake_rollout
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        m = grpo_understanding_step(model, batch, opt, group_size=2, max_new_tokens=3,
                                    logits_chunk=2, logp_micro_bs=1, algo=algo)
        for k, v in m.items():
            assert math.isfinite(v), (algo, k, m)
        if algo == "rft":
            assert m["loss"] > 0, f"rft loss should be the positive-sample NLL >= 0: {m['loss']}"
    print("algo variants OK: grpo/drgrpo/rft all run, rft loss = positive-sample NLL")


def test_pick_source():
    """Probability semantics and backward compatibility of three-pool source picking: direct is
    split off first, the remainder is split by focus_ratio; without direct it equals the old
    two-pool logic value for value; without rest everything non-direct is focus."""
    from chronos_llm.rl.train_grpo_understanding import pick_source

    # Verify the interval partition directly on boundary values:
    # direct=[0,0.2), focus=[0.2,0.2+0.8*0.7), rest=the remainder
    kw = dict(has_rest=True, has_direct=True, focus_ratio=0.7, direct_ratio=0.2)
    assert pick_source(0.0, **kw) == "direct" and pick_source(0.19, **kw) == "direct"
    assert pick_source(0.2, **kw) == "focus" and pick_source(0.75, **kw) == "focus"
    assert pick_source(0.76, **kw) == "rest" and pick_source(0.99, **kw) == "rest"
    # No direct: equals the old two-pool logic (focus_ratio threshold)
    kw2 = dict(has_rest=True, has_direct=False, focus_ratio=0.7, direct_ratio=0.2)
    assert pick_source(0.69, **kw2) == "focus" and pick_source(0.71, **kw2) == "rest"
    # No rest: everything non-direct is focus
    kw3 = dict(has_rest=False, has_direct=True, focus_ratio=0.7, direct_ratio=0.2)
    assert pick_source(0.1, **kw3) == "direct" and pick_source(0.9, **kw3) == "focus"
    print("pick_source OK: three-pool intervals, two-pool backward compatibility, no-rest fallback")


def _grpo_step_once(rank):
    """One deterministic GRPO step (for the distributed-equivalence test): build the tiny model
    with the same seed (identical initial weights across calls), stub the rollout
    deterministically (gen tokens vary with rank => different gradients per rank; texts mix
    correct/wrong => non-zero advantage), SGD without clipping (grad_clip=1e9 keeps it linear).
    Returns the trainable parameters after the update."""
    torch.manual_seed(0)
    from chronos_llm.tests.test_pretrained_peft import _build_tiny_base, _toy_batch
    model, tok = _build_tiny_base()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0                      # stubbed rollout + deterministic forward
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("llm."))
    batch = _toy_batch(model, tok, "understanding")
    batch["meta"] = [{"ground_truth": "Answer: up"}, {"ground_truth": "Answer: down"}]
    B = batch["input_ids"].shape[0]

    def fake_rollout(b, **kw):
        with torch.no_grad():
            e = model.llm.get_input_embeddings()(b["input_ids"])
        gen = [torch.full((B, 3), 10 + rank, dtype=torch.long),
               torch.full((B, 3), 20 + rank, dtype=torch.long)]
        # R=[[1,0],[0,0]]: sample 1 is all wrong => zero in-group variance => advantage=0, only
        # sample 0 contributes gradient.
        # Do NOT use the antisymmetric [[1,0],[0,1]]: the two _toy_batch samples have identical
        # inputs => their advantages are exact opposites and the gradients cancel term by term.
        texts = [["Answer: up", "Answer: nope"], ["Answer: nope", "Answer: nope"]]
        return e, b["attention_mask"], gen, texts

    model.rollout_understanding_rl = fake_rollout
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    grpo_understanding_step(model, batch, opt, group_size=2, max_new_tokens=3,
                            logits_chunk=2, logp_micro_bs=1, grad_clip=1e9)
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _dist_grpo_worker(rank, world, init_file, out_dir):
    """Spawned subprocess: run one step inside a real gloo process group (the step's internal
    _allreduce_grads takes effect) and save the weights."""
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world)
    try:
        sd = _grpo_step_once(rank)
        torch.save(sd, os.path.join(out_dir, f"rank{rank}.pt"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_grpo_dist_allreduce():
    """Multi-GPU data-parallel equivalence (real gloo, 2 processes): (1) after the step both ranks
    have value-identical weights (allreduce is symmetric); (2) they equal the element-wise mean of
    two single-process results -- one SGD step without clipping is linear, W = W0 - lr*g, so
    "average the per-rank gradients then step" == "step separately then average the weights",
    which gives a strict check without exposing internal gradients."""
    import os
    import tempfile

    ref0, ref1 = _grpo_step_once(0), _grpo_step_once(1)   # no process group: allreduce is a no-op
    # The stub gradients must really differ by rank, otherwise the averaging criterion is trivial
    assert any(not torch.allclose(ref0[n], ref1[n]) for n in ref0), "the two single-process references should differ"
    with tempfile.TemporaryDirectory() as d:
        torch.multiprocessing.spawn(
            _dist_grpo_worker, args=(2, os.path.join(d, "init"), d), nprocs=2, join=True)
        w0 = torch.load(os.path.join(d, "rank0.pt"))
        w1 = torch.load(os.path.join(d, "rank1.pt"))
    for n in ref0:
        assert torch.allclose(w0[n], w1[n], atol=1e-6), f"weights differ between the two ranks: {n}"
        ref = (ref0[n] + ref1[n]) / 2
        assert torch.allclose(w0[n], ref, rtol=1e-4, atol=1e-6), \
            f"allreduce result should equal the single-process average: {n}"
    print("GRPO dist allreduce OK: both ranks identical, == average of single-process results (SGD linearity criterion)")


def test_pool_balance():
    """With --pool_balance=1 the pool samples uniformly per dataset (60:6 two files -> ~50% each);
    the default 0 keeps the natural row proportion (small file ~9%), so one large dataset can
    dominate a pool."""
    import json as _json
    import tempfile
    import numpy as np
    from transformers import AutoTokenizer
    from chronos_llm.models.chronos_llm_model import _add_ts_special_tokens
    from chronos_llm.rl.train_grpo_understanding import _loader
    from chronos_llm.tests.test_composite_wiring import LLM

    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    with tempfile.TemporaryDirectory() as d:
        ts = os.path.join(d, "ts.npy")
        np.save(ts, np.random.randn(32).astype(np.float32))
        paths = []
        for name, n in [("big", 60), ("small", 6)]:
            p = os.path.join(d, f"{name}.jsonl")
            with open(p, "w") as f:
                for i in range(n):
                    f.write(_json.dumps({
                        "dataset_name": name, "input_text": [f"q{i}?"], "gt_text": ["yes"],
                        "input_ts": {"original": {"ori_path": ts}}}) + "\n")
            paths.append(p)

        class _A:
            base_dir = None; max_user_tokens = 64; max_tokens = 256
            sample_chunks = 0; batch_size = 6; pool_balance = 1

        def frac_small(args, seed):
            cnt = {"big": 0, "small": 0}
            for batch in _loader(paths, tok, args, seed):
                for m in batch["meta"]:
                    cnt[m["dataset_name"]] += 1
            return cnt["small"] / max(sum(cnt.values()), 1), sum(cnt.values())

        f1, n1 = frac_small(_A, seed=0)
        assert n1 == 66, f"one balanced-sampling pass should still yield 66 rows, got {n1}"
        assert 0.35 < f1 < 0.65, f"after balancing the small share should be ~0.5, got {f1:.3f}"
        _A.pool_balance = 0
        f0, n0 = frac_small(_A, seed=0)
        assert n0 == 66 and abs(f0 - 6 / 66) < 1e-9, \
            f"default path should give the natural proportion {6/66:.3f} (full pass without replacement), got {f0:.3f}"
    print(f"pool_balance OK: balanced small share {f1:.3f}~0.5, default natural proportion {f0:.3f}")


if __name__ == "__main__":
    test_understanding_rollout()
    test_answer_reward()
    test_init_policy_merge_reopen()
    test_grpo_understanding_step()
    test_algo_variants()
    test_pick_source()
    test_grpo_dist_allreduce()
    test_pool_balance()
    print("ALL GRPO UNDERSTANDING TESTS PASSED")
