"""CPU unit tests for token-budget dynamic batch sizing.

Checks:
1. Dynamic packing: per batch sum(token) <= main budget (except a single over-budget sample that
   gets its own batch), sum(patch) <= secondary budget, batch size <= cap; no sample lost or
   duplicated (no drop_last in dynamic mode); __len__ == true batch count; reproducible within an
   epoch, reshuffled across epochs; short samples form large batches, long samples small ones.
2. With budget=0 the behaviour is identical to the fixed-bs baseline.
3. DDP: equal batch counts across ranks, disjoint samples.
4. ``token_lengths``: within <=8 tokens per sample of the input_ids length of the real
   ``build_supervised_ids`` rendering (batch tokenization + scaffold constant concatenation-boundary
   error), and identical value for value once the disk cache is hit.
5. Trainer wiring smoke test: tiny model + budget -> every dataloader batch measured
   sum(text+soft) <= budget.
6. bs-ladder quantization (``bs_ladder``): every size lands on a rung, nothing lost or duplicated,
   budgets still satisfied; disabled == baseline.
"""
import json
import os
import tempfile

import numpy as np
import torch

from chronos_llm.data.chat_utils import build_supervised_ids
from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import DualBranchBatchSampler
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset

N = 200


def _flat(batches):
    return [i for b in batches for i in b]


def test_dynamic_packing():
    rng = np.random.RandomState(0)
    tok = rng.randint(100, 2000, size=N).tolist()    # token cost (widely varying lengths)
    patch = rng.randint(10, 5000, size=N).tolist()   # chronos patch secondary cost
    BUDGET, PBUDGET, MAXBS = 4096, 60000, 16
    s = DualBranchBatchSampler(
        N, 0, 8, 2, seed=0, pool_factor=5,
        understanding_costs=tok,  # sorting proxy (dynamic mode actually sorts by token cost)
        understanding_token_costs=tok, understanding_patch_costs=patch,
        understanding_token_budget=BUDGET, patch_budget=PBUDGET, max_dynamic_bs=MAXBS,
    )
    b0 = list(s)
    assert sorted(_flat(b0)) == list(range(N)), "dynamic mode must not lose or duplicate samples"
    assert len(s) == len(b0), "__len__ must equal the true batch count"
    sizes = [len(b) for b in b0]
    assert max(sizes) > min(sizes), f"batch size should vary with widely varying lengths: {sizes}"
    assert max(sizes) <= MAXBS
    for b in b0:
        assert len(b) == 1 or sum(tok[i] for i in b) <= BUDGET, (b, sum(tok[i] for i in b))
        assert len(b) == 1 or sum(patch[i] for i in b) <= PBUDGET
    # determinism / reshuffle
    s.set_epoch(0); a = list(s)
    s.set_epoch(0); b = list(s)
    s.set_epoch(1); c = list(s)
    assert a == b and a != c
    print(f"dynamic packing OK: {len(b0)} batches, size in [{min(sizes)},{max(sizes)}]")


def test_patch_token_weight():
    """With lambda>0 the token budget is charged by the unified cost tok+lambda*patch: per batch
    sum(tok+lambda*p) <= budget (except single-sample batches), no sample lost or duplicated;
    lambda=0 behaves bit-for-bit like the baseline."""
    rng = np.random.RandomState(1)
    tok = rng.randint(100, 2000, size=N).tolist()
    patch = rng.randint(10, 5000, size=N).tolist()
    BUDGET, LAM = 8192, 0.5
    kw = dict(understanding_token_costs=tok, understanding_patch_costs=patch,
              understanding_token_budget=BUDGET, max_dynamic_bs=64)
    sw = DualBranchBatchSampler(N, 0, 8, 2, seed=0, pool_factor=5,
                                patch_token_weight=LAM, **kw)
    bw = list(sw)
    assert sorted(_flat(bw)) == list(range(N))
    for b in bw:
        cost = sum(tok[i] + LAM * patch[i] for i in b)
        assert len(b) == 1 or cost <= BUDGET, (b, cost)
    s0a = DualBranchBatchSampler(N, 0, 8, 2, seed=0, pool_factor=5, **kw)
    s0b = DualBranchBatchSampler(N, 0, 8, 2, seed=0, pool_factor=5,
                                 patch_token_weight=0.0, **kw)
    assert list(s0a) == list(s0b), "lambda=0 must be bit-for-bit identical to the baseline"
    assert bw != list(s0a), "lambda>0 should change the packing (patch-heavy samples take more budget)"
    print(f"patch_token_weight OK: {len(bw)} batches (lambda={LAM})")


def test_bs_ladder():
    """bs-ladder quantization: every batch size is in the rung set (1 added automatically), no sample
    lost or duplicated, budget constraints still hold (trimming takes a subset + the carry-over chain
    is re-cut by the while loop, so neither token nor patch budget is exceeded); with the ladder
    disabled (default None) identical to the baseline bit for bit (the while degenerates to an if)."""
    rng = np.random.RandomState(7)
    tok = rng.randint(100, 2000, size=N).tolist()
    patch = rng.randint(10, 5000, size=N).tolist()
    BUDGET, PBUDGET, MAXBS = 4096, 60000, 16
    LADDER = (2, 3, 4, 6, 8, 12, 16)  # deliberately without 1, to check the automatic addition
    kw = dict(understanding_costs=tok, understanding_token_costs=tok,
              understanding_patch_costs=patch, understanding_token_budget=BUDGET,
              patch_budget=PBUDGET, max_dynamic_bs=MAXBS)
    s = DualBranchBatchSampler(N, 0, 8, 2, seed=0, pool_factor=5, bs_ladder=LADDER, **kw)
    b = list(s)
    assert sorted(_flat(b)) == list(range(N)), "quantization must not lose or duplicate samples"
    allowed = {1, *LADDER}
    assert {len(x) for x in b} <= allowed, f"batch sizes must be on the ladder: {sorted({len(x) for x in b})}"
    for x in b:
        assert len(x) == 1 or sum(tok[i] for i in x) <= BUDGET
        assert len(x) == 1 or sum(patch[i] for i in x) <= PBUDGET
    assert len(s) == len(b)
    # ladder disabled == baseline (the default-parameter path is unaffected by the while rewrite)
    s_off = DualBranchBatchSampler(N, 0, 8, 2, seed=0, pool_factor=5, **kw)
    s_ref = DualBranchBatchSampler(N, 0, 8, 2, seed=0, pool_factor=5, bs_ladder=None, **kw)
    assert list(s_off) == list(s_ref)
    print(f"bs-ladder quantization OK: {len(b)} batches, size distribution {sorted({len(x) for x in b})}")


def test_budget_off_equals_fixed():
    s_dyn_off = DualBranchBatchSampler(N, 0, 8, 2, seed=3)
    s_dyn_off.set_epoch(2)
    got = list(s_dyn_off)
    g = torch.Generator(); g.manual_seed(3 + 2)
    idx = torch.randperm(N, generator=g).tolist()
    want = [idx[i * 8:(i + 1) * 8] for i in range(N // 8)]
    assert got == want, "with the budget off, behaviour must match the fixed-bs baseline"
    assert len(s_dyn_off) == len(want)
    print("budget off == fixed bs OK")


def test_ddp_dynamic():
    tok = list(np.random.RandomState(1).randint(100, 2000, size=N))
    shards = []
    for rank in range(2):
        s = DualBranchBatchSampler(
            N, 0, 8, 2, num_replicas=2, rank=rank, seed=0, pool_factor=5,
            understanding_token_costs=tok, understanding_token_budget=4096,
        )
        shards.append(list(s))
    assert len(shards[0]) == len(shards[1])
    assert not (set(_flat(shards[0])) & set(_flat(shards[1])))
    print("DDP dynamic-packing sharding OK")


def test_token_lengths_precise():
    from chronos_llm.models.chronos_llm_model import _add_ts_special_tokens
    from chronos_llm.tests.test_pretrained_peft import LLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    with tempfile.TemporaryDirectory() as d:
        # understanding: QA pairs of different lengths
        rows = [{"input_text": [f"question {'with context ' * k}?"], "gt_text": [f"answer {'long ' * k}."],
                 "input_ts": {"channel": 1, "original": {"ori_length": 100, "ori_path": "x.npy"}}}
                for k in (0, 5, 40)]
        jl = os.path.join(d, "u.jsonl")
        with open(jl, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
        ds = UnderstandingJsonlDataset([jl], tok, max_user_tokens=128, max_tokens=512)
        est = ds.token_lengths(cache_dir=d)
        for i, s in enumerate(ds.data):
            true_len = len(build_supervised_ids(tok, s["input_text"][0], ds._answer(s),
                                                max_user_tokens=128, max_tokens=512)[0])
            assert abs(est[i] - true_len) <= 8, (i, est[i], true_len)
        assert ds.token_lengths(cache_dir=d) == est, "must be identical value for value after a cache hit"
        assert any(f.startswith("toklen_") for f in os.listdir(d)), "a cache file should be produced"

        # forecast: reasoning+conclusion rendering
        import pandas as pd
        h, fu = np.zeros(64, np.float32), np.zeros(8, np.float32)
        prows = [{"history_values": h.tolist(), "future_values": fu.tolist(),
                  "background": "bg " * k, "event": "ev", "prompt": "predict",
                  "reasoning": "because trend " * k, "conclusion": "it goes up.",
                  "past_len": 64, "roi_start_idx": 64, "roi_end_idx": 66} for k in (1, 10)]
        pq = os.path.join(d, "f.parquet")
        pd.DataFrame(prows).to_parquet(pq)
        fds = ForecastParquetDataset(pq, tok, split=None, max_user_tokens=128, max_tokens=512)
        fest = fds.token_lengths(cache_dir=d)
        for i in range(len(fds)):
            row = fds.df.iloc[i]
            true_len = len(build_supervised_ids(tok, fds._user_text(row), str(row["conclusion"]).strip(),
                                                reasoning_content=str(row["reasoning"]).strip(),
                                                max_user_tokens=128, max_tokens=512)[0])
            assert abs(fest[i] - true_len) <= 8, (i, fest[i], true_len)
    print("token_lengths precision (+-8 tokens) + cache OK")


def test_trainer_wiring():
    """Tiny model + budget: for every dataloader batch, sum(text tokens + soft tokens) <= budget."""
    from chronos_llm.data.sampler import ConcatBranchDataset
    from chronos_llm.tests.test_pretrained_peft import LLM, _build_tiny_base
    from chronos_llm.trainer import ChronosLLMTrainer
    from transformers import TrainingArguments

    model, tok = _build_tiny_base()
    with tempfile.TemporaryDirectory() as d:
        rows = [{"input_text": [f"q {'ctx ' * (3 * k)}?"], "gt_text": ["yes"],
                 "input_ts": {"channel": 1, "original": {"ori_length": 50, "ori_path": "x.npy"}}}
                for k in range(12)]
        jl = os.path.join(d, "u.jsonl")
        with open(jl, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
        u_ds = UnderstandingJsonlDataset([jl], tok, max_user_tokens=64, max_tokens=256)
        concat = ConcatBranchDataset(u_ds, None)
        BUDGET = 200
        targs = TrainingArguments(output_dir=os.path.join(d, "out"), per_device_train_batch_size=2,
                                  max_steps=1, use_cpu=True, remove_unused_columns=False,
                                  label_names=["labels"], report_to=[])
        trainer = ChronosLLMTrainer(
            model=model, args=targs, train_dataset=concat, processing_class=tok,
            understanding_dataset=u_ds, forecast_dataset=None,
            understanding_bs=4, forecast_bs=2, length_pool_factor=2,
            understanding_token_budget=BUDGET, token_cache_dir=None,
        )
        dl = trainer.get_train_dataloader()
        text = u_ds.token_lengths()
        pc = u_ds.history_patches(2048, patch=model.input_patch_size)
        soft = [model.history_qformer.num_tokens_for_length(p, c) for p, c in pc]
        n_batches, sizes = 0, []
        for batch in dl.batch_sampler:
            cost = sum(text[i] + soft[i] for i in batch)
            assert len(batch) == 1 or cost <= BUDGET, (batch, cost)
            n_batches += 1; sizes.append(len(batch))
        assert sorted(_flat(list(dl.batch_sampler))) == list(range(12))
    print(f"trainer dynamic-packing wiring OK: {n_batches} batches, size={sizes}")


if __name__ == "__main__":
    test_dynamic_packing()
    test_patch_token_weight()
    test_bs_ladder()
    test_budget_off_equals_fixed()
    test_ddp_dynamic()
    test_token_lengths_precise()
    test_trainer_wiring()
    print("ALL OK")
