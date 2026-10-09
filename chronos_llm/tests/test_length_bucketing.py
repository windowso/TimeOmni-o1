"""CPU unit test of the length-bucketing batch sampler (pure python, no model/tokenizer needed).

Checks:
1. ``pool_factor=0`` (default off): batches == legacy behaviour (shuffle then cut sequentially,
   identical batch by batch).
2. Bucketing on: no sample lost or duplicated (within drop_last semantics), reproducible within an
   epoch, reshuffled across epochs, the in-batch cost spread is much smaller than for random
   batches, and the most expensive sample lands in the first batch (OOM exposed early).
3. DDP sharding: batches of different ranks are disjoint and equal in number.
4. ``cost_keys`` of both datasets: monotone in history length / channel count / text length and
   aligned with the sample count (after understanding's empty-answer filtering).
5. Automatic interleaving ratio: when the two batch types are very unbalanced, the rarer one is
   spread evenly over the epoch and is not exhausted early.
"""
import json
import os
import tempfile

import numpy as np
import torch

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import DualBranchBatchSampler
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset

N_U, U_BS = 200, 8


def _flat(batches):
    return [i for b in batches for i in b]


def test_off_equals_legacy():
    """pool_factor=0: batches = randperm cut sequentially (identical batch by batch to the
    pre-refactor implementation)."""
    s = DualBranchBatchSampler(N_U, 0, U_BS, 2, seed=7)
    s.set_epoch(3)
    got = list(s)
    g = torch.Generator(); g.manual_seed(7 + 3)
    idx = torch.randperm(N_U, generator=g).tolist()
    want = [idx[i * U_BS:(i + 1) * U_BS] for i in range(N_U // U_BS)]
    assert got == want, "with bucketing off the batches must match the legacy implementation one by one"
    print("pool_factor=0 == legacy behaviour OK")


def test_bucketing_properties():
    costs = (torch.arange(N_U).float() * 3.7).tolist()  # cost monotone in the index
    kw = dict(understanding_costs=costs, pool_factor=5, seed=0)
    s = DualBranchBatchSampler(N_U, 0, U_BS, 2, **kw)
    s.set_epoch(0); b0 = list(s)
    s.set_epoch(0); b0b = list(s)
    s.set_epoch(1); b1 = list(s)
    assert b0 == b0b, "should be reproducible within an epoch"
    assert b0 != b1, "should reshuffle across epochs"
    assert sorted(_flat(b0)) == list(range(N_U)), "no sample lost or duplicated"
    # In-batch cost spread: after bucketing it should be far smaller than for random batches
    def spread(batches):
        return float(np.mean([max(costs[i] for i in b) - min(costs[i] for i in b) for b in batches]))
    rand = list(DualBranchBatchSampler(N_U, 0, U_BS, 2, seed=0))
    assert spread(b0) < spread(rand) / 3, (spread(b0), spread(rand))
    # The most expensive sample (index N_U-1) is in the first batch
    assert (N_U - 1) in b0[0], "the globally most expensive sample should land in the first batch (OOM exposed early)"
    print(f"bucketing properties OK: in-batch spread {spread(b0):.1f} << random {spread(rand):.1f}")


def test_ddp_shard():
    costs = list(np.random.RandomState(0).rand(N_U))
    shards = []
    for rank in range(2):
        s = DualBranchBatchSampler(N_U, 0, U_BS, 2, num_replicas=2, rank=rank,
                                   understanding_costs=costs, pool_factor=5, seed=0)
        shards.append(list(s))
    assert len(shards[0]) == len(shards[1]), "all ranks must have the same number of batches"
    assert not (set(_flat(shards[0])) & set(_flat(shards[1]))), "samples must be disjoint across ranks"
    print("DDP sharding OK")


def test_dataset_cost_keys():
    with tempfile.TemporaryDirectory() as d:
        # understanding: long/short history + multi-channel + empty answer (filtered; keys must align with the filtered set)
        rows = [
            {"input_text": ["q"], "gt_text": ["a"],
             "input_ts": {"channel": 1, "original": {"ori_length": 160, "ori_path": "x.npy"}}},
            {"input_text": ["q"], "gt_text": ["a"],
             "input_ts": {"channel": 4, "original": {"ori_length": 160000, "ori_path": "y.npy"}}},
            {"input_text": ["q"], "gt_text": [""],  # empty answer -> filtered in init
             "input_ts": {"channel": 1, "original": {"ori_length": 99, "ori_path": "z.npy"}}},
        ]
        p = os.path.join(d, "u.jsonl")
        with open(p, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
        ds = UnderstandingJsonlDataset([p], tokenizer=None)
        keys = ds.cost_keys()
        assert len(keys) == len(ds) == 2
        assert keys[1] > keys[0] * 100, "long history x multi-channel should cost far more than a short history"

        # forecast: long vs short text, long vs short history
        import pandas as pd
        h_s, h_l = np.zeros(16, np.float32), np.zeros(4096, np.float32)
        fu = np.zeros(4, np.float32)
        def row(h, txt):
            return {"history_values": h.tolist(), "future_values": fu.tolist(),
                    "background": txt, "event": "", "prompt": "p",
                    "reasoning": "r", "conclusion": "c",
                    "past_len": len(h), "roi_start_idx": len(h), "roi_end_idx": len(h) + 2}
        pq = os.path.join(d, "f.parquet")
        pd.DataFrame([row(h_s, "x"), row(h_l, "x"), row(h_s, "x" * 4000)]).to_parquet(pq)
        fds = ForecastParquetDataset(pq, tokenizer=None, split=None)
        fk = fds.cost_keys()
        assert fk[1] > fk[0] and fk[2] > fk[0], fk
    print("dataset cost_keys OK")


def test_auto_interleave():
    """When understanding far outnumbers forecast, forecast batches should be spread evenly by
    their actual share and not be exhausted in the first half."""
    n_u, n_f, u_bs, f_bs = 160, 8, 8, 2  # 20 understanding batches + 4 forecast batches
    s = DualBranchBatchSampler(n_u, n_f, u_bs, f_bs, seed=0)
    batches = list(s)
    assert len(batches) == 24
    # Positions of the forecast batches (indices >= n_u in the concatenated index space)
    f_pos = [k for k, b in enumerate(batches) if b[0] >= n_u]
    assert len(f_pos) == 4 and all(len(batches[k]) == f_bs for k in f_pos)
    assert sorted(_flat(batches)) == list(range(n_u + n_f)), "no sample lost or duplicated"
    # Evenness: the gap between adjacent forecast batches is at most 2x the theoretical gap
    # (24/4=6), and the last one falls in the final quarter
    gaps = [b - a for a, b in zip([-1] + f_pos, f_pos + [len(batches)])]
    assert max(gaps) <= 12, f"forecast batches are spread unevenly: {f_pos}"
    assert f_pos[-1] >= 18, f"forecast batches should not be exhausted early: {f_pos}"
    # All-empty / single-branch edge cases
    assert list(DualBranchBatchSampler(0, 0, 4, 2)) == []
    only_u = list(DualBranchBatchSampler(16, 0, 4, 2, seed=1))
    assert len(only_u) == 4 and all(b[0] < 16 for b in only_u)
    print(f"automatic interleaving OK: forecast batch positions {f_pos}")


if __name__ == "__main__":
    test_off_equals_legacy()
    test_bucketing_properties()
    test_ddp_shard()
    test_dataset_cost_keys()
    test_auto_interleave()
    print("ALL OK")
