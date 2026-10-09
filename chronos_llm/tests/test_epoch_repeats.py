"""CPU unit test of per-epoch branch repeat counts (understanding_repeats / forecast_repeats).

Checks:
1. Fixed bs: with forecast_repeats=3 every forecast sample appears exactly 3 times and every
   understanding sample exactly once; each pass covers the full set (nothing lost or duplicated
   within a pass); __len__ == real batch count; interleaving spreads forecast batches evenly.
2. Independent shuffle per pass: the two passes within one epoch have different sample orders;
   reshuffled across epochs, reproducible within an epoch.
3. repeats=1 is identical batch by batch to the legacy behaviour (regression guard).
4. Dynamic batching + repeats: per-sample occurrence == repeats, every batch still within budget.
5. DDP: batches of different ranks are pairwise disjoint (within a pass), and over all ranks each
   sample occurs == repeats times (up to the (world-1) batches dropped by shard truncation).
"""
from collections import Counter

import numpy as np

from chronos_llm.data.sampler import DualBranchBatchSampler


def _flat(batches):
    return [i for b in batches for i in b]


def test_fixed_bs_repeats():
    n_u, n_f, u_bs, f_bs = 40, 12, 8, 4
    s = DualBranchBatchSampler(n_u, n_f, u_bs, f_bs, seed=0, forecast_repeats=3)
    batches = list(s)
    assert len(s) == len(batches)
    u_batches = [b for b in batches if b[0] < n_u]
    f_batches = [b for b in batches if b[0] >= n_u]
    assert len(u_batches) == n_u // u_bs and len(f_batches) == 3 * (n_f // f_bs)
    cnt_u = Counter(_flat(u_batches)); cnt_f = Counter(_flat(f_batches))
    assert all(v == 1 for v in cnt_u.values()) and len(cnt_u) == n_u, "each understanding sample exactly once"
    assert all(v == 3 for v in cnt_f.values()) and len(cnt_f) == n_f, "each forecast sample exactly 3 times"
    # Full coverage per pass: cut the forecast batches into 3 segments in output order; each is exactly the full set
    per_pass = n_f // f_bs
    for k in range(3):
        seg = _flat(f_batches[k * per_pass : (k + 1) * per_pass])
        assert sorted(seg) == list(range(n_u, n_u + n_f)), f"pass {k} must cover the full set"
    # Independent shuffle per pass: at least two passes differ in order
    orders = [tuple(_flat(f_batches[k * per_pass : (k + 1) * per_pass])) for k in range(3)]
    assert len(set(orders)) > 1, "passes should be shuffled independently"
    # Even interleaving: forecast batches should not pile up at the end (both halves contain some)
    half = len(batches) // 2
    assert any(b[0] >= n_u for b in batches[:half]) and any(b[0] >= n_u for b in batches[half:])
    # Reproducibility / reshuffle
    s.set_epoch(0); a = list(s)
    s.set_epoch(0); b = list(s)
    s.set_epoch(1); c = list(s)
    assert a == b and a != c
    print(f"fixed-bs repeats OK: {len(u_batches)}u + {len(f_batches)}f batches")


def test_repeats_one_is_noop():
    n_u, n_f = 40, 12
    base = DualBranchBatchSampler(n_u, n_f, 8, 4, seed=7)
    rep = DualBranchBatchSampler(n_u, n_f, 8, 4, seed=7,
                                 understanding_repeats=1, forecast_repeats=1)
    base.set_epoch(2); rep.set_epoch(2)
    assert list(base) == list(rep), "repeats=1 must match the legacy behaviour batch by batch"
    print("repeats=1 regression OK")


def test_dynamic_budget_repeats():
    rng = np.random.RandomState(0)
    n_u, n_f = 30, 25
    u_tok = rng.randint(100, 1500, size=n_u).tolist()
    f_tok = rng.randint(100, 1500, size=n_f).tolist()
    BUDGET = 4096
    s = DualBranchBatchSampler(
        n_u, n_f, 8, 4, seed=0,
        understanding_token_costs=u_tok, forecast_token_costs=f_tok,
        understanding_token_budget=BUDGET, forecast_token_budget=BUDGET,
        understanding_repeats=2, forecast_repeats=3,
    )
    batches = list(s)
    cnt_u = Counter(i for b in batches for i in b if i < n_u)
    cnt_f = Counter(i for b in batches for i in b if i >= n_u)
    assert all(v == 2 for v in cnt_u.values()) and len(cnt_u) == n_u
    assert all(v == 3 for v in cnt_f.values()) and len(cnt_f) == n_f
    for b in batches:
        tok = [(u_tok[i] if i < n_u else f_tok[i - n_u]) for i in b]
        assert len(b) == 1 or len(b) * max(tok) <= BUDGET, "the padded budget must still hold"
    assert len(s) == len(batches)
    print(f"dynamic batching repeats OK: {len(batches)} batches")


def test_ddp_repeats():
    n_u, n_f, world = 40, 12, 2
    per_rank = []
    for rank in range(world):
        s = DualBranchBatchSampler(n_u, n_f, 8, 4, num_replicas=world, rank=rank,
                                   seed=0, forecast_repeats=3)
        per_rank.append(list(s))
    lens = {len(b) for b in per_rank}
    assert len(lens) == 1, "all ranks must have the same number of batches"
    cnt = Counter(_flat([b for r in per_rank for b in r]))
    # Shard truncation drops at most (world-1) batches => the vast majority of forecast samples
    # occur exactly 3 times and understanding samples exactly once, never more
    assert all(v <= 3 for i, v in cnt.items() if i >= n_u)
    assert all(v <= 1 for i, v in cnt.items() if i < n_u)
    n3 = sum(1 for i, v in cnt.items() if i >= n_u and v == 3)
    assert n3 >= n_f - (world - 1) * 4, f"most forecast samples should complete 3 passes: {n3}/{n_f}"
    print(f"DDP repeats OK: {len(per_rank[0])} batches per rank")


if __name__ == "__main__":
    test_fixed_bs_repeats()
    test_repeats_one_is_noop()
    test_dynamic_budget_repeats()
    test_ddp_repeats()
    print("ALL OK")
