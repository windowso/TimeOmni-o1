"""Dual-branch data assembly: concatenate the understanding and forecasting datasets and use a custom
batch sampler that yields **homogeneous batches** (each batch comes entirely from one branch),
interleaved according to their actual proportions with per-branch batch sizes, and sharded for DDP.
"""

import math
import warnings
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler


class ConcatBranchDataset(Dataset):
    """Concatenate the two branch datasets; forecasting indices are offset by len(understanding)."""

    def __init__(self, understanding: Dataset | None, forecast: Dataset | None):
        self.understanding = understanding
        self.forecast = forecast
        self.n_u = len(understanding) if understanding is not None else 0
        self.n_f = len(forecast) if forecast is not None else 0

    def __len__(self):
        return self.n_u + self.n_f

    def __getitem__(self, idx):
        if idx < self.n_u:
            return self.understanding[idx]
        return self.forecast[idx - self.n_u]


class DualBranchBatchSampler(Sampler):
    """Yield index lists forming homogeneous batches, evenly interleaved by the actual number of batches of
    each kind; supports DDP sharding and per-epoch shuffling.

    In every epoch each branch is produced in full ``*_repeats`` times (default once each); the interleaving
    ratio does not change the total, it only decides the order -- hence the ratio **must** equal the actual
    share of batches (computed automatically). A manually chosen ratio would only exhaust the smaller branch
    early and degrade the tail of the epoch into a single task.

    ``understanding_repeats`` / ``forecast_repeats`` (how many full passes over that branch per epoch, >= 1)
    let the two branches make different numbers of passes within one outer epoch. Each pass
    is shuffled independently (drawn in sequence from the same generator: deterministic, different order per
    pass) and batched independently, then concatenated before the shared sharding / interleaving => that
    branch's batches are spread evenly over the whole epoch, and LR scheduling / epoch counting work as usual
    (HF derives the total number of steps from ``__len__``, which already includes the repeats).

    Optional **length bucketing** (``pool_factor>0`` with costs provided): global shuffle -> cut into pools of
    ``bs*pool_factor`` -> sort each pool by cost descending -> cut batches sequentially -> shuffle the batch
    order (the most expensive batch is moved to the front so an OOM surfaces at step one). Similar cost within
    a batch => far less text / time-series padding waste; randomness across pools plus random batch order
    preserves sampling randomness. ``pool_factor=0`` (default) reproduces the old per-batch behaviour.

    Optional **dynamic batch size under a token budget** (``*_token_budget>0`` with token_costs provided):
    instead of a fixed bs, greedily fill a batch until ``padded LLM tokens = bs x longest sample in the batch
    <= budget`` (LLM compute / memory scale with the padded rectangle, so the padded measure is the one that
    is linear in memory; short samples form large batches, long samples small ones, and per-batch compute is
    roughly constant). Guards: a secondary ``patch_budget`` caps the sum of Chronos patches (prevents "short
    text x long history" from making Chronos the bottleneck) and ``max_dynamic_bs`` caps the batch size; a
    single sample exceeding the budget gets a batch of its own. In dynamic mode ``bs`` is only the anchor for
    the pool size (pool = bs x pool_factor); there is no notion of an "incomplete batch" (every batch is filled
    to budget), no drop_last, and the number of steps floats with the packing -- ``__len__`` counts the real plan.

    Optional **bs ladder quantisation** (``bs_ladder``, dynamic mode only): a filled batch is trimmed to the
    largest ladder value <= its current size, and the trimmed samples carry over into the next batch (no
    sample is dropped; trimming takes a subset, so the token/patch budgets still hold). Motivation: the
    TileLang kernels of fla (the linear attention in Qwen3.5) bake the batch size in as a compile-time
    constant, and every new bs value triggers a JIT compile of tens of seconds on first sight -- quantisation
    reduces the distinct bs values from arbitrary integers to the ladder size. The ladder automatically
    includes 1 (the single-sample-over-budget case relies on it)."""

    def __init__(
        self,
        n_understanding: int,
        n_forecast: int,
        understanding_bs: int,
        forecast_bs: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
        understanding_costs=None,
        forecast_costs=None,
        pool_factor: int = 0,
        understanding_token_costs=None,
        forecast_token_costs=None,
        understanding_patch_costs=None,
        forecast_patch_costs=None,
        understanding_token_budget: int = 0,
        forecast_token_budget: int = 0,
        patch_budget: int = 0,
        max_dynamic_bs: int = 0,
        bs_ladder=None,
        patch_token_weight: float = 0.0,
        understanding_repeats: int = 1,
        forecast_repeats: int = 1,
    ):
        self.n_u = n_understanding
        self.n_f = n_forecast
        self.u_repeats = max(1, int(understanding_repeats))
        self.f_repeats = max(1, int(forecast_repeats))
        self.u_bs = understanding_bs
        self.f_bs = forecast_bs
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        for name, arr, n in (("understanding_costs", understanding_costs, n_understanding),
                             ("forecast_costs", forecast_costs, n_forecast),
                             ("understanding_token_costs", understanding_token_costs, n_understanding),
                             ("forecast_token_costs", forecast_token_costs, n_forecast),
                             ("understanding_patch_costs", understanding_patch_costs, n_understanding),
                             ("forecast_patch_costs", forecast_patch_costs, n_forecast)):
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name} must have length equal to the corresponding sample count {n}")
        if understanding_token_budget and understanding_token_costs is None:
            raise ValueError("understanding_token_budget>0 requires understanding_token_costs")
        if forecast_token_budget and forecast_token_costs is None:
            raise ValueError("forecast_token_budget>0 requires forecast_token_costs")
        self.u_costs = understanding_costs
        self.f_costs = forecast_costs
        self.pool_factor = max(0, int(pool_factor))
        self.u_tok = understanding_token_costs
        self.f_tok = forecast_token_costs
        self.u_patch = understanding_patch_costs
        self.f_patch = forecast_patch_costs
        self.u_token_budget = max(0, int(understanding_token_budget))
        self.f_token_budget = max(0, int(forecast_token_budget))
        self.patch_budget = max(0, int(patch_budget))
        self.max_dynamic_bs = max(0, int(max_dynamic_bs))
        self.bs_ladder = sorted({1, *(int(v) for v in bs_ladder)}) if bs_ladder else None
        # Patch conversion weight lambda: dynamic packing / sorting charge a unified memory cost
        # tok + lambda x patch against the token budget (patch-heavy and text-heavy batches are measured on
        # the same memory-peak scale). 0 = old behaviour (patches are only hard-capped by patch_budget).
        self.patch_token_weight = max(0.0, float(patch_token_weight))
        self._plan_cache = None  # (epoch, u_batches, f_batches): shared by __len__/__iter__, rebuilt per epoch
        self._epoch0_total = None  # total batch count of epoch 0: used to detect per-epoch drift of dynamic packing

    def set_epoch(self, epoch: int):
        if epoch != self.epoch:
            self._plan_cache = None
        self.epoch = epoch

    def _make_batches(self, indices, bs, g, sort_cost=None, tok_cost=None,
                      tok_budget: int = 0, patch_cost=None):
        """Shuffled indices -> list of batches.

        - Bucketing (pool_factor>0 and sort_cost given): sort each pool by cost descending so similar costs
          are adjacent.
        - Dynamic packing (tok_budget>0 and tok_cost given): greedily fill until the token sum <= budget
          (secondary budget / cap: see the class docstring); otherwise cut fixed-size batches (drop_last
          discards the tail).
        - With bucketing, shuffle the batch order and move the batch holding the most expensive sample to
          the front (an OOM surfaces at step one).
        """
        bucketing = self.pool_factor > 0 and sort_cost is not None
        if bucketing:
            pool = bs * self.pool_factor
            ordered = []
            for s in range(0, len(indices), pool):
                ordered.extend(sorted(indices[s : s + pool], key=sort_cost, reverse=True))
            indices = ordered
        dynamic = tok_budget > 0 and tok_cost is not None
        if dynamic:
            ladder = self.bs_ladder

            def _keep(n):  # ladder quantisation: largest ladder value <= n (the ladder contains 1, so a solution exists)
                return max(v for v in ladder if v <= n) if ladder else n

            batches, cur, ml, cp = [], [], 0, 0
            for i in indices:
                t = tok_cost(i)
                p = patch_cost(i) if patch_cost is not None else 0
                # The budget is charged on the padded cost: actual LLM compute / memory is proportional to
                # bs x longest-in-batch (short samples are padded to the longest), not to the sum of real
                # tokens -- "one long sample + many short ones" passes the sum test but explodes when padded.
                # Under bucketing the in-batch lengths are similar => padded ~ sum, so
                # the two measures transition smoothly.
                # `while` rather than `if`: after ladder trimming `cur` keeps the carried-over samples
                # (non-empty), and the new sample may still not fit, so another cut may be needed; each cut
                # removes at least one sample, so it terminates. Without quantisation the carry-over is empty
                # and at most one iteration runs.
                while cur and (
                    (len(cur) + 1) * max(ml, t) > tok_budget
                    or (self.patch_budget and cp + p > self.patch_budget)
                    or (self.max_dynamic_bs and len(cur) >= self.max_dynamic_bs)
                ):
                    keep = _keep(len(cur))
                    batches.append(cur[:keep])
                    cur = cur[keep:]
                    ml = max((tok_cost(j) for j in cur), default=0)
                    cp = sum(patch_cost(j) for j in cur) if patch_cost is not None else 0
                cur.append(i); ml = max(ml, t); cp += p
            while cur:  # the tail batch is quantised as well (split into ladder-sized batches); no "incomplete batch" to drop
                keep = _keep(len(cur))
                batches.append(cur[:keep])
                cur = cur[keep:]
        else:
            n_batches = len(indices) // bs if self.drop_last else math.ceil(len(indices) / bs)
            batches = [indices[i * bs : (i + 1) * bs] for i in range(n_batches)]
        if bucketing and len(batches) > 1:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[p] for p in perm]
            # Move the batch holding the globally most expensive sample to the front: the memory peak shows
            # up at step one instead of mid-training.
            probe = tok_cost if dynamic else sort_cost
            top = max(range(len(batches)), key=lambda b: max(probe(i) for i in batches[b]))
            batches[0], batches[top] = batches[top], batches[0]
        return batches

    def _shard_batches(self, batches):
        """Per rank, take the rank-th batch of every group of num_replicas batches so all ranks get the same count."""
        usable = (len(batches) // self.num_replicas) * self.num_replicas
        batches = batches[:usable]
        return batches[self.rank :: self.num_replicas]

    def _plan(self):
        """Build (or fetch the cached) per-branch batch plan for this epoch; shared by __len__ and __iter__."""
        if self._plan_cache is not None and self._plan_cache[0] == self.epoch:
            return self._plan_cache[1], self._plan_cache[2]
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # Each branch produces `repeats` passes of indices: an independent randperm per pass (drawn in sequence
        # from one generator => deterministic, different order per pass); each pass is batched independently
        # and then concatenated (guaranteeing "one pass = one full coverage of the data"; repeats never mix
        # across passes).
        u_passes = ([torch.randperm(self.n_u, generator=g).tolist() for _ in range(self.u_repeats)]
                    if self.n_u else [])
        f_passes = ([[i + self.n_u for i in torch.randperm(self.n_f, generator=g).tolist()]
                     for _ in range(self.f_repeats)] if self.n_f else [])  # offset into the concatenated index space

        w = self.patch_token_weight
        if w and self.u_tok is not None and self.u_patch is not None:
            u_tok = lambda i: self.u_tok[i] + w * self.u_patch[i]
        elif self.u_tok is not None:
            u_tok = lambda i: self.u_tok[i]
        else:
            u_tok = None
        if w and self.f_tok is not None and self.f_patch is not None:
            f_tok = lambda i: self.f_tok[i - self.n_u] + w * self.f_patch[i - self.n_u]
        elif self.f_tok is not None:
            f_tok = lambda i: self.f_tok[i - self.n_u]
        else:
            f_tok = None
        u_patch = (lambda i: self.u_patch[i]) if self.u_patch is not None else None
        f_patch = (lambda i: self.f_patch[i - self.n_u]) if self.f_patch is not None else None
        u_proxy = (lambda i: self.u_costs[i]) if self.u_costs is not None else None
        f_proxy = (lambda i: self.f_costs[i - self.n_u]) if self.f_costs is not None else None
        # In dynamic mode the sort key is the exact token count (tighter packing); otherwise the approximate cost.
        u_sort = u_tok if (self.u_token_budget and u_tok) else u_proxy
        f_sort = f_tok if (self.f_token_budget and f_tok) else f_proxy

        # Batch each pass, concatenate, then shard once: shard truncation loses at most (world-1) batches (same
        # order as for a single pass), and the interleaving spreads all passes of the branch evenly over the
        # epoch (pass order preserved: pass 1 first, then pass 2).
        u_batches = self._shard_batches([b for idx in u_passes for b in self._make_batches(
            idx, self.u_bs, g, sort_cost=u_sort, tok_cost=u_tok,
            tok_budget=self.u_token_budget, patch_cost=u_patch)]) if self.n_u else []
        f_batches = self._shard_batches([b for idx in f_passes for b in self._make_batches(
            idx, self.f_bs, g, sort_cost=f_sort, tok_cost=f_tok,
            tok_budget=self.f_token_budget, patch_cost=f_patch)]) if self.n_f else []
        if self.rank == 0:
            # A branch silently vanishing must be loud: n<bs (everything dropped by drop_last) or a global batch
            # count < num_replicas (_shard_batches truncates to 0) both make that branch yield no batch for the
            # whole epoch, and mixed training degrades to a single branch without warning.
            for name, n, batches in (("understanding", self.n_u, u_batches),
                                     ("forecast", self.n_f, f_batches)):
                if n and not batches:
                    warnings.warn(
                        f"{name} branch has {n} samples but produced 0 batches this epoch"
                        f" (sample count < bs dropped by drop_last, or global batch count < world_size={self.num_replicas}"
                        f" truncated by sharding) -- this branch will not be trained at all this epoch")
            # Dynamic packing drifts across epochs: the transformers 5.9 epoch loop iterates a fixed number of
            # steps taken from epoch 0 (it does not iterate to exhaustion), so extra batches are silently
            # skipped and missing ones cause idle, misaligned steps.
            total = len(u_batches) + len(f_batches)
            if self._epoch0_total is None:
                self._epoch0_total = total
            elif total != self._epoch0_total:
                warnings.warn(
                    f"epoch {self.epoch} packed {total} batches != epoch 0's {self._epoch0_total}"
                    f" (dynamic packing drifts across epochs): the HF Trainer loops a fixed number of steps taken from epoch 0, "
                    f"so the surplus batches will be silently dropped or the deficit filled with idle steps")
        self._plan_cache = (self.epoch, u_batches, f_batches)
        return u_batches, f_batches

    def __iter__(self) -> Iterator[list]:
        u_batches, f_batches = self._plan()
        # Interleave evenly by the actual batch counts (forecast share = n_f/(n_u+n_f); neither kind runs out early)
        order = self._interleave(len(u_batches), len(f_batches))
        ui = fi = 0
        for is_forecast in order:
            if is_forecast:
                yield f_batches[fi]; fi += 1
            else:
                yield u_batches[ui]; ui += 1

    @staticmethod
    def _interleave(n_u: int, n_f: int) -> list:
        total = n_u + n_f
        if total == 0:
            return []
        ratio = n_f / total
        order = []
        u_left, f_left = n_u, n_f
        # Bresenham-style allocation: forecast batches are spread evenly over the epoch at their actual share
        acc = 0.0
        for _ in range(total):
            acc += ratio
            take_f = (acc >= 1.0 and f_left > 0) or u_left == 0
            if take_f and f_left > 0:
                order.append(True); f_left -= 1; acc -= 1.0
            elif u_left > 0:
                order.append(False); u_left -= 1
            elif f_left > 0:
                order.append(True); f_left -= 1
        return order

    def __len__(self):
        # Count the real plan (under dynamic packing the batch count depends on the packing result; with a
        # fixed bs it matches the original closed-form formula).
        # Note: in dynamic mode the step count floats slightly across epochs; the HF scheduler estimates the
        # total from epoch 0's len, and the error is negligible.
        u_batches, f_batches = self._plan()
        return len(u_batches) + len(f_batches)
