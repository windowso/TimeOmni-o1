"""Equivalence of multi-GPU sharded inference -> merge (CPU, no process group): dist_info is monkeypatched so
run_*_infer simulates two ranks serially, rank1 -> rank0 (barrier is a no-op without an initialised process
group; rank 0 runs last, so all shards are on disk when it merges). The merged output must match the
single-process output row by row / value by value, and the .rank{r} intermediate shards must be cleaned up.

batch_size=1 guarantees that every sample has exactly the same batch composition in both runs => the values
should be bitwise identical (equivalence across batch compositions is covered separately by
test_generate_left_padding / test_encode_row_bucketing).
n=5 (odd) covers uneven sharding: rank 0 gets 3 rows, rank 1 gets 2.
"""
import json
import os
import tempfile

import numpy as np

import chronos_llm.eval.infer_forecast as inf_f
import chronos_llm.eval.infer_understanding as inf_u
from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.tests.test_eval_pipeline import _forecast_parquet, _understanding_jsonl
from chronos_llm.tests.test_pretrained_peft import _build_tiny_base


def _run_sharded(module, fn, world, *args, **kw):
    """Simulate the ranks serially from highest to lowest; returns the value of rank 0 (the last round)."""
    orig = module.dist_info
    try:
        for r in range(world - 1, -1, -1):
            module.dist_info = lambda r=r: (r, world)
            out = fn(*args, **kw)
    finally:
        module.dist_info = orig
    return out


def test_understanding_shard_merge():
    model, tok = _build_tiny_base(); model.eval()
    with tempfile.TemporaryDirectory() as d:
        jf = os.path.join(d, "toy.jsonl"); _understanding_jsonl(jf, n=5)
        kw = dict(batch_size=1, max_new_tokens=3, device="cpu", num_workers=0,
                  max_user_tokens=32, max_tokens=96)
        single_dir, shard_dir = os.path.join(d, "single"), os.path.join(d, "shard")
        inf_u.run_understanding_infer(model, tok, [jf], single_dir, **kw)
        _run_sharded(inf_u, inf_u.run_understanding_infer, 2, model, tok, [jf], shard_dir, **kw)
        a = [json.loads(x) for x in open(os.path.join(single_dir, "toy.jsonl"))]
        b = [json.loads(x) for x in open(os.path.join(shard_dir, "toy.jsonl"))]
        assert len(a) == 5 and a == b                       # row order restored + every field identical
        assert not [f for f in os.listdir(shard_dir) if ".rank" in f]  # shards cleaned up
    print("understanding shard merge == single process OK")


def test_forecast_shard_merge():
    model, tok = _build_tiny_base(); model.eval()
    with tempfile.TemporaryDirectory() as d:
        pq = os.path.join(d, "toy.parquet"); _forecast_parquet(pq, n=5)
        ds = ForecastParquetDataset(pq, tok, split="test", inference_mode=True,
                                    max_user_tokens=32, max_tokens=96)
        kw = dict(batch_size=1, max_new_tokens=3, device="cpu", num_workers=0)
        single, shard = os.path.join(d, "single.npz"), os.path.join(d, "shard.npz")
        inf_f.run_forecast_infer(model, tok, ds, single, **kw)
        out = _run_sharded(inf_f, inf_f.run_forecast_infer, 2, model, tok, ds, shard, **kw)
        assert out is not None and out["gt"].shape[0] == 5  # rank 0 returns the merged result
        za, zb = np.load(single, allow_pickle=True), np.load(shard, allow_pickle=True)
        for k in ("pred_quantiles", "gt", "roi_mask", "valid_mask", "quantile_levels"):
            np.testing.assert_allclose(za[k].astype(np.float32), zb[k].astype(np.float32),
                                       rtol=0, atol=0, equal_nan=True, err_msg=k)
        assert list(za["ids"]) == list(zb["ids"])           # row order restored to the original
        assert "gen_text" not in za.files and "gen_text" not in zb.files  # text does not go into the npz
        # Side-car jsonl: after the shard merge, row by row and field by field == single process (including the
        # gen_text; row order restored + _idx stripped)
        ja = [json.loads(x) for x in open(os.path.join(d, "single.jsonl"))]
        jb = [json.loads(x) for x in open(os.path.join(d, "shard.jsonl"))]
        assert len(ja) == 5 and ja == jb and all("_idx" not in r for r in jb)
        assert not [f for f in os.listdir(d) if ".rank" in f]
    print("forecast shard merge == single process OK")


if __name__ == "__main__":
    test_understanding_shard_merge()
    test_forecast_shard_merge()
    print("ALL OK")
