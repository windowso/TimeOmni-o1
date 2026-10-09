"""CPU unit tests for uniform window sampling of over-long inputs + the global overview track.

When not triggered (T <= K*W) the output must be bitwise identical to the current behaviour; when triggered,
the K native-resolution windows must include the first and last positions and must not overlap, the overview
is a NaN-safe per-bucket nanmean, and chunk_pos gives (first patch position, stride) for every chunk.
"""
import json
import math
import os
import tempfile

import numpy as np
import torch

from transformers import AutoTokenizer

from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset, _sample_windows

LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")


def _mk_jsonl_npy(tmp, lengths_ch):
    """Build jsonl + npy: lengths_ch = [(T, C), ...]. The npy stores (T, C) (_load_ts_2d transposes to (C,T))."""
    rows = []
    for i, (T, C) in enumerate(lengths_ch):
        p = os.path.join(tmp, f"ts{i}.npy")
        np.save(p, np.random.RandomState(i).randn(T, C).astype(np.float32))
        rows.append({
            "id": str(i), "dataset_name": "unit", "task": "Classification",
            "input_ts": {"already_segment": False, "channel": C,
                         "original": {"ori_path": p, "ori_length": T}},
            "input_text": ["Classify the series above."],
            "gt_text": ["label_a"],
        })
    jp = os.path.join(tmp, "data.jsonl")
    with open(jp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return jp


def test_dataset_sampling_wiring():
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    with tempfile.TemporaryDirectory() as tmp:
        jp = _mk_jsonl_npy(tmp, [(200, 2), (50, 1)])   # sample 0 is over-long and triggers; sample 1 does not
        ds = UnderstandingJsonlDataset([jp], tok, sample_chunks=3, overview_chunk=True,
                                       sample_window=32)
        it0, it1 = ds[0], ds[1]
        assert it0["history"].shape == (2, 4 * 32) and it0["chunk_pos"].shape == (4, 2)
        assert it1["history"].shape == (1, 50) and "chunk_pos" not in it1
        # cost accounting == actual compact patch count (triggered 4*ceil(32/16)=8; not triggered ceil(50/16)=4)
        hp = ds.history_patches(max_context=240_000)
        assert hp[0] == (8, 2) and hp[1] == (4, 1), f"history_patches accounting mismatch: {hp}"
        ck = ds.cost_keys(max_context=240_000)
        assert ck[0] - len("Classify the series above.") // 4 == 2 * 8
        # K=0 disabled == old behaviour
        ds0 = UnderstandingJsonlDataset([jp], tok, sample_chunks=0)
        assert ds0[0]["history"].shape == (2, 200) and "chunk_pos" not in ds0[0]


def test_not_triggered_identity():
    arr = np.random.RandomState(0).randn(2, 64).astype(np.float32)
    out, pos = _sample_windows(arr, k=2, window=32, overview=True)   # T=64 == k*W, not triggered
    assert pos is None and out is arr, "T<=K*W must return the input unchanged (bitwise equal to the current behaviour)"
    out, pos = _sample_windows(arr, k=0, window=32, overview=True)   # k=0 disabled
    assert pos is None and out is arr


def test_triggered_windows_and_pos():
    T, W, K = 200, 32, 3
    arr = np.arange(2 * T, dtype=np.float32).reshape(2, T)
    out, pos = _sample_windows(arr, k=K, window=W, overview=False)
    assert out.shape == (2, K * W) and pos.shape == (K, 2)
    assert pos.dtype == np.int64, "chunk_pos must be int64 (floats get cast under DeepSpeed bf16)"
    starts = [round(i * (T - W) / (K - 1)) for i in range(K)]
    assert starts[0] == 0 and starts[-1] == T - W, "first and last windows must be included"
    for i, s in enumerate(starts):
        np.testing.assert_array_equal(out[:, i * W:(i + 1) * W], arr[:, s:s + W])
        assert pos[i, 0] == s and pos[i, 1] == W
    # non-overlapping: adjacent starts are >= W apart (guaranteed by T > K*W)
    assert all(b - a >= W for a, b in zip(starts, starts[1:]))


def test_overview_nanmean():
    T, W, K = 130, 16, 2   # T not divisible by W: mixed bucket sizes 9/8
    arr = np.ones((1, T), dtype=np.float32) * 5.0
    arr[0, :10] = np.nan                      # bucket containing NaN: nanmean should ignore it
    out, pos = _sample_windows(arr, k=K, window=W, overview=True)
    assert out.shape == (1, (K + 1) * W) and pos.shape == (K + 1, 2)
    ov = out[0, K * W:]
    bounds = (np.arange(W, dtype=np.int64) * T) // W
    for j in range(W):
        s, e = bounds[j], (bounds[j + 1] if j + 1 < W else T)
        seg = arr[0, s:e]
        ref = np.nan if np.isnan(seg).all() else np.nanmean(seg)
        if np.isnan(ref):
            assert np.isnan(ov[j])
        else:
            assert abs(ov[j] - ref) < 1e-5, f"bucket {j} overview != nanmean"
    assert pos[-1, 0] == 0 and pos[-1, 1] == T, "overview row = (0, T) (in raw-point units)"


from chronos_llm.models.qformer import SlidingWindowQFormer, _sinusoidal_pe, _sinusoidal_pe_at


def test_patch_pos_pe():
    # 1) pe_at(arange) == pe (the cornerstone of bitwise-unchanged behaviour for non-triggered samples without patch_pos)
    pe_ref = _sinusoidal_pe(40, 16, torch.device("cpu"), torch.float32)
    pe_at = _sinusoidal_pe_at(torch.arange(40, dtype=torch.float32), 16, torch.float32)
    assert torch.equal(pe_ref, pe_at)

    torch.manual_seed(0)
    qf = SlidingWindowQFormer(in_dim=16, out_dim=24, queries_per_window=2,
                              target_windows=4, min_windows=2, global_queries=3,
                              num_heads=4, num_layers=1)
    qf.eval()
    src = torch.randn(1, 2, 20, 16)
    with torch.no_grad():
        ref = qf(src)
        same = qf(src, patch_pos=torch.arange(20, dtype=torch.float32))   # == sequential indices
        shifted = qf(src, patch_pos=torch.arange(20, dtype=torch.float32) * 100 + 7.5)
    assert torch.equal(ref, same), "patch_pos=arange must be bitwise identical to passing nothing"
    assert not torch.allclose(ref, shifted), "a non-trivial patch_pos should change the global-branch output"
    assert torch.equal(ref[:, 3:], shifted[:, 3:]), "patch_pos may only affect the first G global tokens"
    # 2) length validation
    try:
        qf(src, patch_pos=torch.arange(19, dtype=torch.float32))
        assert False, "a patch_pos length mismatch should raise"
    except ValueError:
        pass


from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.models.chronos_llm_model import ChronosLLM
from chronos_llm.tests.test_pretrained_peft import _build_tiny_base


def test_patch_positions_expansion():
    """Pin the numerics of the glue that expands the int64 [start, span] payload into float32 per-patch positions."""
    cp = torch.tensor([[32, 32], [0, 112]], dtype=torch.int64)   # detail window s=32,W=32; overview T=112
    pos = ChronosLLM._patch_positions(cp, [2, 7], window=32, device=torch.device("cpu"))
    exp = [2.0, 3.0] + [j * 3.5 for j in range(7)]               # pos0=32/16=2, dpos=1; overview dpos=112/32=3.5
    assert torch.allclose(pos, torch.tensor(exp)), pos
    assert ChronosLLM._patch_positions(None, [2], window=32, device=torch.device("cpu")) is None


def test_model_chunk_pos_e2e():
    """Full chain dataset sampling output -> collator -> model encode: positions expand correctly, gradients
    are connected, omitting chunk_pos is bitwise identical to the current behaviour, and a row-count mismatch
    fails loudly."""
    model, tok = _build_tiny_base()
    model.eval()
    model.chronos_window = 32        # matches the dataset's sample_window

    with tempfile.TemporaryDirectory() as tmp:
        jp = _mk_jsonl_npy(tmp, [(300, 2), (40, 1)])
        ds = UnderstandingJsonlDataset([jp], tok, sample_chunks=3, overview_chunk=True,
                                       sample_window=32)
        coll = ChronosLLMCollator(tokenizer=tok)
        batch = coll([ds[0], ds[1]])
        assert "chunk_pos" in batch and batch["chunk_pos"][0].shape == (4, 2)
        assert batch["chunk_pos"][1] is None, "the chunk_pos slot of a non-triggered sample should be None"

        with torch.no_grad():
            soft = model._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch["n_channels"],
                batch["chunk_pos"])
        assert len(soft) == 2 and all(torch.isfinite(s).all() for s in soft)

        # omitting chunk_pos == current behaviour (same batch data, 4th argument omitted)
        with torch.no_grad():
            ref = model._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch["n_channels"])
            sans = model._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch["n_channels"], None)
        for a, b in zip(ref, sans):
            assert torch.equal(a, b)

        # pos only affects the global tokens: everything beyond the first G (local-window branch) must be
        # bitwise identical to the no-pos run; the triggered sample (i=0) should have its global tokens
        # changed by the real positions.
        G = model.config.sw_global_queries
        for a, b in zip(soft, ref):
            assert torch.equal(a[G:], b[G:]), "pos may only affect the first G global tokens"
        assert not torch.equal(soft[0][:G], ref[0][:G]), "the triggered sample's global tokens should be affected by pos"

        # a mismatch between chunk_pos rows and the actual chunk count (sample_window != chronos_window) must raise
        model.chronos_window = 16    # doubles the number of chunks -> disagrees with the 4 chunk_pos rows
        try:
            model._encode_history_to_soft_prompt(
                batch["context"], batch["true_lengths"], batch["n_channels"],
                batch["chunk_pos"])
            assert False, "the mismatch should raise ValueError"
        except ValueError:
            pass
        model.chronos_window = 32

    # gradient connectivity in training mode (C=1 compatibility case)
    model.train()
    arr = np.random.RandomState(7).randn(1, 300).astype(np.float32)
    compact, cp = _sample_windows(arr, k=3, window=32, overview=True)
    ctx = torch.from_numpy(compact)
    soft = model._encode_history_to_soft_prompt(
        ctx, torch.tensor([ctx.shape[1]]), torch.tensor([1]),
        [torch.from_numpy(cp)])
    sum(s.float().pow(2).mean() for s in soft).backward()
    g = [p.grad for p in model.history_qformer.global_qformer.parameters()
         if p.grad is not None]
    assert g and any(x.abs().sum() > 0 for x in g), "global-branch gradient not connected"


if __name__ == "__main__":
    test_not_triggered_identity()
    test_triggered_windows_and_pos()
    test_overview_nanmean()
    test_dataset_sampling_wiring()
    test_patch_pos_pe()
    test_patch_positions_expansion()
    test_model_chunk_pos_e2e()
    print("ALL OK")
