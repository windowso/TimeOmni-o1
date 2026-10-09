"""Data hygiene + per-epoch reshuffle CPU unit tests.

Covers the following fixes:
1. ``_truncate_to_tokens``: over-long user text is truncated in the **middle**, keeping both ends
   (the question is usually at the tail and must not be cut).
2. ``build_supervised_ids``: empty answer / over-long rendering -> returns None (instead of labels
   all -100 -> CE=NaN, or silently truncating the conclusion).
3. Both datasets: empty-answer rows are filtered at load time; over-long samples are lazily skipped
   in __getitem__ and replaced by the next sample (batch shape unchanged).
4. ``_load_ts_2d``: drops all-NaN channels (produced by coercing CSV text/timestamp columns; they
   would pollute the loss with NaN).
5. ``_EpochAwareDataLoader.set_epoch`` -> DualBranchBatchSampler reshuffles every epoch
   (a bare DataLoader has no set_epoch, the transformers Trainer never calls it -> same order every epoch).
6. The collator attention_mask is built from the true lengths (a pad token inside the text does not punch a hole).
"""
import json
import os
import tempfile
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer

from chronos_llm.data.chat_utils import _truncate_to_tokens, build_supervised_ids
from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import DualBranchBatchSampler
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset, _load_ts_2d
from chronos_llm.models.chronos_llm_model import _add_ts_special_tokens
from chronos_llm.tests.test_pretrained_peft import LLM


def _tok():
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    return tok


def test_truncate_middle(tok):
    head_word, tail_word = "HEADMARKER", "TAILQUESTION"
    text = head_word + " " + "filler word " * 600 + " " + tail_word
    out = _truncate_to_tokens(tok, text, 64)
    assert head_word in out, "truncation should keep the head (background/instructions)"
    assert tail_word in out, "truncation should keep the tail (question/options)"
    assert "(truncated)" in out
    n = len(tok(out, add_special_tokens=False)["input_ids"])
    assert n <= 64 + 16, f"length after truncation should be ~max_tokens, got {n}"
    # Returned unchanged when not over-long
    assert _truncate_to_tokens(tok, "short text", 64) == "short text"
    print("_truncate_to_tokens middle truncation keeps both ends OK")


def test_supervised_ids_none(tok):
    # Empty answer: LCP == full length -> None (labels all -100 would give CE=NaN)
    assert build_supervised_ids(tok, "q?", "", reasoning_content=None,
                                max_user_tokens=64, max_tokens=256) is None
    # Over-long rendering -> None (truncation would silently cut the conclusion)
    assert build_supervised_ids(tok, "q?", "word " * 400, reasoning_content=None,
                                max_user_tokens=64, max_tokens=128) is None
    # A normal sample still returns (ids, labels) with supervised tokens
    ids, labels = build_supervised_ids(tok, "q?", "answer.", reasoning_content=None,
                                       max_user_tokens=64, max_tokens=256)
    assert len(ids) == len(labels) and any(l != -100 for l in labels)
    print("build_supervised_ids returns None for empty answer / over-long OK")


def test_understanding_filter_and_skip(tok):
    with tempfile.TemporaryDirectory() as d:
        ts = os.path.join(d, "ts.npy")
        np.save(ts, np.random.randn(32).astype(np.float32))
        rows = [
            {"input_text": ["valid?"], "gt_text": ["yes"],
             "input_ts": {"original": {"ori_path": ts}}},
            {"input_text": ["empty answer"], "gt_text": [""],          # empty answer -> filtered in init
             "input_ts": {"original": {"ori_path": ts}}},
            {"input_text": ["overlong"], "gt_text": ["word " * 400],   # over-long rendering -> lazily skipped
             "input_ts": {"original": {"ori_path": ts}}},
        ]
        p = os.path.join(d, "u.jsonl")
        with open(p, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = UnderstandingJsonlDataset([p], tok, max_user_tokens=64, max_tokens=256)
            assert len(ds) == 2, f"empty answers should be filtered at load time, len={len(ds)}"
            it1 = ds[1]  # over-long sample -> replaced by sample 0
        assert any(l != -100 for l in it1["labels"])
        assert 1 in ds._skip
        # Inference mode: empty-answer samples are kept (gt is left to eval)
        ds_inf = UnderstandingJsonlDataset([p], tok, max_user_tokens=64, max_tokens=256,
                                           inference_mode=True)
        assert len(ds_inf) == 3
    print("understanding empty-answer filter + over-long lazy skip OK")


def test_forecast_filter_and_skip(tok):
    import pandas as pd
    h, fu = np.random.randn(16).astype(np.float32), np.random.randn(4).astype(np.float32)

    def row(conclusion, reasoning="because trend"):
        return {"history_values": h.tolist(), "future_values": fu.tolist(),
                "background": "bg", "event": "ev", "prompt": "predict",
                "reasoning": reasoning, "conclusion": conclusion,
                "past_len": 16, "roi_start_idx": 16, "roi_end_idx": 18}

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.parquet")
        pd.DataFrame([row("up"), row("  "), row("up", reasoning="word " * 400)]).to_parquet(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = ForecastParquetDataset(p, tok, split=None, max_user_tokens=64, max_tokens=256)
            assert len(ds) == 2, f"empty conclusions should be filtered at load time, len={len(ds)}"
            it1 = ds[1]  # over-long reasoning -> replaced by the next sample
        assert any(l != -100 for l in it1["labels"])
        assert 1 in ds._skip
    print("forecast empty-conclusion filter + over-long lazy skip OK")


def test_load_ts_2d_nan_channel():
    with tempfile.TemporaryDirectory() as d:
        a = np.random.randn(20, 3).astype(np.float32)
        a[:, 1] = np.nan  # column 1 (channel 1 after transpose) all NaN, e.g. a coerced timestamp column
        p = os.path.join(d, "a.npy")
        np.save(p, a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _load_ts_2d(p)
        assert out.shape == (2, 20), f"all-NaN channel should be dropped, got {out.shape}"
        assert not np.isnan(out).all(axis=1).any()
        # All channels NaN -> placeholder
        np.save(p, np.full((20, 2), np.nan, dtype=np.float32))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _load_ts_2d(p).shape == (1, 16)
    print("_load_ts_2d drops all-NaN channels OK")


def test_load_ts_2d_wav():
    """torchaudio 2.10's .load() decodes through the TorchCodec backend and needs torchcodec==0.10.0
    (see the _load_ts_2d comment: 0.11+ is ABI-incompatible with torch 2.10, 0.9- does not support
    system FFmpeg 4). If it is missing/mismatched, every .wav silently degrades to the (1,16) all-zero
    placeholder (corrupting the series of understanding samples such as Powdermill bird calls). This
    test guards normal .wav decoding: PCM int16 normalised to [-1,1] (/32768), mono (T,)->(1,T),
    stereo (T,2)->(2,T). The fixture is written with scipy to decouple it from the reading library."""
    from scipy.io import wavfile

    with tempfile.TemporaryDirectory() as d:
        # mono int16
        n = 100
        mono = (np.sin(np.linspace(0, 12, n)) * 10000).astype(np.int16)
        pm = os.path.join(d, "mono.wav")
        wavfile.write(pm, 32000, mono)
        out = _load_ts_2d(pm)
        assert out.shape == (1, n), f".wav should not degrade to the placeholder, got {out.shape}"
        np.testing.assert_allclose(out[0], mono.astype(np.float32) / 32768.0,
                                   rtol=0, atol=1e-4)
        # stereo int16 -> (2, T)
        stereo = np.stack([mono, (mono // 2)], axis=1)  # (T, 2)
        ps = os.path.join(d, "stereo.wav")
        wavfile.write(ps, 32000, stereo)
        out2 = _load_ts_2d(ps)
        assert out2.shape == (2, n), f"stereo should be (2,T), got {out2.shape}"
    print("_load_ts_2d .wav loading via torchaudio+torchcodec OK")


def test_ts_len_ch_robust():
    """_ts_len_ch is robust to dirty metadata: when seg_length is a placeholder string (e.g.
    DEBUG_SEG_LENGTH) it falls back to ori_length instead of int() raising ValueError --
    cost_keys/history_patches (length bucketing and dynamic batching) both obtain the history
    length through it, and a crash would prevent training from starting at all."""
    from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset

    f = UnderstandingJsonlDataset._ts_len_ch
    # Real Release_train_standard shape: seg is a DEBUG placeholder, ori_length is real
    s = {"input_ts": {"channel": 3, "segment": {"seg_length": "DEBUG_SEG_LENGTH"},
                      "original": {"ori_length": 6000}}}
    assert f(s) == (6000, 3)
    # seg_length is a real number (already segmented) -> used directly
    assert f({"input_ts": {"channel": 1, "segment": {"seg_length": 36865}, "original": {}}}) == (36865, 1)
    # input_ts not a dict / missing / all placeholders -> (0, 1) fallback
    assert f({"input_ts": "not a dict"}) == (0, 1)
    assert f({}) == (0, 1)
    assert f({"input_ts": {"channel": "DEBUG", "segment": {"seg_length": None},
                           "original": {"ori_length": "x"}}}) == (0, 1)
    print("_ts_len_ch robust to dirty metadata OK")


def test_epoch_reshuffle():
    from chronos_llm.trainer import _EpochAwareDataLoader

    class _Dummy(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            return i

    bs = DualBranchBatchSampler(n_understanding=64, n_forecast=0,
                                understanding_bs=4, forecast_bs=2, seed=0)
    dl = _EpochAwareDataLoader(_Dummy(), batch_sampler=bs, collate_fn=lambda x: x)
    assert hasattr(dl, "set_epoch"), "the Trainer only calls set_epoch when the dataloader itself has it"
    dl.set_epoch(0); order0 = [b for b in dl]
    dl.set_epoch(0); order0b = [b for b in dl]
    dl.set_epoch(1); order1 = [b for b in dl]
    assert order0 == order0b, "the same epoch should be reproducible"
    assert order0 != order1, "different epochs should reshuffle (a bare DataLoader gives the same order every epoch)"
    assert sorted(sum(order1, [])) == list(range(64)), "reshuffling must not lose samples"
    print("set_epoch per-epoch reshuffle OK")


def test_collator_attention_by_length(tok):
    pad_id = tok.pad_token_id
    coll = ChronosLLMCollator(tok)
    batch = [
        {"branch": "understanding", "history": torch.randn(8),
         "input_ids": [5, pad_id, 7], "labels": [-100, -100, 7]},  # pad token appearing inside the text
        {"branch": "understanding", "history": torch.randn(8),
         "input_ids": [5], "labels": [-100]},
    ]
    out = coll(batch)
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]], (
        "mask should be built from the true lengths: a pad token inside the text does not punch a hole, padding is all 0"
    )
    print("collator attention_mask built from lengths OK")


if __name__ == "__main__":
    tok = _tok()
    test_truncate_middle(tok)
    test_supervised_ids_none(tok)
    test_understanding_filter_and_skip(tok)
    test_forecast_filter_and_skip(tok)
    test_load_ts_2d_nan_channel()
    test_load_ts_2d_wav()
    test_ts_len_ch_robust()
    test_epoch_reshuffle()
    test_collator_attention_by_length(tok)
    print("ALL OK")
