"""CPU unit test for multi-channel dataset parsing: understanding _load_ts_2d + multi-channel forecast parquet (including single-channel compatibility)."""
import os
import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.understanding_dataset import _load_ts_2d
from chronos_llm.models.chronos_llm_model import _add_ts_special_tokens
from chronos_llm.tests.test_pretrained_peft import LLM


def test_load_ts_2d():
    with tempfile.TemporaryDirectory() as d:
        # 2D (T=20, C=3) -> (3, 20)
        p2 = os.path.join(d, "a.npy"); np.save(p2, np.random.randn(20, 3).astype(np.float32))
        a = _load_ts_2d(p2)
        assert a.shape == (3, 20), a.shape
        # 1D (20,) -> (1, 20)
        p1 = os.path.join(d, "b.npy"); np.save(p1, np.random.randn(20).astype(np.float32))
        b = _load_ts_2d(p1)
        assert b.shape == (1, 20), b.shape
        # bad path -> (1,16) placeholder
        c = _load_ts_2d(os.path.join(d, "nope.npy"))
        assert c.shape == (1, 16)
    print("_load_ts_2d (C,T) + single channel + placeholder OK")


def _row(history, future, future_cov=None):
    # stored as nested lists (pyarrow cannot put a 2D ndarray straight into an object column; real multi-channel parquet also stores list<list>).
    r = {
        "history_values": history.tolist(), "future_values": future.tolist(),
        "background": "bg", "event": "ev", "prompt": "predict",
        "reasoning": "because trend", "conclusion": "up",
        "past_len": int(history.shape[-1]),
        "roi_start_idx": 0, "roi_end_idx": 2,
    }
    if future_cov is not None:
        r["future_covariates"] = future_cov.tolist()
    return r


def test_forecast_parquet_multichannel():
    """Multi-channel and single-channel data are stored in two separate parquet files (a column needs a uniform type: multi-channel list<list>, single-channel list<float>)."""
    import pandas as pd
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    with tempfile.TemporaryDirectory() as d:
        # multi-channel parquet: C1=3 (2 targets + 1 known-future covariate), future (2,5), future_cov (1,5)
        mc_path = os.path.join(d, "mc.parquet")
        pd.DataFrame([_row(
            np.random.randn(3, 10).astype(np.float32),
            np.random.randn(2, 5).astype(np.float32),
            np.random.randn(1, 5).astype(np.float32),
        )]).to_parquet(mc_path)
        mc = ForecastParquetDataset(mc_path, tok, split=None, max_user_tokens=64, max_tokens=128)[0]
        assert tuple(mc["history"].shape) == (3, 10), mc["history"].shape
        assert tuple(mc["future"].shape) == (2, 5), mc["future"].shape
        assert tuple(mc["future_covariates"].shape) == (1, 5)

        # single-channel parquet: 1D history/future (the current format), no covariate column
        sc_path = os.path.join(d, "sc.parquet")
        pd.DataFrame([_row(
            np.random.randn(8).astype(np.float32), np.random.randn(4).astype(np.float32),
        )]).to_parquet(sc_path)
        sc = ForecastParquetDataset(sc_path, tok, split=None, max_user_tokens=64, max_tokens=128)[0]
        assert sc["history"].dim() == 1 and sc["history"].shape[0] == 8
        assert sc["future"].dim() == 1 and sc["future"].shape[0] == 4
        assert "future_covariates" not in sc
    print("forecast parquet multi-channel parsing + single-channel compatibility OK")


if __name__ == "__main__":
    test_load_ts_2d()
    test_forecast_parquet_multichannel()
    print("ALL OK")
