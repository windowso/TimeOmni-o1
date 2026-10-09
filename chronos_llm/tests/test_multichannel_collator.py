"""CPU unit tests for the collator's multi-channel folding (no model; pure tensors + a stub tokenizer)."""
import torch

from chronos_llm.data.collator import ChronosLLMCollator


class _Tok:
    pad_token_id = 0


def test_fold_channels_and_group_ids():
    col = ChronosLLMCollator(_Tok())
    batch = [
        {"branch": "understanding", "history": torch.arange(8.0).reshape(2, 4),
         "input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
        {"branch": "understanding", "history": torch.arange(3.0).reshape(1, 3),
         "input_ids": [1, 2], "labels": [-100, 2]},
    ]
    out = col(batch)
    assert out["context"].shape[0] == 3, out["context"].shape          # ΣC = 2+1
    assert out["group_ids"].tolist() == [0, 0, 1]                       # same sample -> same group
    assert out["true_lengths"].tolist() == [4, 4, 3]                    # valid length per channel row
    assert out["n_channels"].tolist() == [2, 1]
    # right-aligned: the most recent values sit at the tail (sample 0, channel 0 = [0,1,2,3])
    assert torch.equal(out["context"][0, -4:], torch.tensor([0.0, 1, 2, 3]))
    print("multi-channel folding + group_ids OK")


def test_single_channel_compat():
    """A 1-D history (the default) is treated as (1,T): after folding (B,L), group_ids=[0,1], n_channels=[1,1], equivalent to the legacy logic."""
    col = ChronosLLMCollator(_Tok())
    batch = [
        {"branch": "understanding", "history": torch.arange(5.0),
         "input_ids": [1, 2], "labels": [-100, 2]},
        {"branch": "understanding", "history": torch.arange(3.0),
         "input_ids": [1], "labels": [1]},
    ]
    out = col(batch)
    assert out["context"].shape == (2, 5)
    assert out["group_ids"].tolist() == [0, 1]
    assert out["true_lengths"].tolist() == [5, 3]
    assert out["n_channels"].tolist() == [1, 1]
    print("single-channel compatibility OK")


def test_forecast_fold_targets_and_covariates():
    col = ChronosLLMCollator(_Tok(), forecast_max_context=100)
    batch = [{
        "branch": "forecast",
        "history": torch.randn(3, 10),               # C1=3 (2 targets + 1 covariate)
        "future": torch.randn(2, 5),                 # n_targets=2
        "future_covariates": torch.randn(1, 5),      # n_fut=1 (known-future)
        "roi": torch.zeros(5),
        "input_ids": [1, 2, 3], "labels": [-100, -100, 3],
    }]
    out = col(batch)
    assert out["context"].shape[0] == 3 and out["group_ids"].tolist() == [0, 0, 0]
    assert out["future"].shape == (2, 5)                 # Σn_targets
    assert out["n_targets"].tolist() == [2]
    assert out["future_covariates"].shape == (1, 5)      # Σn_fut
    assert out["n_future_covariates"].tolist() == [1]
    assert out["roi_mask"].shape == (2, 5)               # one row per target (the shared roi is broadcast)
    print("forecast multi-channel folding (targets + covariates + roi broadcast) OK")


def test_forecast_single_channel_compat():
    """Forecast with a 1-D future (the default): n_targets=1, no covariates -> legacy behaviour."""
    col = ChronosLLMCollator(_Tok(), forecast_max_context=100)
    batch = [{
        "branch": "forecast", "history": torch.randn(8), "future": torch.randn(4),
        "roi": torch.zeros(4), "input_ids": [1, 2], "labels": [-100, 2],
    }]
    out = col(batch)
    assert out["context"].shape == (1, 8)
    assert out["future"].shape == (1, 4) and out["n_targets"].tolist() == [1]
    assert "future_covariates" not in out      # the key is absent when there are no covariates
    assert out["roi_mask"].shape == (1, 4)
    print("forecast single-channel compatibility OK")


if __name__ == "__main__":
    test_fold_channels_and_group_ids()
    test_single_channel_compat()
    test_forecast_fold_targets_and_covariates()
    test_forecast_single_channel_compat()
    print("ALL OK")
