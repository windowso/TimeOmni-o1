"""CPU unit tests of the multi-channel (multivariate) forecasting branch: covariate inputs + multiple target outputs + cross_states broadcasting."""
import torch

from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.models.chronos_llm_model import add_lora
from chronos_llm.tests.test_pretrained_peft import _build_tiny_base


def _forecast_item(model, tok, C1, n_targets, n_fut, T=40, fl=8):
    pre = [model.ts_start_id, model.ts_end_id] + tok("predict.", add_special_tokens=False)["input_ids"]
    ans = tok(" up.", add_special_tokens=False)["input_ids"]
    item = {
        "branch": "forecast",
        "history": torch.randn(C1, T),
        "future": torch.randn(n_targets, fl),
        "roi": torch.zeros(fl),
        "input_ids": pre + ans,
        "labels": [-100] * len(pre) + ans,
    }
    if n_fut > 0:
        item["future_covariates"] = torch.randn(n_fut, fl)
    return item


def test_forecast_losses_target_slicing():
    """forecast_losses computes the loss only on target rows and outputs (sum n_targets, Q, H)."""
    model, _ = _build_tiny_base()
    SC, L = 3, 64
    ctx = torch.randn(SC, L)
    target_idx = torch.tensor([True, True, False])     # first 2 rows targets, 3rd row a covariate
    future = torch.randn(2, 16)
    res = model.chronos.forecast_losses(
        context=ctx, future_target=future, num_output_patches=1,
        group_ids=torch.zeros(SC, dtype=torch.long), target_idx=target_idx,
        roi_mask=torch.zeros(2, 16))
    assert res["quantile_preds"].shape[0] == 2, res["quantile_preds"].shape   # target rows only
    assert torch.isfinite(res["pred_loss"]).all() and torch.isfinite(res["roi_loss"]).all()
    print("forecast_losses target slicing + output shape OK")


def test_forward_forecast_multichannel():
    """End to end: multi-channel forecast (sample 0 C1=3/n_t=2/n_fut=1, sample 1 C1=2/n_t=1/n_fut=1) -> finite losses."""
    torch.manual_seed(0)
    model, tok = _build_tiny_base()
    base = add_lora(model, r=4, alpha=8, dropout=0.0).get_base_model()
    col = ChronosLLMCollator(tok, forecast_max_context=200)
    batch = col([
        _forecast_item(base, tok, C1=3, n_targets=2, n_fut=1),
        _forecast_item(base, tok, C1=2, n_targets=1, n_fut=1),
    ])
    assert batch["context"].shape[0] == 5 and batch["group_ids"].tolist() == [0, 0, 0, 1, 1]
    assert batch["future"].shape[0] == 3 and batch["n_targets"].tolist() == [2, 1]
    assert batch["future_covariates"].shape[0] == 2
    for blk in base.chronos.encoder.block:   # open the gate to verify the feedback path is connected
        blk.cross_attn.gate.data.fill_(0.3)
    out = base.forward_forecast(batch)
    for k in ("text_loss", "pred_loss", "roi_loss", "loss"):
        assert torch.isfinite(out[k]).all(), k
    print("multi-channel forward_forecast losses finite OK")


def test_forecast_single_channel_compat():
    """Single-channel forecast (1D history/future, no covariates) -> same behaviour as before, finite losses."""
    torch.manual_seed(0)
    model, tok = _build_tiny_base()
    base = add_lora(model, r=4, alpha=8, dropout=0.0).get_base_model()
    col = ChronosLLMCollator(tok, forecast_max_context=200)
    it = _forecast_item(base, tok, C1=1, n_targets=1, n_fut=0)
    it["history"] = it["history"][0]   # 1D
    it["future"] = it["future"][0]     # 1D
    batch = col([it])
    assert batch["context"].shape[0] == 1 and "future_covariates" not in batch
    out = base.forward_forecast(batch)
    assert all(torch.isfinite(out[k]).all() for k in ("text_loss", "pred_loss", "roi_loss", "loss"))
    print("single-channel forecast compatibility OK")


if __name__ == "__main__":
    test_forecast_losses_target_slicing()
    test_forward_forecast_multichannel()
    test_forecast_single_channel_compat()
    print("ALL OK")
