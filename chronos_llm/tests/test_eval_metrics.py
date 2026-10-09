"""CPU unit tests for the forecast metrics: hand-computed WQL check + perfect-prediction boundary + mask effect (pure numpy, no GPU)."""
import numpy as np

from chronos_llm.eval.metrics import forecast_metrics, sample_wql, sample_mae, sample_mape, sample_pcc

CHRONOS_LEVELS = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])


def test_sample_wql_hand():
    # gt=2, 3 quantiles [0.1,0.5,0.9] predicting [1,2,3]: pinball=0.2+0+0.2=0.4, denom=3*2=6 -> 0.0666...
    y = np.array([2.0]); q = np.array([[1.0], [2.0], [3.0]]); lv = np.array([0.1, 0.5, 0.9])
    w = sample_wql(y, q, lv, np.array([True]))
    assert abs(w - 0.4 / 6.0) < 1e-6, w
    print(f"sample_wql hand-computed check OK ({w:.6f})")


def test_perfect_prediction():
    # all quantiles = gt -> pinball all 0 -> CRPS=0; median=gt -> MAE=0, MAPE=0, PCC=1.
    gt = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pred = np.broadcast_to(gt[:, None, :], (1, 21, 5)).copy()
    mask = np.ones_like(gt, bool)
    m = forecast_metrics(pred, gt, mask, CHRONOS_LEVELS)
    assert abs(m["MAE"]) < 1e-8, m["MAE"]
    assert abs(m["MAPE"]) < 1e-4, m["MAPE"]
    assert abs(m["PCC"] - 1.0) < 1e-6, m["PCC"]
    assert abs(m["CRPS"]) < 1e-8, m["CRPS"]
    print(f"perfect-prediction boundary OK MAE={m['MAE']:.2e} MAPE={m['MAPE']:.2e} PCC={m['PCC']:.4f} CRPS={m['CRPS']:.2e}")


def test_sample_mae_hand():
    # y-p = [1,-2,3] -> MAE = mean(|1|,|-2|,|3|) = 2; matches the official STReasoner protocol
    # (raw units, no exclusion of y≈0, no normalisation). Positions with y=0 are counted as usual
    # (unlike the exclusion behaviour of MAPE).
    y = np.array([0.0, 10.0, 20.0]); p = np.array([1.0, 8.0, 23.0]); m = np.ones(3, bool)
    mae = sample_mae(y, p, m)
    assert abs(mae - 2.0) < 1e-9, mae
    assert np.isnan(sample_mae(y, p, np.zeros(3, bool)))  # no valid positions -> nan
    print(f"sample_mae hand-computed check OK ({mae:.4f})")


def test_mape_constant_offset():
    # median = gt + 1 -> MAPE = mean(1/(|gt|+eps))*100; PCC is still 1.
    gt = np.array([[10.0, 20.0, 40.0, 50.0]])
    pred = np.broadcast_to((gt + 1.0)[:, None, :], (1, 21, 4)).copy()
    mask = np.ones_like(gt, bool)
    expect = float(np.mean(1.0 / np.abs(gt[0])) * 100.0)
    assert abs(sample_mape(gt[0], pred[0, 10], mask[0]) - expect) < 1e-3
    assert abs(sample_pcc(gt[0], pred[0, 10], mask[0]) - 1.0) < 1e-6
    print(f"constant-offset MAPE check OK (~{expect:.3f}%)")


def test_mask_and_roi():
    # full vs subset mask must differ; samples without valid positions are ignored.
    gt = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    pred = np.broadcast_to(gt[:, None, :], (2, 21, 4)).copy()
    pred[0, :, 2:] += 5.0          # sample 0: second half of the forecast is off
    full = np.ones_like(gt, bool)
    roi = np.zeros_like(gt, bool); roi[:, :2] = True   # only the first half (sample 0 is accurate there)
    mf = forecast_metrics(pred, gt, full, CHRONOS_LEVELS)
    mr = forecast_metrics(pred, gt, roi, CHRONOS_LEVELS)
    assert mr["MAPE"] < mf["MAPE"], (mr["MAPE"], mf["MAPE"])   # ROI (first half) is more accurate
    # a sample whose mask is all False must be skipped
    empty = np.zeros_like(gt, bool); empty[0] = True
    me = forecast_metrics(pred, gt, empty, CHRONOS_LEVELS)
    assert me["n_valid_pcc"] == 1, me["n_valid_pcc"]
    print(f"mask/ROI effect OK (full MAPE={mf['MAPE']:.3f} > roi MAPE={mr['MAPE']:.3f})")


def test_zero_target_handling():
    # y=0 positions are excluded from MAPE (an eps denominator would produce 1e8-scale bogus values
    # that swamp the mean); WQL of an all-zero ground truth returns nan.
    y = np.array([0.0, 10.0]); p = np.array([1.0, 11.0]); m = np.ones(2, bool)
    assert abs(sample_mape(y, p, m) - 10.0) < 1e-6, sample_mape(y, p, m)  # only the 10% of 10->11 counts
    assert np.isnan(sample_mape(np.zeros(2), p, m))                        # all zero -> no countable positions
    q = np.ones((3, 2)); lv = np.array([0.1, 0.5, 0.9])
    assert np.isnan(sample_wql(np.zeros(2), q, lv, m))                     # denominator Σ|y|=0 -> nan
    print("y=0 exclusion / all-zero WQL=nan OK")


if __name__ == "__main__":
    test_sample_wql_hand()
    test_perfect_prediction()
    test_sample_mae_hand()
    test_mape_constant_offset()
    test_mask_and_roi()
    test_zero_target_handling()
    print("ALL OK")
