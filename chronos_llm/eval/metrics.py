"""Forecast metrics: MAE, MAPE, PCC, CRPS (normalized WQL). Pure numpy, no GPU/model needed,
unit-testable on CPU.

Conventions:
- The point forecast is the **median quantile q=0.5** (index 10 of chronos2's 21 quantiles).
- MAE / MAPE / PCC / CRPS are each computed **per sample over its valid positions, then averaged
  over samples**.
- MAE = raw scale, not normalized, y~0 positions not excluded -- the same protocol as the official
  MAE of external benchmarks (e.g. STReasoner/ST-Bench evaluate_qa.py), used for cross-benchmark
  tables; internally the normalized MAPE/CRPS below are used more often (MAE is not comparable
  across domains with very different scales and is only meaningful within one benchmark).
- MAPE excludes positions with |y|~0 (MAPE is undefined there; an eps denominator would produce
  1e8-scale artefacts that swamp the mean); reported in percent.
- CRPS = **normalized Weighted Quantile Loss**: per sample
  ``sum_{k,t} 2|(y-q_k)(1{y<=q_k}-alpha_k)| / (K * sum_t |y_t|)`` (K = number of quantiles), then
  averaged over samples. Consistent with chronos2's pinball formula (``2|(y-q)((y<=q)-alpha)|``).
"""
from typing import Dict

import numpy as np

EPS = 1e-8


def _median_index(quantile_levels: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.asarray(quantile_levels, dtype=float) - 0.5)))


def sample_mape(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    """Per-sample MAPE (%); only positions inside the mask with |y|>EPS count (MAPE is undefined at
    y~0; an eps denominator would produce 1e8-scale artefacts that nanmean cannot remove -- so those
    positions are excluded outright); returns nan when there is no valid position."""
    m = mask.astype(bool) & (np.abs(y) > EPS)
    if m.sum() == 0:
        return np.nan
    y, p = y[m], p[m]
    return float(np.mean(np.abs(y - p) / np.abs(y)) * 100.0)


def sample_mae(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    """Per-sample MAE (raw scale, not normalized, y~0 positions not excluded); mean(|y-p|) over the
    masked positions, nan when there is none. Same protocol as the official STReasoner
    evaluate_qa.py::evaluate_forecasting_predictions (per-sample mean absolute error over the K steps,
    plain arithmetic mean across samples, no weighting by point count/scenario)."""
    m = mask.astype(bool)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y[m] - p[m])))


def sample_pcc(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    """Per-sample Pearson correlation; needs >=2 valid positions and non-zero variance on both sides,
    otherwise returns nan."""
    m = mask.astype(bool)
    if m.sum() < 2:
        return np.nan
    y, p = y[m], p[m]
    if np.std(y) < EPS or np.std(p) < EPS:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def sample_wql(y: np.ndarray, q: np.ndarray, levels: np.ndarray, mask: np.ndarray) -> float:
    """Per-sample normalized WQL (CRPS proxy).

    y (T,), q (Q, T) per-quantile forecasts, levels (Q,) quantile levels, mask (T,). Returns nan when
    there is no valid position or the denominator is 0.
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return np.nan
    y_m = y[m]                              # (Tm,)
    denom_y = float(np.sum(np.abs(y_m)))
    if denom_y < EPS:
        return np.nan  # all-zero ground truth cannot be normalized: return nan to exclude it (an eps denominator would produce artefacts)
    q_m = q[:, m]                           # (Q, Tm)
    a = np.asarray(levels, dtype=float)[:, None]  # (Q, 1)
    # pinball: 2|(y - q)(1{y<=q} - alpha)|, per (quantile, position)
    pinball = 2.0 * np.abs((y_m[None, :] - q_m) * ((y_m[None, :] <= q_m).astype(float) - a))
    denom = q_m.shape[0] * denom_y   # K * sum|y|
    return float(np.sum(pinball) / denom)


def forecast_metrics(
    pred_quantiles: np.ndarray,   # (N, Q, T)
    gt: np.ndarray,               # (N, T)
    mask: np.ndarray,             # (N, T) bool, True = counted
    quantile_levels: np.ndarray,  # (Q,)
) -> Dict[str, float]:
    """Compute per-sample metrics for a batch and average them (ignoring samples with no valid
    position / that cannot be computed)."""
    levels = np.asarray(quantile_levels, dtype=float)
    qi = _median_index(levels)
    median = pred_quantiles[:, qi, :]                 # (N, T)
    maes, mapes, pccs, wqls = [], [], [], []
    for n in range(pred_quantiles.shape[0]):
        maes.append(sample_mae(gt[n], median[n], mask[n]))
        mapes.append(sample_mape(gt[n], median[n], mask[n]))
        pccs.append(sample_pcc(gt[n], median[n], mask[n]))
        wqls.append(sample_wql(gt[n], pred_quantiles[n], levels, mask[n]))
    return {
        "MAE": float(np.nanmean(maes)) if np.any(~np.isnan(maes)) else np.nan,
        "MAPE": float(np.nanmean(mapes)) if np.any(~np.isnan(mapes)) else np.nan,
        "PCC": float(np.nanmean(pccs)) if np.any(~np.isnan(pccs)) else np.nan,
        "CRPS": float(np.nanmean(wqls)) if np.any(~np.isnan(wqls)) else np.nan,
        "n_samples": int(pred_quantiles.shape[0]),
        "n_valid_pcc": int(np.sum(~np.isnan(pccs))),
    }
