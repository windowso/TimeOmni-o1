"""Map a forecast npz with nine quantile levels (TimesFM-3.0) onto the 21 Chronos-2 levels.

CRPS averages the quantile loss over the stored levels, so forecasts on different quantile grids are
scored on the same 21 levels. The quantile function is interpolated linearly between the source
levels; target levels outside the source range are extrapolated with the slope of the two outermost
source quantiles, which keeps the quantiles monotone.

The target grid is the one stored in the Chronos-2 checkpoints, which hold the levels in bf16
(e.g. 0.99 is stored as 0.98828125). The nine TimesFM-3.0 levels coincide with every second entry of
this grid, so only the twelve added levels are interpolated.

    python chronos_llm/scripts/utils/interp_quantiles.py IN.npz OUT.npz

OUT.npz carries every array of IN.npz, with ``pred_quantiles`` and ``quantile_levels`` replaced.
"""

import argparse

import numpy as np

CHRONOS_21 = (0.010009766, 0.050048828, 0.100097656, 0.150390625, 0.200195312, 0.25,
              0.30078125, 0.349609375, 0.400390625, 0.44921875, 0.5, 0.55078125,
              0.6015625, 0.6484375, 0.69921875, 0.75, 0.80078125, 0.8515625,
              0.8984375, 0.94921875, 0.98828125)


def interp_quantiles(pred, src_levels, dst_levels=CHRONOS_21, tol=0.01):
    """(N, Q_src, H) quantile forecasts -> (N, len(dst), H) on ``dst_levels``.

    A forecast already on the target grid (within ``tol``, which absorbs the bf16 / float32 storage
    of the same levels) is returned unchanged.
    """
    src = np.asarray(src_levels, dtype=float)
    dst = np.asarray(dst_levels, dtype=float)
    if len(src) == len(dst) and np.all(np.abs(src - dst) <= tol):
        return pred, src
    n, _, h = pred.shape
    flat = np.asarray(pred, dtype=np.float64).transpose(0, 2, 1).reshape(-1, len(src))
    out = np.empty((flat.shape[0], len(dst)), dtype=np.float64)
    for j, q in enumerate(dst):
        if q <= src[0]:
            slope = (flat[:, 1] - flat[:, 0]) / (src[1] - src[0])
            out[:, j] = flat[:, 0] + slope * (q - src[0])
        elif q >= src[-1]:
            slope = (flat[:, -1] - flat[:, -2]) / (src[-1] - src[-2])
            out[:, j] = flat[:, -1] + slope * (q - src[-1])
        else:
            i = int(np.searchsorted(src, q) - 1)
            w = (q - src[i]) / (src[i + 1] - src[i])
            out[:, j] = flat[:, i] * (1.0 - w) + flat[:, i + 1] * w
    return out.reshape(n, h, len(dst)).transpose(0, 2, 1).astype(pred.dtype), dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp", help="npz written by infer_forecast.py or baseline_forecast_zeroshot.py")
    ap.add_argument("out", help="output npz on the 21 Chronos-2 levels")
    a = ap.parse_args(argv)
    d = dict(np.load(a.inp, allow_pickle=True))
    d["pred_quantiles"], levels = interp_quantiles(d["pred_quantiles"], d["quantile_levels"])
    d["quantile_levels"] = np.asarray(levels, dtype=np.float32)
    np.savez(a.out, **d)
    print(f"{a.inp} -> {a.out}: {d['pred_quantiles'].shape[1]} quantile levels")


if __name__ == "__main__":
    main()
