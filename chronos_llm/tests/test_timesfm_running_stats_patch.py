"""Guards the backward-stability patch for TimesFM's CPM-RevIN statistics (the monkey-patch in
`cross_attn_timesfm`).

When a patch is entirely masked (the leading patches of a left-padded history) or the variance is 0,
upstream ``util.update_running_stats`` is fine in the forward pass -- ``torch.where`` discards the
0/0 -- but produces NaN in the **backward** pass, because the branch that was not selected
contributes ``1/0=inf`` and ``d sqrt(0)=inf``, which are then multiplied by the 0 handed down by
``where``. Upstream is inference-only (decode carries no_grad) so it never gets there; we train with
it and therefore must fix it.

The requirement on the patch is that it **changes only the backward pass and not the forward pass**,
so both properties are checked here:
1) forward: for random inputs and for inputs containing fully masked patches, the patched version is
   **bit-identical** to the upstream original;
2) backward: the upstream original produces NaN gradients on fully masked patches, the patched
   version gives finite gradients.
"""

import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO, os.path.join(_REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Importing applies the patch (module side effect); the original implementation is kept around for
# comparison.
from chronos_llm.models.cross_attn_timesfm import (  # noqa: E402
    _ORIG_UPDATE_RUNNING_STATS,
    _update_running_stats_safe,
)


def _run(fn, x, mask):
    """Run the rolling (n, mu, sigma) update over every patch and return the stacked statistics (the
    caller sums them into the scalar to backpropagate from)."""
    b, v, n_patch, p = x.shape
    n = torch.zeros(b, v)
    mu = torch.zeros(b, v)
    sigma = torch.zeros(b, v)
    outs = []
    for i in range(n_patch):
        n, mu, sigma = fn(n, mu, sigma, x[:, :, i, :], mask[:, :, i, :])
        outs.append(torch.stack([n, mu, sigma], dim=-1))
    return torch.stack(outs, dim=2)


def main() -> None:
    torch.manual_seed(0)
    b, v, n_patch, p = 2, 3, 6, 8

    cases = {
        "no mask": torch.zeros(b, v, n_patch, p, dtype=torch.bool),
        "partial mask": torch.rand(b, v, n_patch, p) < 0.3,
    }
    # The critical scenario: the first two patches are fully masked (= the leading patches of a
    # left-padded history)
    lead = torch.zeros(b, v, n_patch, p, dtype=torch.bool)
    lead[:, :, :2, :] = True
    cases["leading patches fully masked"] = lead
    # A constant series => the variance is always 0 => the gradient of sqrt(0) is inf
    cases["constant series (zero variance)"] = torch.zeros(b, v, n_patch, p, dtype=torch.bool)

    orig_broken: list[str] = []
    for tag, mask in cases.items():
        x = torch.randn(b, v, n_patch, p)
        if tag == "constant series (zero variance)":
            x = torch.full((b, v, n_patch, p), 3.5)

        # ---- 1) the forward pass is bit-identical ----
        with torch.no_grad():
            ref = _run(_ORIG_UPDATE_RUNNING_STATS, x, mask)
            got = _run(_update_running_stats_safe, x, mask)
        d = (ref - got).abs().max().item()
        print(f"[{tag}] forward max|delta(patched - upstream)| = {d:.3e}")
        assert d == 0.0, f"{tag}: the patch changed the forward values"

        # ---- 2) backward: the patched version must be finite ----
        xg = x.clone().requires_grad_(True)
        _run(_update_running_stats_safe, xg, mask).sum().backward()
        ok = bool(torch.isfinite(xg.grad).all())
        xo = x.clone().requires_grad_(True)
        _run(_ORIG_UPDATE_RUNNING_STATS, xo, mask).sum().backward()
        orig_ok = bool(torch.isfinite(xo.grad).all())
        print(f"[{tag}] backward finite: patched={ok}  upstream original={orig_ok}")
        assert ok, f"{tag}: the patched version still produces NaN/Inf gradients"
        if not orig_ok:
            orig_broken.append(tag)

    # Deliberately no per-scenario assertion that "upstream must break" -- that would be a brittle
    # assertion: the backward of ``torch.where`` is itself a where (not a multiplication), so the NaN
    # from the 0/0 branch is blocked by the next where, and "leading patches fully masked" turns out
    # not to break upstream in practice. What really breaks is the **gradient of sqrt(0), which is
    # inf**, i.e. any **constant segment** -- and flat stretches are common in real time series
    # (saturated sensors, zeros at night, left padding filled with 0), so this patch is a training
    # necessity, not merely a padding fix. All that is required here is that upstream breaks in at
    # least one scenario, confirming that the patch still has a purpose.
    print(f"\nScenarios where the upstream original produces NaN gradients: {orig_broken}")
    assert orig_broken, ("upstream no longer produces NaN gradients in any scenario -- it may have "
                         "been fixed there, in which case re-examine whether this patch is still "
                         "needed (forward equivalence is already guarded by the check above)")
    print("RUNNING-STATS BACKWARD-STABILITY PATCH CHECKS PASSED (forward bit-identical, backward no longer NaN)")


if __name__ == "__main__":
    main()
