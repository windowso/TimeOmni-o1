"""Checks for the TimesFM-3.0 -> chronos2 interface adapter (the arm of the backbone ablation that
swaps the TSFM).

The critical path is **bucketing by channel count**: TimesFM wants a rectangular (b, v, n, p), while
our context is (sum_C, L) folded rows + group_ids (the number of rows in a group = that sample's
channel count, which varies). If the bucketing implementation ever crosses rows over, training will
not raise -- it will just quietly learn the wrong thing. Check 3 therefore runs every group on its
own as ground truth and requires the bucketed batch run to match it **value for value**.

1) interface surface: every attribute/method the trunk reads exists and has the expected value;
2) encode shape (R, n_patch, d_model), context patches first;
3) **bucketing equivalence**: with mixed channel counts (groups of C=1/2/3 in one batch), identical
   value for value to running each group separately;
4) forecast_losses: the loss is finite and differentiable, roi_loss has the same magnitude as
   pred_loss, target_idx slicing is correct;
5) the feedback works: with cross_states supplied and the gate opened, the forecast really changes;
6) de-normalisation: quantile_preds come back to the original scale (same magnitude as future);
7) the minimal C=1 / B=1 configuration.
"""

import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO, os.path.join(_REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from chronos_llm.models.timesfm_backbone import load_timesfm3_backbone  # noqa: E402

CKPT = os.environ.get("TIMESFM3_PATH", "checkpoints/TimesFM3.0")


def main() -> None:
    torch.manual_seed(0)
    bb = load_timesfm3_backbone(CKPT)
    bb.eval()

    d_model = bb.config.d_model
    ops = bb.chronos_config.output_patch_size
    ips = bb.chronos_config.input_patch_size

    # ---- 1: interface surface ----
    assert d_model == 1280, d_model
    assert ips == 32 and ops == 64, (ips, ops)
    assert bb.chronos_config.context_length == 8192, "the sliding-window chunk size must match chronos-2 (protocol decision 1)"
    assert len(bb.encoder.block) == 20, len(bb.encoder.block)
    assert bb.num_quantiles == 9 and abs(float(bb.quantiles[4]) - 0.5) < 1e-9
    assert bb.dtype == torch.float32
    _, (loc0, scale0) = bb.instance_norm(torch.randn(3, 64))
    assert loc0.shape == (3, 1) and scale0.shape == (3, 1), (loc0.shape, scale0.shape)
    print(f"[1] interface surface OK: d_model={d_model} patch={ips}/{ops} layers={len(bb.encoder.block)} "
          f"quantiles={bb.num_quantiles} ctx_len={bb.chronos_config.context_length}")

    # ---- 2: encode shape ----
    L = 256
    ctx = torch.randn(4, L) * 3.0 + 10.0
    gids = torch.tensor([0, 0, 1, 1])          # two samples, 2 channels each
    _, (loc, scale) = bb.instance_norm(ctx)
    with torch.no_grad():
        enc, ls, _, ncp = bb.encode(context=ctx, num_output_patches=1, group_ids=gids,
                                    loc_scale=(loc, scale))
    hid = enc[0]
    print(f"[2] encode -> {tuple(hid.shape)}, ncp={ncp} (L={L}, patch={ips} => expected {L // ips})")
    assert ncp == L // ips
    assert hid.shape[0] == 4 and hid.shape[2] == d_model
    assert hid.shape[1] >= ncp, "context patches must come first, and the total patch count must be >= ncp"
    assert torch.isfinite(hid).all()

    # ---- 3: bucketing equivalence (mixed channel counts) ----
    # 5 groups with C = 1, 3, 2, 1, 3 => three buckets, the order deliberately shuffled
    group_sizes = [1, 3, 2, 1, 3]
    gids_mix = torch.tensor([g for g, c in enumerate(group_sizes) for _ in range(c)])
    R = int(gids_mix.numel())
    ctx_mix = torch.randn(R, L) * torch.rand(R, 1) * 5 + torch.randn(R, 1) * 10
    _, (locm, scalem) = bb.instance_norm(ctx_mix)
    with torch.no_grad():
        enc_batch, _, _, _ = bb.encode(context=ctx_mix, num_output_patches=1, group_ids=gids_mix,
                                       loc_scale=(locm, scalem))
    hid_batch = enc_batch[0]

    # ground truth: run every group on its own
    hid_ref = torch.empty_like(hid_batch)
    row = 0
    with torch.no_grad():
        for g, c in enumerate(group_sizes):
            rows = slice(row, row + c)
            e_g, _, _, _ = bb.encode(
                context=ctx_mix[rows], num_output_patches=1,
                group_ids=torch.zeros(c, dtype=torch.long), loc_scale=(locm[rows], scalem[rows]),
            )
            hid_ref[rows] = e_g[0]
            row += c
    d_bucket = (hid_batch - hid_ref).abs().max().item()
    print(f"[3] bucketing equivalence (groups with channel counts {group_sizes} in one batch): max|delta| = {d_bucket:.3e}")
    assert d_bucket < 1e-4, "the bucketed batch run disagrees with running each group on its own -- the bucketing/scatter indices cross rows over"

    # ---- 4: forecast_losses ----
    fl = 48
    nop = -(-fl // ops)
    ctx_f = torch.randn(6, L) * 2.0 + 5.0
    gids_f = torch.arange(6)                    # univariate: every row is its own group (as the forecasting corpus really is)
    future = torch.randn(6, fl) * 2.0 + 5.0
    roi = torch.zeros(6, fl)
    roi[:, 10:30] = 1.0
    res = bb.forecast_losses(context=ctx_f, future_target=future, num_output_patches=nop,
                             group_ids=gids_f, roi_mask=roi)
    pl, rl, qp = res["pred_loss"], res["roi_loss"], res["quantile_preds"]
    print(f"[4] forecast_losses: pred_loss={float(pl):.4f} roi_loss={float(rl):.4f} "
          f"quantile_preds={tuple(qp.shape)}")
    assert torch.isfinite(pl) and torch.isfinite(rl)
    assert qp.shape == (6, bb.num_quantiles, nop * ops)
    assert 0.05 < float(pl) / float(rl) < 20, "pred/roi magnitudes differ too much -- both should share the same pinball"
    pl.backward()
    # WARNING: this must check for "non-zero" and not merely "not None": an all-zero gradient means
    # just as surely that this part of the backbone cannot learn, and `p.grad is not None` is true for
    # an all-zero gradient too.
    # Only the parameters that **took part in this forward pass** are asserted on: no cross_states were
    # given here, so the injected cross_attn is naturally not in the graph (its gradients are covered
    # by check 5 and by test_timesfm_cross_attn).
    body = [(n, p) for n, p in bb.named_parameters() if ".cross_attn." not in ("." + n)]
    n_fin = sum(1 for _, p in body if p.grad is not None and torch.isfinite(p.grad).all())
    zero_names = [n for n, p in body if p.grad is None or float(p.grad.abs().sum()) == 0]
    print(f"[4] backward: {n_fin}/{len(body)} backbone-proper parameters have finite gradients, {len(body) - len(zero_names)} are non-zero"
          + (f"; zero gradients {zero_names[:3]}" if zero_names else ""))
    assert n_fin == len(body), "some backbone-proper parameters did not receive a finite gradient"
    assert not zero_names, f"{len(zero_names)} backbone parameters have an all-zero gradient: {zero_names[:5]}"
    bb.zero_grad(set_to_none=True)

    # ---- 4b: backward under NaN padding (the real collator left-pads with NaN) ----
    # If the padding is not handed to TimesFM as a mask, its causal running-stats compute std=0 over
    # the all-zero prefix => RevIN divides by zero => the forward pass is still finite (the loss looks
    # normal) while the **backward pass is NaN throughout**.
    bb.zero_grad(set_to_none=True)
    ctx_pad = ctx_f.clone()
    ctx_pad[1, :96] = float("nan")      # left-pad half of it
    ctx_pad[3, :160] = float("nan")     # left-pad most of it, leaving only 96 points of real history
    res_pad = bb.forecast_losses(context=ctx_pad, future_target=future, num_output_patches=nop,
                                 group_ids=gids_f, roi_mask=roi)
    assert torch.isfinite(res_pad["pred_loss"]), "pred_loss is not finite under NaN padding"
    res_pad["pred_loss"].backward()
    body_pad = [(n, p) for n, p in bb.named_parameters() if ".cross_attn." not in ("." + n)]
    bad = [n for n, p in body_pad
           if p.grad is None or not bool(torch.isfinite(p.grad).all()) or float(p.grad.abs().sum()) == 0]
    print(f"[4b] backward under NaN padding: {len(body_pad) - len(bad)}/{len(body_pad)} backbone parameters have finite non-zero gradients")
    assert not bad, f"under NaN padding {len(bad)} backbone parameters have NaN/zero gradients: {bad[:5]}"
    bb.zero_grad(set_to_none=True)

    # target_idx slicing: the first 3 rows are targets
    tidx = torch.zeros(6, dtype=torch.bool)
    tidx[:3] = True
    res_t = bb.forecast_losses(context=ctx_f, future_target=future[:3], num_output_patches=nop,
                               group_ids=gids_f, target_idx=tidx)
    assert res_t["quantile_preds"].shape[0] == 3, res_t["quantile_preds"].shape
    print(f"[4] target_idx slicing OK -> {tuple(res_t['quantile_preds'].shape)}")

    # ---- 4c: bf16 backbone + fp32 input (the real situation under ZeRO-2) ----
    # At scale the parameters are cast to bf16 while the collator still hands over an fp32 context; if
    # the adapter does not align the dtypes it blows up on the first matmul.
    # A single-GPU bf16 smoke test does not catch it -- on that path autocast promotes automatically
    # and the parameters are in fact still fp32.
    bb_bf16 = bb.to(torch.bfloat16)
    try:
        res_bf = bb_bf16.forecast_losses(context=ctx_f, future_target=future,
                                         num_output_patches=nop, group_ids=gids_f, roi_mask=roi)
        assert torch.isfinite(res_bf["pred_loss"]), "pred_loss is not finite under bf16"
        enc_bf, _, _, _ = bb_bf16.encode(context=ctx_f, num_output_patches=1, group_ids=gids_f)
        print(f"[4c] bf16 backbone + fp32 input: pred_loss={float(res_bf['pred_loss']):.4f} "
              f"encode dtype={enc_bf[0].dtype} quantile_preds dtype={res_bf['quantile_preds'].dtype}")
        assert enc_bf[0].dtype == torch.bfloat16, "the encode output should inherit the backbone dtype"
    finally:
        bb_bf16.to(torch.float32)
    bb.zero_grad(set_to_none=True)

    # ---- 5: the feedback works ----
    with torch.no_grad():
        for n, prm in bb.named_parameters():
            if n.endswith(".cross_attn.gate"):
                prm.fill_(1.0)
            if ".cross_attn.attn.out_proj.weight" in ("." + n):
                prm.normal_(0, 0.02)
    M = 12
    cs = torch.randn(6, M, d_model)
    csm = torch.ones(6, M)
    with torch.no_grad():
        qp_off = bb(ctx_f, num_output_patches=nop, group_ids=gids_f).quantile_preds
        qp_on = bb(ctx_f, num_output_patches=nop, group_ids=gids_f,
                   cross_states=cs, cross_states_mask=csm).quantile_preds
    d_flow = (qp_off - qp_on).abs().max().item()
    print(f"[5] feedback: max|delta(with cross_states vs without)| = {d_flow:.3e}")
    assert d_flow > 1e-4, "cross_states did not make it through the adapter into TimesFM"

    # ---- 6: de-normalised back to the original scale ----
    med = qp_off[:, bb.num_quantiles // 2, :fl]
    print(f"[6] median forecast mean={med.mean():.3f} (context mean={ctx_f.mean():.3f}, should be the same magnitude)")
    assert abs(float(med.mean()) - float(ctx_f.mean())) < 5.0, "the forecast did not come back to the original scale"

    # ---- 7: C=1 / B=1 ----
    ctx1 = torch.randn(1, 128) * 2 + 7
    with torch.no_grad():
        e1, _, _, ncp1 = bb.encode(context=ctx1, num_output_patches=1,
                                   group_ids=torch.zeros(1, dtype=torch.long))
        q1 = bb(ctx1, num_output_patches=1, group_ids=torch.zeros(1, dtype=torch.long)).quantile_preds
    print(f"[7] B=1/C=1: encode={tuple(e1[0].shape)} ncp={ncp1} quantile_preds={tuple(q1.shape)}")
    assert e1[0].shape == (1, e1[0].shape[1], d_model) and ncp1 == 128 // ips
    assert q1.shape == (1, bb.num_quantiles, ops) and torch.isfinite(q1).all()

    # ---- 8: unimplemented paths must fail loudly instead of silently ----
    for kw in ({"future_covariates": torch.zeros(6, fl)}, {"stop_at_layer": 3}):
        try:
            bb.encode(context=ctx_f, group_ids=gids_f, **kw)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{list(kw)[0]} should raise explicitly instead of being ignored silently")
    print("[8] the unimplemented paths (covariates / stop_at_layer) both raise explicitly")

    print("\nTIMESFM ADAPTER CHECKS PASSED")


if __name__ == "__main__":
    main()
