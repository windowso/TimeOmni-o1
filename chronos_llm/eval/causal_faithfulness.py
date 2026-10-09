"""Counterfactual magnitude edit: does the numeric forecast follow the magnitude statement in the
forecast instruction?

Two checks:
  (a) gate=0 identity (``gate_identity_report``): with the cross-attention gate at zero the model's
      time series branch is numerically identical to the original Chronos-2 (also covered by
      ``chronos_llm/tests/test_cross_attn_identity.py``).
  (b) counterfactual forward pass (``fresh_forward_report``, ``--fresh_forward``): for each sampled test
      row, the magnitude statement of the annotated forecast instruction is replaced by each of five
      statements (``BAND_CLAUSES``), everything else (history, background, event, reasoning) is kept,
      and the model is run teacher-forced. The forecast peak / mean inside the ROI is compared across
      the five statements: Spearman correlation between the stated level and the forecast, and the
      direction agreement for opposite statements.

Usage:
  python chronos_llm/eval/causal_faithfulness.py                        # gate=0 identity check only
  python chronos_llm/eval/causal_faithfulness.py --fresh_forward --skip_aggregate \\
    --model_path <checkpoint dir> --fresh_n_samples 80
"""
import argparse
import copy
import json
import os
import re

import numpy as np
from scipy.stats import spearmanr

from chronos_llm.scripts.utils.make_structured_conclusion import BAND_TAG

CHRONOS_CKPT = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
BANDS = ["tiny", "well_below", "mod_below", "typical", "above"]


# ---------------------------------------------------------------- (a) gate=0 identity
def gate_identity_report():
    """Re-verify that with gate=0 Chronos2WithCrossAttn is numerically identical to the original
    Chronos2Model (same loading logic as test_cross_attn_identity, so the report is self-contained and
    does not require that test file to have been run)."""
    import torch
    import safetensors.torch as st
    from chronos.chronos2 import Chronos2Model
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn

    torch.manual_seed(0)
    cfg = Chronos2CoreConfig.from_pretrained(CHRONOS_CKPT)
    base = Chronos2Model(cfg)
    base.load_state_dict(st.load_file(CHRONOS_CKPT + "/model.safetensors"), strict=False)
    base = base.eval()
    xattn = load_chronos2_with_cross_attn(CHRONOS_CKPT)

    B, L, n_out, M = 2, 160, 4, 16
    context = torch.randn(B, L)
    cross = torch.randn(B, M, xattn.config.d_model)
    cross_mask = torch.ones(B, M)
    with torch.no_grad():
        out_base = base(context=context, num_output_patches=n_out)
        out_gate0 = xattn(context=context, num_output_patches=n_out,
                          cross_states=cross, cross_states_mask=cross_mask)
    diff = (out_base.quantile_preds - out_gate0.quantile_preds).abs().max().item()
    ok = diff < 1e-4
    print(f"[identity ablation] gate=0: max|base - xattn(cross_states provided)| = {diff:.3e} "
          f"{'OK: exactly identical' if ok else 'FAIL: not identical -- mechanism-level problem!'}")
    print("  => without textual feedback the model is numerically the original Chronos-2 "
          "(a mechanism guarantee, not a training outcome).")
    return ok


# ---------------------------------------------------------------- helpers
def _peak(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.nanmax(a)) if a.size else np.nan


# ---------------------------------------------------------------- (b) counterfactual forward pass
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/timeomni-o1")
DEFAULT_FRESH_PARQUET = (
    "data/forecast/mmtr_forecast_corpus.parquet"
)

# The five magnitude statements substituted into the forecast instruction, ordered from lowest to
# highest stated level. above / well_below and above / tiny are the opposite-direction pairs.
BAND_CLAUSES = {
    "tiny": "the peak reaches only a small fraction of the series' typical high level",
    "well_below": "the peak runs well below the series' typical highs",
    "mod_below": "the peak sits moderately below the series' typical highs",
    "typical": "the peak sits close to the series' typical high level",
    "above": "the peak runs well above the series' typical highs",
}
_OVERALL_RE = re.compile(r"\s*Overall,.*$", re.S)
_MAGTAG_RE = re.compile(r"^\[Magnitude:\s*[A-Z_]+\]\s*")
_OPPOSITE_PAIRS = [("above", "well_below"), ("above", "tiny")]


def edit_conclusion(concl, clause, band=None):
    """Strip any existing "Overall, ..." magnitude sentence and append the one for the requested band.
    Only the conclusion changes; reasoning / background / event and history_values are untouched.
    When ``band`` is not None, a leading `[Magnitude: TAG]` tag is replaced as well, so that the tag
    and the final sentence state the same level."""
    s = str(concl)
    prefix = ""
    m = _MAGTAG_RE.match(s)
    if m:
        s = s[m.end():]
        prefix = f"[Magnitude: {BAND_TAG[band]}] " if band is not None else m.group(0)
    base = _OVERALL_RE.sub("", s).rstrip()
    if base and base[-1] not in ".!?":
        base += "."
    return f"{prefix}{base} Overall, {clause}."


def _variant_dataset(base_ds, sub_df):
    """Shallow-copy base_ds (reusing its tokenizer / rendering hyper-parameters) and swap only the df --
    all variants share one rendering code path (build_supervised_ids' LCP supervision locator etc.)
    instead of re-deriving the input format."""
    v = copy.copy(base_ds)
    v.df = sub_df.reset_index(drop=True)
    v._skip = set()
    return v


def _build_items_strict(ds):
    """Call _build_item row by row and require non-None, bypassing __getitem__'s lazy skip-and-substitute.
    A diagnostic script needs strict row alignment: any row whose edited rendering fails (too long / no
    supervised content) must fail loudly instead of being silently replaced by another sample (otherwise
    the row order across variants drifts silently and every row-wise comparison is wrong)."""
    items = []
    for i in range(len(ds)):
        it = ds._build_item(i)
        if it is None:
            row = ds.df.iloc[i]
            raise RuntimeError(
                f"row {i} (id={row.get('id')}) failed to render after the counterfactual edit (too long / no supervised content) -- "
                "the diagnostic requires strict row alignment; inspect this row or lower --fresh_n_samples to resample")
        items.append(it)
    return items


def _gate_params(model):
    return [(n, p) for n, p in model.named_parameters() if "chronos" in n and "cross_attn.gate" in n]


def _row_peak_mean(qp_row, valid_row, roi_row, median_idx):
    """Peak / mean of the q0.5 median forecast curve within ROI∩valid (falls back to all valid positions
    when the ROI is empty)."""
    med = qp_row[median_idx]
    roi = roi_row.astype(bool) & valid_row.astype(bool)
    m = roi if roi.any() else valid_row.astype(bool)
    if not m.any():
        return float("nan"), float("nan")
    vals = med[m]
    return float(np.nanmax(vals)), float(np.nanmean(vals))


def _rank_summary(mat, orig, ok_mask, band_order):
    """mat/orig: (len(band_order), N) / (N,) peak or mean ratios normalised by typ. Spearman is invariant to a
    positive per-row rescaling (typ is one constant across the 5 bands of a row) -- the normalisation only
    makes the "response magnitude" comparable across datasets; it does not affect the correlation or the
    direction-agreement rate."""
    idx = np.where(ok_mask)[0]
    rhos, strict = [], 0
    for i in idx:
        col = mat[:, i]
        if np.isfinite(col).all():
            r = spearmanr(np.arange(len(band_order)), col).statistic
            if np.isfinite(r):
                rhos.append(r)
                strict += int(np.all(np.diff(col) > 0))
    rhos = np.array(rhos)
    out = {
        "n": int(len(idx)), "n_finite_spearman": int(len(rhos)),
        "spearman_mean": float(np.mean(rhos)) if rhos.size else float("nan"),
        "spearman_pos_rate": float(np.mean(rhos > 0)) if rhos.size else float("nan"),
        "strict_mono_rate": float(strict / len(rhos)) if rhos.size else float("nan"),
    }
    for a, b in _OPPOSITE_PAIRS:
        ai, bi = band_order.index(a), band_order.index(b)
        va, vb = mat[ai, idx], mat[bi, idx]
        ok = np.isfinite(va) & np.isfinite(vb)
        out[f"dir_follow_rate({a}>{b})"] = float(np.mean(va[ok] > vb[ok])) if ok.any() else float("nan")
    dev = float(np.nanmean([np.nanmean(np.abs(mat[b, idx] - orig[idx])) for b in range(len(band_order))]))
    out["mean_abs_dev_vs_orig"] = dev
    out["ratio_orig"] = float(np.nanmean(orig[idx])) if idx.size else float("nan")
    for b, name in enumerate(band_order):
        out[f"ratio_{name}"] = float(np.nanmean(mat[b, idx])) if idx.size else float("nan")
    return out


def fresh_forward_report(model_path, parquet, n_samples=80, seed=0, batch_size=16,
                         max_user_tokens=1500, max_tokens=4096, device="cuda",
                         gate_scales=(1.0,), dump_examples=3, output=None,
                         save_per_sample_raw=None):
    """Counterfactual forward pass: load with ChronosLLM.from_pretrained -> edit the conclusion of the
    same sampled rows band by band (history_values/background/event/reasoning all unchanged) ->
    generate_forecast_teacher_forced forward pass (no autoregressive generation; the hidden states of the
    teacher-forced text are fed back into Chronos-2, the same code path as the teacher-forced mode of
    chronos_llm/eval/infer_forecast.py) -> compare the peak / mean of the median (q0.5) forecast curve
    within ROI∩valid.

    ``save_per_sample_raw``: when a path is given, additionally dump the raw per-sample, per-band
    ``peak_ratio``/``mean_ratio`` matrices (``_rank_summary`` keeps only aggregates). Single gate_scale
    only (raises when ``len(gate_scales) > 1``).
    """
    if save_per_sample_raw and len(gate_scales) > 1:
        raise ValueError(
            "--save_per_sample_raw supports a single gate_scale only (several were given, so it is unclear "
            "which one's raw matrices to save)")
    import torch
    from chronos_llm.data.collator import ChronosLLMCollator
    from chronos_llm.data.forecast_dataset import ForecastParquetDataset
    from chronos_llm.eval.metrics import _median_index
    from chronos_llm.eval.text_alignment_eval import domain_of
    from chronos_llm.models.chronos_llm_model import ChronosLLM

    print(f"[fresh_forward] loading model {model_path} ...")
    model = ChronosLLM.from_pretrained(model_path, merge=True).eval().to(device)
    tok = model.tokenizer
    median_idx = _median_index(model.chronos.quantiles.detach().cpu().float().numpy())
    collator = ChronosLLMCollator(tokenizer=tok)

    base_ds = ForecastParquetDataset(
        parquet, tok, split="test", inference_mode=False, emit_meta=False,
        max_user_tokens=max_user_tokens, max_tokens=max_tokens,
    )
    full_df = base_ds.df
    print(f"[fresh_forward] usable test rows = {len(full_df)}")
    # Per-dataset "typical peak" = p90 of the future peaks over all (unsampled) test rows of that dataset;
    # estimating on the full test split rather than the sampled subset is more stable.
    typ_map = full_df["future_values"].map(_peak).groupby(full_df["dataset_name"]).quantile(0.90).to_dict()

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(full_df))
    idx = np.sort(rng.choice(len(full_df), size=n, replace=False))
    sub_df_base = full_df.iloc[idx].reset_index(drop=True)
    ds_names = sub_df_base["dataset_name"].astype(str).to_numpy()
    ids = (sub_df_base["id"].astype(str).to_numpy() if "id" in sub_df_base.columns
           else np.array([str(i) for i in range(n)]))
    typ = np.array([typ_map.get(d, np.nan) for d in ds_names])
    typ_ok = np.isfinite(typ) & (typ > 1e-9)
    print(f"[fresh_forward] sampled {n} rows (seed={seed}), "
          f"dataset_name coverage: {sub_df_base['dataset_name'].value_counts().to_dict()}")
    # Reuse the source -> domain mapping of text_alignment_eval; sources outside the list go to "Unknown"
    # (never silently dropped).
    domains = np.array([domain_of(d) or "Unknown" for d in ds_names])

    gate_snapshot = [p.detach().clone() for _, p in _gate_params(model)]
    if not gate_snapshot:
        raise RuntimeError("no chronos cross_attn.gate parameters found in the model -- does this checkpoint architecture lack gated cross-attention?")
    print(f"[fresh_forward] found {len(gate_snapshot)} chronos cross_attn.gate scalar parameters")

    def _to_device(batch):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _run_variant(items):
        preds, valids, rois = [], [], []
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            batch = collator(chunk)
            n_tg = batch.get("n_targets")
            if n_tg is not None:
                assert int(n_tg.sum()) == len(chunk), (
                    "this function assumes single-channel / single-target rows (the MMTR forecasting corpus has no covariates); "
                    "sum of n_targets differs from the number of samples -- extra multi-target expansion logic would be needed")
            batch_dev = _to_device(batch)
            fl = batch["future"].shape[-1]
            with torch.no_grad():
                out = model.generate_forecast_teacher_forced(batch_dev, horizon=fl)
            qp = out["quantile_preds"].detach().cpu().float().numpy()[..., :fl]
            gt_np = batch["future"].numpy()
            valid = ~np.isnan(gt_np)
            roi_np = batch["roi_mask"].numpy()
            for i in range(qp.shape[0]):
                preds.append(qp[i]); valids.append(valid[i]); rois.append(roi_np[i])
        return preds, valids, rois

    band_order = BANDS  # lowest to highest stated level (tiny/well_below/mod_below/typical/above)
    report = {"experiments": {}, "examples": []}
    for scale in gate_scales:
        with torch.no_grad():
            for (_, p), orig_p in zip(_gate_params(model), gate_snapshot):
                p.copy_(orig_p * scale)
        variants = ["orig"] + band_order
        band_preds = {}
        ref_valid = ref_roi = None
        examples = ([{"row": int(i), "id": str(ids[i]), "dataset_name": str(ds_names[i]), "bands": {}}
                     for i in range(min(dump_examples, n))] if scale == gate_scales[0] else [])
        for band in variants:
            sub = sub_df_base.copy()
            if band != "orig":
                sub["conclusion"] = [edit_conclusion(c, BAND_CLAUSES[band], band=band)
                                      for c in sub["conclusion"]]
            vds = _variant_dataset(base_ds, sub)
            items = _build_items_strict(vds)
            for ex in examples:  # plumbing-check material: tail of the rendered text after editing (for manual verification)
                ex["bands"][band] = tok.decode(items[ex["row"]]["input_ids"], skip_special_tokens=False)[-500:]
            preds, valids, rois = _run_variant(items)
            band_preds[band] = preds
            if band == "orig":
                ref_valid, ref_roi = valids, rois
            else:
                # Alignment guard: future/roi must not change with the conclusion edit; a mismatch means the
                # row order drifted between variants.
                for i in range(n):
                    if not (np.array_equal(valids[i], ref_valid[i]) and np.array_equal(rois[i], ref_roi[i])):
                        raise RuntimeError(f"row {i} (band={band}): valid/roi differ from orig -- row order drifted between variants")

        peak_mat = np.full((len(band_order), n), np.nan)
        mean_mat = np.full((len(band_order), n), np.nan)
        for bi, band in enumerate(band_order):
            for i in range(n):
                pk, mn = _row_peak_mean(band_preds[band][i], ref_valid[i], ref_roi[i], median_idx)
                peak_mat[bi, i] = pk; mean_mat[bi, i] = mn
        orig_pk = np.array([_row_peak_mean(band_preds["orig"][i], ref_valid[i], ref_roi[i], median_idx)[0]
                            for i in range(n)])
        orig_mn = np.array([_row_peak_mean(band_preds["orig"][i], ref_valid[i], ref_roi[i], median_idx)[1]
                            for i in range(n)])
        denom = np.where(typ_ok, typ, np.nan)
        peak_ratio, mean_ratio = peak_mat / denom, mean_mat / denom
        orig_pk_ratio, orig_mn_ratio = orig_pk / denom, orig_mn / denom
        peak_summary = _rank_summary(peak_ratio, orig_pk_ratio, typ_ok, band_order)
        mean_summary = _rank_summary(mean_ratio, orig_mn_ratio, typ_ok, band_order)
        # Per-domain split (intersected with typ_ok); _rank_summary also computes for domains with few rows.
        peak_by_domain, mean_by_domain = {}, {}
        for dom in sorted(set(domains.tolist())):
            dom_mask = typ_ok & (domains == dom)
            peak_by_domain[dom] = _rank_summary(peak_ratio, orig_pk_ratio, dom_mask, band_order)
            mean_by_domain[dom] = _rank_summary(mean_ratio, orig_mn_ratio, dom_mask, band_order)

        if save_per_sample_raw:
            os.makedirs(os.path.dirname(os.path.abspath(save_per_sample_raw)) or ".", exist_ok=True)
            np.savez(
                save_per_sample_raw,
                peak_ratio=peak_ratio, mean_ratio=mean_ratio,
                orig_pk_ratio=orig_pk_ratio, orig_mn_ratio=orig_mn_ratio,
                typ_ok=typ_ok, ids=ids, dataset_names=ds_names, domains=domains,
                band_order=np.array(band_order), gate_scale=np.array(scale),
            )
            saved_path = save_per_sample_raw if save_per_sample_raw.endswith(".npz") \
                else save_per_sample_raw + ".npz"
            print(f"[fresh_forward] raw per-sample, per-band matrices written -> {saved_path} "
                  f"(peak_ratio/mean_ratio shape={peak_ratio.shape})")

        # Plumbing self-check (ignores direction; only asks whether the edit changed the model output at all,
        # to catch "batching / edit did not take effect" script bugs: 0 means the variant forecasts are
        # bitwise equal).
        a_preds, b_preds = band_preds["above"], band_preds["well_below"]
        raw_diff = np.array([float(np.nanmax(np.abs(a_preds[i] - b_preds[i]))) for i in range(n)])
        plumbing = {
            "frac_rows_pred_changed_above_vs_well_below": float(np.mean(raw_diff > 1e-6)),
            "mean_abs_raw_diff_above_vs_well_below": float(np.nanmean(raw_diff)),
        }
        report["experiments"][str(scale)] = {
            "n": n, "peak": peak_summary, "roi_mean": mean_summary, "plumbing_check": plumbing,
            "peak_by_domain": peak_by_domain, "roi_mean_by_domain": mean_by_domain,
        }
        if examples:
            report["examples"] = examples
        p_sum = peak_summary
        print(f"[fresh_forward][gate={scale}] n={n} peak Spearman={p_sum['spearman_mean']:.3f} "
              f"(rho>0 {p_sum['spearman_pos_rate']:.2f}, strictly monotone {p_sum['strict_mono_rate']:.2f}) "
              f"above>well_below={p_sum['dir_follow_rate(above>well_below)']:.2f} "
              f"above>tiny={p_sum['dir_follow_rate(above>tiny)']:.2f} "
              f"mean_abs_dev_vs_orig={p_sum['mean_abs_dev_vs_orig']:.4f} | "
              f"plumbing: fraction of rows whose forecast changed={plumbing['frac_rows_pred_changed_above_vs_well_below']:.2f}"
              f" (above vs well_below raw quantile tensor, mean of max|diff|="
              f"{plumbing['mean_abs_raw_diff_above_vs_well_below']:.4f})")
        print(f"{'domain':10} {'n':>4} | {'peak spearman':>13} | {'peak dir>tiny':>13} | "
              f"{'roi_mean spearman':>18} | {'roi_mean dir>tiny':>18}")
        print("-" * 90)
        for dom in ["ALL"] + sorted(set(domains.tolist())):
            pd_, md_ = (peak_summary, mean_summary) if dom == "ALL" else (peak_by_domain[dom], mean_by_domain[dom])
            print(f"{dom:10} {pd_['n']:>4} | {pd_['spearman_mean']:>13.3f} | "
                  f"{pd_['dir_follow_rate(above>tiny)']:>13.3f} | "
                  f"{md_['spearman_mean']:>18.3f} | {md_['dir_follow_rate(above>tiny)']:>18.3f}")

    with torch.no_grad():  # restore the gates to their checkpoint values (in case the model object is reused)
        for (_, p), orig_p in zip(_gate_params(model), gate_snapshot):
            p.copy_(orig_p)

    report.update({
        "model_path": os.path.abspath(model_path), "parquet": parquet,
        "n_samples": n, "seed": seed, "band_clauses": BAND_CLAUSES,
        "sampled_positions": idx.tolist(),
    })
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[fresh_forward] written -> {output}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_aggregate", action="store_true",
                    help="skip the gate=0 identity check (run only --fresh_forward)")
    ap.add_argument("--fresh_forward", action="store_true",
                    help="run the counterfactual forward pass (ChronosLLM.from_pretrained + teacher-forced); needs a GPU")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="checkpoint for --fresh_forward")
    ap.add_argument("--fresh_parquet", default=DEFAULT_FRESH_PARQUET)
    ap.add_argument("--fresh_n_samples", type=int, default=80)
    ap.add_argument("--fresh_seed", type=int, default=0)
    ap.add_argument("--fresh_batch_size", type=int, default=16)
    ap.add_argument("--gate_scales", default="1.0",
                    help="comma-separated multipliers applied to the cross-attention gates at inference "
                         "(1.0 = gates as trained); all scales run within one model load")
    ap.add_argument("--dump_examples", type=int, default=3,
                    help="store the rendered-text tail of the first N sampled rows for every band in the output JSON, for manual verification that the edit took effect")
    ap.add_argument("--fresh_output", default="outputs/eval/causal_faithfulness/fresh_forward_result.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_per_sample_raw", default=None,
                    help="optional npz path for the raw per-sample, per-band peak_ratio/mean_ratio matrices; "
                         "single --gate_scales value only")
    args = ap.parse_args()

    print("=" * 70)
    print("Counterfactual magnitude-edit report")
    print("=" * 70)

    if not args.skip_aggregate:
        gate_identity_report()

    if args.fresh_forward:
        print("\n" + "=" * 70)
        print("Counterfactual forward pass")
        print("=" * 70)
        gate_scales = [float(x) for x in args.gate_scales.split(",") if x.strip()]
        fresh_forward_report(
            args.model_path, args.fresh_parquet, n_samples=args.fresh_n_samples,
            seed=args.fresh_seed, batch_size=args.fresh_batch_size, device=args.device,
            gate_scales=gate_scales, dump_examples=args.dump_examples, output=args.fresh_output,
            save_per_sample_raw=args.save_per_sample_raw,
        )


if __name__ == "__main__":
    main()
