"""CPU unit tests for the structured magnitude tags (make_structured_conclusion).

Covers: the taxonomy matches the 5-band wording of causal_faithfulness;
the tag band is consistent with the true ratio; all rows (including test) are processed; multi-channel /
invalid-typical rows are skipped safely and kept unchanged; determinism; B=1/C=1 compatibility (tagged
rows pass ForecastParquetDataset rendering + collator batching, and the LCP supervision span is unaffected
by the tag); the closing prompt instruction is rewritten in sync (it must ask for `[Magnitude: BAND]` and
list the five options, so the instruction does not drift from the supervised training format);
per-dataset_name exclusion; and the counterfactual edit of a tagged conclusion.
"""
import os
import tempfile

import numpy as np
import pandas as pd

from chronos_llm.scripts.utils.make_structured_conclusion import (
    BAND_EDGES, BAND_ORDER, BAND_TAG, _TASK_TAIL_NEW, _TASK_TAIL_OLD, _future_peak,
    add_structured_tags, band_of, filter_excluded_datasets, inject_prompt_instruction, make_tag,
)

LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")

_REAL_PROMPT_TAIL = (
    "## Task\nTreating the event as an established fact, reason forward from the history "
    "window and the event, then commit to a conclusion. Output your reasoning wrapped in "
    "<think> ... </think>, immediately followed by a single conclusion sentence:\n<think>\n"
    "2-4 sentence forward chain: (i) the ROI clock time lines up with the event time; "
    "(ii) event + history -> shape.\n</think>\n" + _TASK_TAIL_OLD
)


def _mk_df(seed=0, include_multichannel=True):
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ["CGTSF/MSPG", "fnf/traffic"]:
        for i, mult in enumerate([0.05, 0.25, 0.55, 1.0, 1.8, 1.0, 1.0, 1.0]):
            T, F = 64, 16
            hist = (np.abs(rng.normal(5, 2, T))).astype(np.float32)
            fut = (np.abs(rng.normal(5, 2, F)) * mult).astype(np.float32)
            rows.append({
                "id": f"{ds}/{i}", "dataset_name": ds, "split": "test" if i == 0 else "train",
                "history_values": hist.tolist(), "future_values": fut.tolist(),
                "background": "bg text", "event": "clear skies",
                "prompt": f"Infer the future.\n\n{_REAL_PROMPT_TAIL}",
                "plain_prompt": "Plain background.",
                "reasoning": "The ROI covers indices [64, 80) of the overall series.",
                "conclusion": "In the span [64, 80), a steady daily profile.",
                "past_len": T, "future_len": F, "roi_start_idx": 64, "roi_end_idx": 80,
            })
    if include_multichannel:
        # Multi-channel row (2D future/history): must be skipped safely. Note that 2D and 1D values cannot
        # coexist in one parquet column (pyarrow type-inference limitation) -- this row is only mixed into
        # the in-memory DataFrame tests (never written to parquet); tests that need to write parquet build
        # the frame separately with include_multichannel=False.
        rows.append({
            "id": "fnf/traffic/mc0", "dataset_name": "fnf/traffic", "split": "train",
            "history_values": np.zeros((2, 64), dtype=np.float32).tolist(),
            "future_values": np.ones((2, 16), dtype=np.float32).tolist(),
            "background": "bg", "event": "clear",
            "prompt": f"Infer.\n\n{_REAL_PROMPT_TAIL}", "plain_prompt": "p",
            "reasoning": "r", "conclusion": "multichannel row untouched.",
            "past_len": 64, "future_len": 16, "roi_start_idx": 64, "roi_end_idx": 80,
        })
    return pd.DataFrame(rows)


def test_taxonomy_matches_causal_faithfulness():
    from chronos_llm.eval.causal_faithfulness import BAND_CLAUSES, BANDS
    assert BAND_ORDER == BANDS, "band order drifted from causal_faithfulness"
    assert list(BAND_TAG.keys()) == BAND_ORDER, "tag key order drifted"
    assert set(BAND_TAG.keys()) == set(BAND_CLAUSES.keys()), "band set drifted from causal_faithfulness"
    print("taxonomy matches the 5-band convention of causal_faithfulness OK")


def test_band_of_boundaries():
    assert band_of(0.05) == "tiny"
    assert band_of(0.10) == "well_below"  # half-open intervals: == upper edge belongs to the next band
    assert band_of(0.39) == "well_below"
    assert band_of(0.70) == "typical"  # half-open intervals: == upper edge belongs to the next band
    assert band_of(0.69) == "mod_below"
    assert band_of(0.71) == "typical"
    assert band_of(1.29) == "typical"
    assert band_of(1.31) == "above"
    assert band_of(10.0) == "above"
    print("band_of boundaries OK")


def test_tags_match_true_ratio_and_cover_all_rows():
    df = _mk_df()
    out, stats = add_structured_tags(df)
    assert len(out) == len(df)
    # The typ_map convention must match add_structured_tags internally value by value (including the
    # multi-channel row contributing its first channel via _future_peak to the p90 pool).
    typ_map = (df["future_values"].map(_future_peak)
               .groupby(df["dataset_name"]).quantile(0.90).to_dict())
    n_1d = int((df["future_values"].map(lambda v: np.ndim(v[0]) if len(v) else 0) == 0).sum())
    assert stats["n_touch"] == n_1d, "all 1D rows should be tagged (no eligible-dataset restriction / no evidence gating)"
    assert stats["n_skip"] == len(df) - n_1d
    for pos in range(len(df)):
        row = df.iloc[pos]
        concl_new = out.iloc[pos]["conclusion"]
        v = row["future_values"]
        if np.ndim(v[0]) > 0:
            assert concl_new == row["conclusion"], "multi-channel row conclusion must not be modified"
            assert pd.isna(out.iloc[pos]["structtag_band"])
            continue
        peak = float(np.nanmax(np.asarray(v, dtype=float)))
        ratio = peak / typ_map[row["dataset_name"]]
        band = band_of(ratio)
        assert out.iloc[pos]["structtag_band"] == band
        assert concl_new.startswith(f"[Magnitude: {BAND_TAG[band]}] ")
        assert concl_new.endswith(row["conclusion"]), "the original conclusion text must be kept verbatim after the tag"
    print(f"tag band consistent with the true ratio + all 1D rows covered OK ({stats['n_touch']}/{stats['n_total']}, "
          f"band distribution {stats['by_band']})")


def test_deterministic_and_originals_unmutated():
    df = _mk_df()
    out1, _ = add_structured_tags(df)
    out2, _ = add_structured_tags(df)
    assert out1["conclusion"].tolist() == out2["conclusion"].tolist(), "non-deterministic: two runs differ"
    # add_structured_tags must not mutate the input df in place
    assert df["conclusion"].tolist() == [
        "In the span [64, 80), a steady daily profile."] * (len(df) - 1) + ["multichannel row untouched."]
    print("deterministic + input df not mutated in place OK")


def test_prompt_instruction_injected_for_all_rows():
    """The closing prompt sentence must be rewritten to the version that asks for [Magnitude: BAND] and lists
    the five bands, and this must apply to all rows (including untagged multi-channel rows -- the instruction
    describes the task format and does not depend on whether a given row was tagged)."""
    df = _mk_df(include_multichannel=False)  # 2D/1D cannot coexist in one parquet column
    out, _ = add_structured_tags(df)
    for p in out["prompt"]:
        assert _TASK_TAIL_OLD not in p, "old closing sentence was not replaced"
        assert _TASK_TAIL_NEW in p, "new closing sentence was not injected"
        for tag in ["TINY", "WELL_BELOW", "MOD_BELOW", "TYPICAL", "ABOVE"]:
            assert f"[Magnitude: {tag}]" in p, f"new instruction does not list band {tag}"
    print(f"closing prompt instruction rewritten in sync OK ({len(out)} rows all hit, all five options listed)")


def test_inject_prompt_instruction_idempotent_and_guards_drift():
    """Idempotent (an already-new prompt is not replaced again); upstream template drift (changed closing
    sentence) must raise instead of being skipped silently."""
    once = inject_prompt_instruction(f"x\n\n{_REAL_PROMPT_TAIL}")
    twice = inject_prompt_instruction(once)
    assert once == twice, "repeated injection must not change the result (idempotent)"
    try:
        inject_prompt_instruction("some prompt without the expected tail sentence")
        raised = False
    except ValueError:
        raised = True
    assert raised, "a non-matching closing sentence must raise, so upstream template drift is not swallowed silently"
    print("prompt instruction injection idempotent + template-drift guard OK")


def test_filter_excluded_datasets():
    df = _mk_df(include_multichannel=False)
    assert set(df["dataset_name"]) == {"CGTSF/MSPG", "fnf/traffic"}
    out = filter_excluded_datasets(df, ["fnf/traffic"])
    assert set(out["dataset_name"]) == {"CGTSF/MSPG"}
    assert len(out) == (df["dataset_name"] == "CGTSF/MSPG").sum()
    # An empty / None exclusion list returns the data unchanged (same frame, same behaviour)
    assert filter_excluded_datasets(df, []) .equals(df)
    assert filter_excluded_datasets(df, None).equals(df)
    print(f"per-dataset_name exclusion OK ({len(df)} -> {len(out)} rows)")


def test_dataset_render_b1_c1():
    """B=1/C=1 compatibility: tagged rows pass ForecastParquetDataset rendering + collator batching, the
    leading tag does not break the LCP supervision-span locator (labels non-empty, answer segment starts
    with the tag), and the new prompt instruction rendered into the context (unsupervised segment) does
    not break rendering."""
    import torch
    from transformers import AutoTokenizer

    from chronos_llm.data.collator import ChronosLLMCollator
    from chronos_llm.data.forecast_dataset import ForecastParquetDataset
    from chronos_llm.models.chronos_llm_model import _add_ts_special_tokens

    df = _mk_df(include_multichannel=False)  # 2D/1D cannot coexist in one parquet column
    out, stats = add_structured_tags(df)
    assert stats["n_touch"] > 0
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "structtag.parquet")
        out.drop(columns=["structtag_band"]).to_parquet(p)  # the provenance column is not fed to the model; mimic real training input
        ds = ForecastParquetDataset(p, tok, split="train", max_user_tokens=1500, max_tokens=4096)
        tagged_pos = [i for i in range(len(ds.df))
                      if str(ds.df.iloc[i]["conclusion"]).startswith("[Magnitude:")]
        assert tagged_pos, "no tagged rows in the rendered set"
        item = ds._build_item(tagged_pos[0])
        assert item is not None, "tagged row failed to render (too long / no supervised content)"
        sup_mask = np.asarray(item["labels"]) != -100
        assert sup_mask.sum() > 0, "supervised segment of the tagged row is empty"
        sup_ids = np.asarray(item["input_ids"])[sup_mask]
        sup_text = tok.decode(sup_ids, skip_special_tokens=False)
        assert "[Magnitude:" in sup_text, "tag did not land inside the supervised segment (LCP locator offset)"
        full_text = tok.decode(item["input_ids"], skip_special_tokens=False)
        assert "well below typical" in full_text, "new prompt instruction segment (band description) was not rendered into the context"
        batch = ChronosLLMCollator(tokenizer=tok)([item])  # B=1
        assert batch["context"].shape[0] == 1 and batch["future"].shape[0] == 1  # C=1
        assert torch.isfinite(batch["future"]).any()
    print("B=1/C=1 rendering + batching OK (tag inside the supervised segment, segment non-empty, new prompt instruction did not break rendering)")


def test_edit_conclusion_handles_structtag_prefix():
    """causal_faithfulness.edit_conclusion: when the conclusion carries a `[Magnitude: TAG]` structured prefix,
    the counterfactual edit must replace the prefix as well, otherwise the prefix (original band) contradicts the
    edited final sentence (target band)."""
    from chronos_llm.eval.causal_faithfulness import BAND_CLAUSES as _CLAUSES
    from chronos_llm.eval.causal_faithfulness import edit_conclusion as cf_edit

    tagged = "[Magnitude: WELL_BELOW] A calm midday span. Overall, the peak runs well below the series' typical highs."
    untagged = "A calm midday span. Overall, the peak runs well below the series' typical highs."
    target_clause = _CLAUSES["above"]

    for edit_fn in (cf_edit,):
        # With prefix + band given: prefix and final sentence both switch to the target band, consistently; the narrative sentence is kept.
        out = edit_fn(tagged, target_clause, band="above")
        assert out.startswith(f"[Magnitude: {BAND_TAG['above']}] "), f"{edit_fn}: prefix not switched to the target band"
        assert out.count("Overall,") == 1 and target_clause in out
        assert "WELL_BELOW" not in out, f"{edit_fn}: old-band prefix left over, counterfactual is impure"
        assert "A calm midday span." in out, f"{edit_fn}: narrative sentence was wrongly removed"
        # Without band: prefix is kept as is.
        out_compat = edit_fn(tagged, target_clause)
        assert out_compat.startswith("[Magnitude: WELL_BELOW] "), f"{edit_fn}: prefix must not be touched when band=None"
        # No prefix (untagged conclusion): only the final sentence changes.
        out_plain = edit_fn(untagged, target_clause, band="above")
        assert not out_plain.startswith("["), f"{edit_fn}: an untagged row must not grow a tag out of nowhere"
        assert out_plain == f"A calm midday span. Overall, {target_clause}."
    print("edit_conclusion structured-prefix handling OK")


def main():
    test_taxonomy_matches_causal_faithfulness()
    test_band_of_boundaries()
    test_tags_match_true_ratio_and_cover_all_rows()
    test_deterministic_and_originals_unmutated()
    test_prompt_instruction_injected_for_all_rows()
    test_inject_prompt_instruction_idempotent_and_guards_drift()
    test_filter_excluded_datasets()
    test_dataset_render_b1_c1()
    test_edit_conclusion_handles_structtag_prefix()
    print("ALL test_structured_conclusion OK")


if __name__ == "__main__":
    main()
