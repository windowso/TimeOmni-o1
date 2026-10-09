"""Unit tests for `summarize_agg6_by_domain.py` (agg6 = the MMTR understanding test pool): domain assignment +
metric scale + weighting.

Three things are guarded; getting any of them wrong makes the table silently draw the wrong conclusion:

1. **All three sources of domain assignment must be right** (fixed sub-tasks / the Time-RA scene / the VeriTime
   question-text regex); a wrong assignment counts samples toward the wrong discipline.
2. **The scale must be unified to 0-100.** `weighted_f1` returns a 0-1 fraction, while accuracy and the OpenTSLM
   official protocol are 0-100 -- multiply by 100 before mixing them into the same weighted average.
3. **Domain scores are count-weighted**, not a simple mean.
"""
import json

from chronos_llm.scripts.utils.summarize_agg6_by_domain import (
    cell_score, collect, domain_of, domain_summary,
)


def test_domain_of_three_sources():
    # (a) fixed sub-tasks
    assert domain_of({"task": "veritime_CTU"}) == "Energy"
    assert domain_of({"task": "veritime_RCW"}) == "Bioacoustics"
    assert domain_of({"task": "ecg_qa_cot"}) == "Physiology"
    assert domain_of({"task": "hitsr_l2"}) == "Cross_domain_Synthetic"
    # (b) the Time-RA scene field
    assert domain_of({"task": "time_ra_uni", "scene": "Server-YAHOO"}) == "Information_Technology"
    assert domain_of({"task": "time_ra_multi", "scene": "Ocean-Tropical Atmosphere"}) == "Meteorology"
    # (c) the VeriTime Scenario family: the domain is in the question text
    it = "This is a metric called X collected from Traffic and Transportation with length of 256:."
    assert domain_of({"task": "veritime_Anomaly_detection", "input_text": it}) == "Urbanism"
    # (d) fallback: the question lacks that sentence => Unlabeled (instead of silently counted toward some discipline)
    assert domain_of({"task": "veritime_Scenario_attribution", "input_text": "no domain here"}) == "Unlabeled"
    assert domain_of({"task": "time_ra_uni", "scene": "SomethingNew"}) == "Unlabeled"
    print("domain assignment from three sources + fallback OK")


def _ra_rows(n_norm, n_anom):
    """Build Time-RA rows: gt and pred all agree => perfect classification, F1 must be 100 (not 1.0)."""
    rows = [{"ground_truth": "Answer: normal sequence", "generated_text": "Answer: normal sequence"}
            for _ in range(n_norm)]
    rows += [{"ground_truth": "Answer: sudden spike", "generated_text": "Answer: sudden spike"}
             for _ in range(n_anom)]
    return rows


def test_time_ra_scaled_to_percent():
    scores = dict(cell_score("time_ra_uni", _ra_rows(6, 4)))
    assert set(scores) == {"Label-F1", "Action-F1"}
    for k, v in scores.items():
        assert abs(v - 100.0) < 1e-6, f"{k} should be 100 (0-100 scale), got {v} -- missing x100"
    print("Time-RA F1 on the 0-100 scale OK")


def test_accuracy_cell_same_scale():
    rows = [{"ground_truth": "Answer: A", "generated_text": "Answer: A"},
            {"ground_truth": "Answer: B", "generated_text": "Answer: C"}]
    (name, v), = cell_score("veritime_CTU", rows)
    assert name == "acc" and abs(v - 50.0) < 1e-9
    print("accuracy cell on the same scale OK")


def test_domain_summary_is_count_weighted():
    # 1000 rows at 90 + 10 rows at 0: weighted ~ 89.11, a simple mean would be 45
    s = domain_summary({("a", "acc"): (90.0, 1000), ("b", "acc"): (0.0, 10)})
    assert abs(s - 90.0 * 1000 / 1010) < 1e-9, f"should be count-weighted, got {s}"
    # nan cells enter neither numerator nor denominator
    s2 = domain_summary({("a", "acc"): (80.0, 5), ("b", "acc"): (float("nan"), 5)})
    assert abs(s2 - 80.0) < 1e-9
    print("domain summary count-weighted, nan does not pollute OK")


def test_collect_n_distinct_and_sources(tmp_path):
    """Run the real `collect()`: `n` must be the **distinct sample count**, and sources must show that one dataset
    spans several domains.

    One group of Time-RA samples yields **two** cells (Label-F1 and Action-F1); summing per cell would count it twice.
    """
    def row(task, scene=None, it=None, gt="normal sequence"):
        r = {"task": task, "ground_truth": f"Answer: {gt}", "generated_text": f"Answer: {gt}"}
        if scene:
            r["scene"] = scene
        if it:
            r["input_text"] = it
        return r

    d = tmp_path / "infer"
    d.mkdir()
    # Time-RA: 10 same-domain samples -> two metric cells
    (d / "time_ra_test.jsonl").write_text(
        "\n".join(json.dumps(row("time_ra_uni", scene="Server-YAHOO")) for _ in range(10)) + "\n")
    # VeriTime: samples of the same dataset land in two different disciplines (exactly why we report by domain)
    (d / "veritime_test.jsonl").write_text("\n".join([
        json.dumps(row("veritime_CTU")),                                   # -> Energy
        json.dumps(row("veritime_Anomaly_detection",
                       it="metric collected from Redis Database with length of 256:")),  # -> IT
    ]) + "\n")

    cells, n_samples, sources = collect(str(d))
    assert n_samples["Information_Technology"] == 11, \
        f"Time-RA 10 + VeriTime 1 = 11, got {n_samples['Information_Technology']} (double metric counted twice?)"
    assert len(cells["Information_Technology"]) == 3      # Label-F1 / Action-F1 / veritime acc
    assert n_samples["Energy"] == 1
    # The same upstream dataset appears under two disciplines
    assert sources["Energy"]["VeriTime (TSRBench)"] == 1
    assert sources["Information_Technology"]["VeriTime (TSRBench)"] == 1
    assert sources["Information_Technology"]["Time-RA (RATs40K)"] == 10
    print("collect(): n counts distinct samples + one dataset spanning several disciplines OK")
