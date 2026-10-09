"""Unit tests for text_alignment_eval (pure text parsing + coordinate conversion + aggregation
arithmetic, CPU; the LLM-call part uses an injected fake client, so no network access and no API
budget)."""
import json
import os
import tempfile

import numpy as np
import pandas as pd

from chronos_llm.eval.text_alignment_eval import (
    absolute_gt_roi,
    build_records,
    call_shape_judge,
    compute_roi_alignment,
    dataset_name_of,
    domain_of,
    group_records_by_domain,
    load_predictions,
    load_roi_reference,
    parse_judge_response,
    roi_alignment_stats,
    roi_overlap,
    run_shape_judge_batch,
    summarize_roi,
    summarize_shape,
)
from chronos_llm.rl.reward import roi_iou
from chronos_llm.scripts.utils.paper_forecast_by_domain import DOMAIN_SOURCES


def test_absolute_gt_roi():
    assert absolute_gt_roi((1, 3), past_len=20) == (21, 23)
    assert absolute_gt_roi([28, 66], past_len=480) == (508, 546)
    print("absolute_gt_roi OK")


def test_roi_overlap():
    assert roi_overlap((0, 10), (5, 15)) is True           # partial overlap
    assert roi_overlap((0, 10), (10, 20)) is False          # half-open intervals merely touching: no overlap
    assert roi_overlap((0, 10), (100, 110)) is False        # fully disjoint
    assert roi_overlap((0, 100), (40, 60)) is True           # fully contained
    assert roi_overlap(None, (0, 10)) is False               # extraction failed
    print("roi_overlap OK")


def test_roi_alignment_stats():
    gt = (508, 546)
    # bracket format hit (same phrasing as the real data)
    s1 = roi_alignment_stats("The ROI [510, 552) covers the active period.", gt)
    assert s1["extracted_ok"] and s1["extracted"] == (510, 552)
    assert s1["overlap_hit"] is True
    assert abs(s1["iou"] - roi_iou((510, 552), gt)) < 1e-9

    # "rows X through Y" format, exact hit
    s2 = roi_alignment_stats("The ROI spans rows 508 through 545, matching exactly.", gt)
    assert s2["extracted"] == (508, 546) and s2["iou"] == 1.0

    # extraction succeeds but no overlap at all
    s3 = roi_alignment_stats("The ROI [600, 650) is where it matters.", gt)
    assert s3["extracted_ok"] and s3["overlap_hit"] is False and s3["iou"] == 0.0

    # parse failure: not removed from the denominator, overlap_hit/iou recorded as False/0
    s4 = roi_alignment_stats("No numeric range mentioned at all here.", gt)
    assert s4["extracted"] is None and not s4["extracted_ok"]
    assert s4["overlap_hit"] is False and s4["iou"] == 0.0
    print("roi_alignment_stats OK")


def test_summarize_roi():
    items = [
        {"extracted_ok": True, "overlap_hit": True, "iou": 0.8},
        {"extracted_ok": True, "overlap_hit": False, "iou": 0.0},
        {"extracted_ok": False, "overlap_hit": False, "iou": 0.0},   # extraction failed
        {"extracted_ok": True, "overlap_hit": True, "iou": 1.0},
    ]
    s = summarize_roi(items)
    assert s["n"] == 4
    assert abs(s["overlap_hit_rate"] - 2 / 4) < 1e-9            # 2 hits / 4 total (failures stay in the denominator)
    assert abs(s["iou_mean"] - (0.8 + 0.0 + 0.0 + 1.0) / 4) < 1e-9
    assert abs(s["extract_fail_rate"] - 1 / 4) < 1e-9
    assert abs(s["overlap_hit_rate_given_extracted"] - 2 / 3) < 1e-9   # 2 hits among the 3 successful extractions
    empty = summarize_roi([])
    assert empty["n"] == 0 and np.isnan(empty["overlap_hit_rate"])
    print("summarize_roi OK")


def test_dataset_name_of_and_domain_of():
    assert dataset_name_of("CGTSF/MSPG/0") == "CGTSF/MSPG"
    assert dataset_name_of("fnf/bitcoin/17") == "fnf/bitcoin"
    assert domain_of("CGTSF/MSPG") == "Solar"
    assert domain_of("fnf/bitcoin") == "Finance"
    assert domain_of("timemmd/Climate") == "Climate"
    assert domain_of("nonexistent/source") is None
    print("dataset_name_of/domain_of OK")


def test_domain_of_covers_all_known_sources():
    # Check entry by entry against paper_forecast_by_domain.DOMAIN_SOURCES, to make sure this module
    # reuses that very mapping rather than a re-typed copy that could drift.
    total_sources = 0
    for dom, srcs in DOMAIN_SOURCES.items():
        for src in srcs:
            assert domain_of(src) == dom, (src, dom)
            total_sources += 1
    assert total_sources == 10, total_sources   # 10 sources -> 5 forecasting domains
    print("domain_of covers all DOMAIN_SOURCES OK")


def test_group_records_by_domain():
    records = [
        {"id": "a", "domain": "Solar"},
        {"id": "b", "domain": "Solar"},
        {"id": "c", "domain": "Finance"},
        {"id": "d", "domain": None},   # unknown source -> Unknown bucket
    ]
    groups = group_records_by_domain(records)
    assert set(groups) == {"Solar", "Finance", "Unknown"}
    assert len(groups["Solar"]) == 2 and len(groups["Finance"]) == 1 and len(groups["Unknown"]) == 1
    print("group_records_by_domain OK")


def _toy_parquet(path):
    rows = [
        {"id": "CGTSF/MSPG/0", "dataset_name": "CGTSF/MSPG", "split": "test",
         "past_len": 480, "roi_start_idx": 508, "roi_end_idx": 546},
        {"id": "fnf/bitcoin/3", "dataset_name": "fnf/bitcoin", "split": "test",
         "past_len": 100, "roi_start_idx": 120, "roi_end_idx": 130},
        {"id": "unknownsrc/9", "dataset_name": "unknownsrc", "split": "test",
         "past_len": 50, "roi_start_idx": 60, "roi_end_idx": 70},
        {"id": "train/row", "dataset_name": "CGTSF/MSPG", "split": "train",   # should be filtered out by split
         "past_len": 10, "roi_start_idx": 12, "roi_end_idx": 14},
    ]
    pd.DataFrame(rows).to_parquet(path)


def _toy_jsonl(path):
    rows = [
        {"id": "CGTSF/MSPG/0", "gen_text": "The ROI [510, 552) shows a rise then fall.",
         "gt_conclusion": "The span [508, 546) rises then falls.", "roi": [28, 66]},
        {"id": "fnf/bitcoin/3", "gen_text": "No numeric range given here.",
         "gt_conclusion": "The span [120, 130) stays flat.", "roi": [20, 30]},
        {"id": "unknownsrc/9", "gen_text": "The ROI [60, 70) is flat.",
         "gt_conclusion": "The span [60, 70) stays flat.", "roi": [10, 20]},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_build_records_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        pq_path = os.path.join(d, "toy.parquet")
        jsonl_path = os.path.join(d, "toy.jsonl")
        _toy_parquet(pq_path)
        _toy_jsonl(jsonl_path)

        ref_df = load_roi_reference(pq_path)
        assert len(ref_df) == 3   # the train row is filtered out by split

        pred_rows = load_predictions(jsonl_path)
        assert len(pred_rows) == 3

        records = build_records(pred_rows, ref_df)
        assert len(records) == 3
        by_id = {r["id"]: r for r in records}
        assert by_id["CGTSF/MSPG/0"]["gt_roi_abs"] == (508, 546)
        assert by_id["CGTSF/MSPG/0"]["domain"] == "Solar"
        assert by_id["fnf/bitcoin/3"]["gt_roi_abs"] == (120, 130)
        assert by_id["fnf/bitcoin/3"]["domain"] == "Finance"
        assert by_id["unknownsrc/9"]["domain"] is None   # not in DOMAIN_SOURCES

        per_sample = compute_roi_alignment(records)
        stats = {s["id"]: s for s in per_sample}
        assert stats["CGTSF/MSPG/0"]["overlap_hit"] is True     # [510,552) overlaps [508,546)
        assert stats["fnf/bitcoin/3"]["extracted_ok"] is False   # no numeric interval written
        assert stats["unknownsrc/9"]["iou"] == 1.0                # exact hit
    print("build_records end-to-end OK")


def test_build_records_rejects_inconsistent_roi():
    # Data-hygiene guard: when jsonl.roi (relative) + past_len disagrees with the parquet's absolute
    # roi it must raise, not silently evaluate against the wrong ground truth (protects against
    # silent contamination from coordinate-convention drift / mismatched file versions).
    with tempfile.TemporaryDirectory() as d:
        pq_path = os.path.join(d, "toy.parquet")
        _toy_parquet(pq_path)
        ref_df = load_roi_reference(pq_path)
        bad_row = [{"id": "CGTSF/MSPG/0", "gen_text": "x", "gt_conclusion": "y",
                    "roi": [0, 1]}]   # disagrees with the parquet's roi_start/end_idx=(508,546)
        raised = False
        try:
            build_records(bad_row, ref_df)
        except AssertionError:
            raised = True
        assert raised, "inconsistent coordinates should have raised AssertionError"
    print("build_records rejects inconsistent roi OK")


def test_parse_judge_response():
    assert parse_judge_response('{"score": 4, "reason": "similar shape"}') == (4, "similar shape")
    # must also parse when wrapped in markdown code fences
    wrapped = '```json\n{"score": 2, "reason": "different turning points"}\n```'
    assert parse_judge_response(wrapped) == (2, "different turning points")
    # out-of-range score: JSON parses but is outside 1-5 -> falls back to the regex; the raw text has
    # no bare 1-5 digit -> None
    assert parse_judge_response('{"score": 7, "reason": "bad"}')[0] is None
    # not JSON but contains a bare digit
    score, _reason = parse_judge_response("I'd rate this a 3 out of 5 honestly.")
    assert score == 3
    # empty / None
    assert parse_judge_response("")[0] is None
    assert parse_judge_response(None)[0] is None
    print("parse_judge_response OK")


def test_summarize_shape():
    items = [{"score": 5}, {"score": 3}, {"score": None}, {"score": 4}, {"score": 4}]
    s = summarize_shape(items)
    assert s["n"] == 5 and s["n_scored"] == 4
    assert abs(s["judge_fail_rate"] - 1 / 5) < 1e-9
    assert abs(s["mean"] - (5 + 3 + 4 + 4) / 4) < 1e-9
    assert s["distribution"][4] == 2 and s["distribution"][5] == 1 and s["distribution"][1] == 0
    empty = summarize_shape([])
    assert empty["n"] == 0 and np.isnan(empty["judge_fail_rate"])
    print("summarize_shape OK")


# ---------------------------------------------------------------------------
# Fake client: checks the call + parse + orchestration logic of call_shape_judge /
# run_shape_judge_batch without ever touching the network (no real HTTP dependency; only duck-types
# the openai client's .chat.completions.create).
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content_fn):
        self._content_fn = content_fn   # callable(model, messages) -> content str (or raises)
        self.calls = []

    def create(self, model=None, messages=None, **kwargs):
        self.calls.append((model, messages))
        return _FakeResponse(self._content_fn(model, messages))


class _FakeChat:
    def __init__(self, content_fn):
        self.completions = _FakeCompletions(content_fn)


class _FakeClient:
    def __init__(self, content_fn):
        self.chat = _FakeChat(content_fn)


def test_call_shape_judge_with_fake_client():
    fake = _FakeClient(lambda model, messages: '{"score": 4, "reason": "similar rising shape"}')
    score, reason, _raw = call_shape_judge(
        "rises then falls", "rises then falls", fake, model="fake-model")
    assert score == 4 and reason == "similar rising shape"
    assert fake.chat.completions.calls[0][0] == "fake-model"
    print("call_shape_judge(fake client) OK")


def test_run_shape_judge_batch_checkpoint_resume():
    call_count = {"n": 0}

    def _content_fn(model, messages):
        call_count["n"] += 1
        return '{"score": 5, "reason": "ok"}'

    fake = _FakeClient(_content_fn)
    records = [
        {"id": "a", "gen_text": "g1", "gt_conclusion": "t1"},
        {"id": "b", "gen_text": "g2", "gt_conclusion": "t2"},
    ]
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "ckpt.jsonl")
        r1 = run_shape_judge_batch(records, fake, model="fake", max_workers=2, checkpoint_path=ckpt)
        assert set(r1) == {"a", "b"}
        assert all(r1[k]["score"] == 5 for k in r1)
        assert call_count["n"] == 2

        # Rerun with one extra sample -- the finished a/b should be read back from the checkpoint
        # without calling the API again (resumability)
        records2 = records + [{"id": "c", "gen_text": "g3", "gt_conclusion": "t3"}]
        r2 = run_shape_judge_batch(records2, fake, model="fake", max_workers=2, checkpoint_path=ckpt)
        assert set(r2) == {"a", "b", "c"}
        assert call_count["n"] == 3   # only 1 new call (c)
    print("run_shape_judge_batch checkpoint resume OK")


def test_run_shape_judge_batch_handles_errors():
    def _content_fn(model, messages):
        if "boom" in messages[1]["content"]:
            raise RuntimeError("simulated API failure")
        return '{"score": 2, "reason": "ok"}'

    fake = _FakeClient(_content_fn)
    records = [
        {"id": "ok1", "gen_text": "fine", "gt_conclusion": "fine ref"},
        {"id": "bad1", "gen_text": "boom", "gt_conclusion": "irrelevant"},
    ]
    results = run_shape_judge_batch(records, fake, model="fake", max_workers=2,
                                     max_attempts=1, base_delay=0.0)
    assert results["ok1"]["score"] == 2 and results["ok1"]["error"] is None
    assert results["bad1"]["score"] is None and results["bad1"]["error"]   # one failure does not sink the batch
    print("run_shape_judge_batch handles per-sample errors OK")


def main():
    test_absolute_gt_roi()
    test_roi_overlap()
    test_roi_alignment_stats()
    test_summarize_roi()
    test_dataset_name_of_and_domain_of()
    test_domain_of_covers_all_known_sources()
    test_group_records_by_domain()
    test_build_records_end_to_end()
    test_build_records_rejects_inconsistent_roi()
    test_parse_judge_response()
    test_summarize_shape()
    test_call_shape_judge_with_fake_client()
    test_run_shape_judge_batch_checkpoint_resume()
    test_run_shape_judge_batch_handles_errors()
    print("ALL TEXT ALIGNMENT EVAL TESTS PASSED")


if __name__ == "__main__":
    main()
