"""CPU unit tests for forecast_visual_faithfulness.py: full coverage of the pure functions
(plotting / message building / record loading & alignment / aggregation); the network parts
(call_visual_judge/run_visual_judge_batch) get a fake client injected and never touch the network."""
import json
import os
import tempfile

import numpy as np
import pandas as pd

from chronos_llm.eval.forecast_visual_faithfulness import (
    build_visual_judge_messages,
    call_visual_judge,
    load_records,
    render_forecast_png,
    run_visual_judge_batch,
    summarize,
)


def test_render_forecast_png_returns_valid_png():
    hist = np.linspace(1, 2, 50)
    hist_ts = pd.date_range("2024-01-01", periods=50, freq="h")
    future_ts = pd.date_range("2024-01-03", periods=20, freq="h")
    median = np.linspace(2, 3, 20)
    roi = np.zeros(20, dtype=bool)
    roi[5:10] = True
    png = render_forecast_png(hist, hist_ts, future_ts, median, median - 0.2, median + 0.2, roi,
                              title="t", hist_n=30)
    assert isinstance(png, bytes) and png[:8] == b"\x89PNG\r\n\x1a\n"
    # hist_n truncation works: an over-long history must neither raise nor be ignored (the plotting function crops to the most recent N points internally; here we only check that it does not raise)
    long_hist = np.arange(500.0)
    long_ts = pd.date_range("2020-01-01", periods=500, freq="h")
    png2 = render_forecast_png(long_hist, long_ts, future_ts, median, median - 0.2, median + 0.2, roi,
                               hist_n=30)
    assert png2[:8] == b"\x89PNG\r\n\x1a\n"
    print("render_forecast_png OK")


def test_build_visual_judge_messages_structure():
    median = np.array([1.234, 5.678, 9.0])
    png = b"\x89PNG\r\n\x1a\nFAKE"
    msgs = build_visual_judge_messages("some reasoning", median, 1, 3, png)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    text_block, image_block = msgs[1]["content"]
    assert text_block["type"] == "text" and "some reasoning" in text_block["text"]
    assert "1.234" in text_block["text"] and "5.678" in text_block["text"]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")
    print("build_visual_judge_messages OK")


def _make_synthetic_inputs(tmp_dir):
    ids = np.array(["fnf/load/1", "fnf/traffic/2", "unknownsrc/3"])
    dataset_names = np.array(["fnf/load", "fnf/traffic", "unknownsrc"])
    H = 10
    rng = np.random.default_rng(0)
    pred_quantiles = rng.random((3, 21, H)).astype(np.float32)
    gt = rng.random((3, H)).astype(np.float32)
    valid_mask = np.ones((3, H), dtype=np.float32)
    roi_mask = np.zeros((3, H), dtype=np.float32)
    roi_mask[:, 2:5] = 1
    quantile_levels = np.linspace(0.01, 0.99, 21)
    npz_path = os.path.join(tmp_dir, "forecast_preds.npz")
    np.savez(npz_path, pred_quantiles=pred_quantiles, gt=gt, roi_mask=roi_mask, valid_mask=valid_mask,
             quantile_levels=quantile_levels, ids=ids, dataset_names=dataset_names)

    jsonl_path = os.path.join(tmp_dir, "forecast_preds.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "fnf/load/1", "gen_text": "text A"}) + "\n")
        f.write(json.dumps({"id": "fnf/traffic/2", "gen_text": "text B"}) + "\n")
        f.write(json.dumps({"id": "unknownsrc/3", "gen_text": "text C"}) + "\n")
        # one id (missing/4) is deliberately left out -- load_records should skip it silently rather than raise

    # Mix in one split="train" row (its id overlaps neither npz nor jsonl, and its history is deliberately huge) --
    # load_records must exclude the whole row from the read via filters predicate pushdown, not read it into
    # memory and filter afterwards (see the comment inside load_records).
    df = pd.DataFrame({
        "id": ["fnf/load/1", "fnf/traffic/2", "unknownsrc/3", "missing/4", "trainrow/5"],
        "dataset_name": ["fnf/load", "fnf/traffic", "unknownsrc", "missing", "fnf/load"],
        "split": ["test", "test", "test", "test", "train"],
        "history_values": [list(np.arange(20.0) + i) for i in range(4)] + [list(np.arange(50000.0))],
        "history_timestamps": [list(pd.date_range("2024-01-01", periods=20, freq="h"))] * 4
                               + [list(pd.date_range("2024-01-01", periods=50000, freq="h"))],
        "future_timestamps": [list(pd.date_range("2024-01-02", periods=H, freq="h"))] * 5,
    })
    parquet_path = os.path.join(tmp_dir, "test.parquet")
    df.to_parquet(parquet_path)
    return npz_path, jsonl_path, parquet_path


def test_load_records_alignment_and_domain_mapping():
    with tempfile.TemporaryDirectory() as d:
        npz_path, jsonl_path, parquet_path = _make_synthetic_inputs(d)
        records = load_records(npz_path, jsonl_path, parquet_path)
    # all 3 jsonl rows are found in the parquet (missing/4 exists only in the parquet, not in npz/jsonl, and is not pulled in)
    assert len(records) == 3
    by_id = {r["id"]: r for r in records}
    assert by_id["fnf/load/1"]["domain"] == "Load"
    assert by_id["fnf/traffic/2"]["domain"] == "Traffic"
    assert by_id["unknownsrc/3"]["domain"] is None  # not in the DOMAIN_SOURCES list; the caller puts it in the Unknown bucket
    r = by_id["fnf/load/1"]
    assert r["gen_text"] == "text A"
    assert r["median"].shape == (10,)
    assert r["roi_start"] == 2 and r["roi_end"] == 5
    print("load_records alignment + domain mapping OK")


def test_summarize():
    s = summarize([{"score": 5}, {"score": 3}, {"score": None}])
    assert s["n"] == 3 and s["n_scored"] == 2
    assert abs(s["mean"] - 4.0) < 1e-9
    assert abs(s["judge_fail_rate"] - 1 / 3) < 1e-9
    assert s["distribution"][5] == 1 and s["distribution"][3] == 1
    empty = summarize([])
    assert empty["n"] == 0 and np.isnan(empty["mean"])
    print("summarize OK")


# ---------------------------------------------------------------------------
# Fake client: verifies the call + parsing + orchestration logic of call_visual_judge / run_visual_judge_batch without touching the network.
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
        self._content_fn = content_fn
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


def test_call_visual_judge_with_fake_client():
    fake = _FakeClient(lambda model, messages: '{"score": 4, "reason": "consistent trend"}')
    median = np.array([1.0, 2.0, 3.0])
    png = b"\x89PNG\r\n\x1a\nFAKE"
    score, reason, _raw = call_visual_judge("rises steadily", median, 0, 2, png, fake, model="fake-model")
    assert score == 4 and reason == "consistent trend"
    assert fake.chat.completions.calls[0][0] == "fake-model"
    # confirm the image really was put into the outgoing message (not text only)
    sent_messages = fake.chat.completions.calls[0][1]
    assert any(block.get("type") == "image_url" for block in sent_messages[1]["content"])
    print("call_visual_judge(fake client) OK")


def test_run_visual_judge_batch_checkpoint_resume_and_errors():
    with tempfile.TemporaryDirectory() as d:
        npz_path, jsonl_path, parquet_path = _make_synthetic_inputs(d)
        records = load_records(npz_path, jsonl_path, parquet_path)

        call_count = {"n": 0}

        def _content_fn(model, messages):
            call_count["n"] += 1
            if "text C" in messages[1]["content"][0]["text"]:
                raise RuntimeError("simulated API failure")
            return '{"score": 5, "reason": "ok"}'

        fake = _FakeClient(_content_fn)
        ckpt = os.path.join(d, "ckpt.jsonl")
        r1 = run_visual_judge_batch(records, fake, model="fake", max_workers=2,
                                    max_attempts=1, base_delay=0.0, checkpoint_path=ckpt)
        assert set(r1) == {"fnf/load/1", "fnf/traffic/2", "unknownsrc/3"}
        assert r1["fnf/load/1"]["score"] == 5 and r1["fnf/load/1"]["error"] is None
        assert r1["unknownsrc/3"]["score"] is None and r1["unknownsrc/3"]["error"]  # a single failure does not take down the batch
        assert call_count["n"] == 3

        # resume from checkpoint: the three finished rows must be read back from the checkpoint without calling the API again
        r2 = run_visual_judge_batch(records, fake, model="fake", max_workers=2,
                                    max_attempts=1, base_delay=0.0, checkpoint_path=ckpt)
        assert set(r2) == set(r1)
        assert call_count["n"] == 3  # no new calls
    print("run_visual_judge_batch checkpoint resume + error handling OK")


def main():
    test_render_forecast_png_returns_valid_png()
    test_build_visual_judge_messages_structure()
    test_load_records_alignment_and_domain_mapping()
    test_summarize()
    test_call_visual_judge_with_fake_client()
    test_run_visual_judge_batch_checkpoint_resume_and_errors()
    print("ALL FORECAST VISUAL FAITHFULNESS TESTS PASSED")


if __name__ == "__main__":
    main()
