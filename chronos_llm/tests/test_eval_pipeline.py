"""CPU integration smoke test of the evaluation pipeline: a tiny model runs the full infer->eval
on synthetic parquet/jsonl (device=cpu).

Verifies the real wiring of the infer scripts (generate -> npz packing / horizon slicing / meta
flow / quantile selection) and the eval outputs, without depending on a real checkpoint
(the from_pretrained loading itself is covered by test_pretrained_peft).
"""
import csv
import json
import os
import tempfile

import numpy as np
import pandas as pd

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.eval import eval_forecast, eval_understanding
from chronos_llm.eval.infer_forecast import run_forecast_infer
from chronos_llm.eval.infer_understanding import run_understanding_infer
from chronos_llm.tests.test_pretrained_peft import _build_tiny_base


def _forecast_parquet(path, n=4, pl=20, fl=6):
    rows = []
    for i in range(n):
        rows.append({
            "id": f"s{i}", "dataset_name": "Toy", "split": "test",
            "history_values": list(np.random.randn(pl).astype(np.float32)),
            "future_values": list((np.random.randn(fl) * 5 + 20).astype(np.float32)),
            "background": "bg", "event": "spike", "prompt": "forecast it",
            "reasoning": "trend up", "conclusion": "rises",
            "past_len": pl, "roi_start_idx": pl + 1, "roi_end_idx": pl + 3,  # roi -> future[1:3]
        })
    pd.DataFrame(rows).to_parquet(path)


def _understanding_jsonl(path, n=4):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": f"u{i}", "uid": f"uid{i}", "dataset_name": "Toy",
                "task": "Classification", "scene": "toy",
                "input_text": ["classify this series please"],
                "gt_text": ["up"], "gt_result": {"gt_class": {"default": ["up", "down"]}},
            }) + "\n")


def test_forecast_pipeline():
    model, tok = _build_tiny_base(); model.eval()
    with tempfile.TemporaryDirectory() as d:
        pq = os.path.join(d, "toy.parquet"); _forecast_parquet(pq)
        ds = ForecastParquetDataset(pq, tok, split="test", inference_mode=True,
                                    max_user_tokens=32, max_tokens=96)
        npz = os.path.join(d, "preds.npz")
        run_forecast_infer(model, tok, ds, npz, batch_size=2, max_new_tokens=3,
                           device="cpu", num_workers=0)
        z = np.load(npz, allow_pickle=True)
        assert z["pred_quantiles"].shape[0] == 4 and z["pred_quantiles"].shape[1] == 21
        assert z["gt"].shape == z["valid_mask"].shape and z["valid_mask"].any()
        # Text goes to the companion jsonl: the npz only holds numerics/ids, gen_text is not in it
        assert "gen_text" not in z.files
        # The npz carries the real id + dataset_name (per-dataset analysis groups by the labels inside the npz, not by positional alignment)
        assert [str(x) for x in z["ids"]] == ["s0", "s1", "s2", "s3"]
        assert [str(x) for x in z["dataset_names"]] == ["Toy"] * 4
        # Companion human-readable jsonl: one line per sample, all fields present, arrays trimmed to the valid horizon, roi converted to an interval
        jrows = [json.loads(x) for x in open(npz[:-4] + ".jsonl", encoding="utf-8")]
        assert len(jrows) == 4
        r0 = jrows[0]
        assert set(r0) >= {"id", "prompt", "gen_text", "gt_reasoning", "gt_conclusion",
                           "median_pred", "gt_series", "roi"}
        h0 = int(z["valid_mask"][0].sum())
        assert len(r0["median_pred"]) == len(r0["gt_series"]) == h0
        assert r0["roi"] == [1, 3]          # roi_start/end(pl+1,pl+3) - past_len -> future[1:3]
        assert r0["prompt"]                 # inference_mode passes the user text through (background/event/prompt)
        assert r0["gt_reasoning"] == "trend up" and r0["gt_conclusion"] == "rises"  # text gt passed through
        csvp = os.path.join(d, "m.csv")
        eval_forecast.main(["--pred", npz, "--output_csv", csvp])
        rows = list(csv.DictReader(open(csvp)))
        regions = {r["region"] for r in rows}
        assert regions == {"full", "roi"}, regions
        for r in rows:                              # all three metrics computed (finite values)
            for k in ("MAPE", "PCC", "CRPS"):
                assert r[k] not in ("", "nan"), (r["region"], k, r[k])
    print("forecast infer->eval pipeline (MAPE/PCC/CRPS on full region + ROI) OK")


def test_forecast_teacher_forced_pipeline():
    """Smoke test of the exposure-bias control path: training rendering (true reasoning/conclusion
    teacher-forced) -> hidden states fed back into chronos to produce the forecast -> npz -> eval.
    Also guards the id/dataset_name records of the tf npz: tf uses the training rendering
    (inference_mode=False) and must explicitly pass emit_meta=True to carry the real id/dataset_name,
    otherwise infer falls back to row numbers and the tf npz could not be split by dataset)."""
    model, tok = _build_tiny_base(); model.eval()
    with tempfile.TemporaryDirectory() as d:
        pq = os.path.join(d, "toy.parquet"); _forecast_parquet(pq)
        # Compatibility guard: by default (training path, emit_meta not passed) the training rendering produces no meta => zero impact on training batches.
        assert "meta" not in ForecastParquetDataset(
            pq, tok, split="test", inference_mode=False, max_user_tokens=32, max_tokens=96)[0]
        ds = ForecastParquetDataset(pq, tok, split="test", inference_mode=False,
                                    max_user_tokens=32, max_tokens=96, emit_meta=True)
        npz = os.path.join(d, "preds_tf.npz")
        run_forecast_infer(model, tok, ds, npz, batch_size=2, device="cpu", num_workers=0,
                           teacher_forced=True)
        z = np.load(npz, allow_pickle=True)
        assert z["pred_quantiles"].shape[0] == 4 and z["pred_quantiles"].shape[1] == 21
        assert np.isfinite(z["pred_quantiles"][z["valid_mask"][:, None, :].repeat(21, 1)]).all()
        assert "gen_text" not in z.files
        # The tf npz carries the real id + dataset_name (not row numbers), so per-dataset analysis can group directly
        assert [str(x) for x in z["ids"]] == ["s0", "s1", "s2", "s3"]
        assert [str(x) for x in z["dataset_names"]] == ["Toy"] * 4
        # Companion jsonl: emit_meta=True => prompt/gt_* populated; teacher-forced has no generation => gen_text is the empty string
        jrows = [json.loads(x) for x in open(npz[:-4] + ".jsonl", encoding="utf-8")]
        assert len(jrows) == 4 and all(r["gen_text"] == "" for r in jrows)
        assert all(r["prompt"] and r["gt_reasoning"] == "trend up"
                   and r["gt_conclusion"] == "rises" for r in jrows)
        assert all(len(r["median_pred"]) == len(r["gt_series"]) for r in jrows)
        eval_forecast.main(["--pred", npz])  # enough that the metric computation runs (same eval as the self-generated path)
    print("forecast teacher-forced control pipeline (training rendering -> feedback -> eval, tf npz carries real id/dataset) OK")


def test_understanding_pipeline():
    model, tok = _build_tiny_base(); model.eval()
    with tempfile.TemporaryDirectory() as d:
        jl = os.path.join(d, "toy.jsonl"); _understanding_jsonl(jl)
        infer_dir = os.path.join(d, "infer")
        outs = run_understanding_infer(model, tok, [jl], infer_dir, batch_size=2,
                                       max_new_tokens=3, device="cpu", num_workers=0,
                                       max_user_tokens=32, max_tokens=96)
        recs = [json.loads(x) for x in open(outs[0])]
        assert len(recs) == 4 and all("generated_text" in r and "gt_result" in r for r in recs)
        csvp = os.path.join(d, "u.csv")
        eval_understanding.evaluate_all_files(infer_dir, csvp)
        rows = list(csv.DictReader(open(csvp)))
        assert any(r["filename"] == "toy.jsonl" for r in rows)      # classification eval ran and produced a row for this file
        assert any("accuracy" in r and r["accuracy"] not in ("", None) for r in rows)
    print("understanding infer->eval pipeline (classification metrics) OK")


if __name__ == "__main__":
    test_forecast_pipeline()
    test_forecast_teacher_forced_pipeline()
    test_understanding_pipeline()
    print("ALL OK")
