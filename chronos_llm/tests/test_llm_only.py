"""End-to-end CPU test of the LLM-only model (--llm_only + ts_as_text full-precision
textualisation of the series).

Covers:
1. ts_text rendering/parsing units: float32 full-precision round trip, pre-cut ellipsis marker,
   multi-channel, C=1 compatibility, forecast value-array parsing (exact / extra values truncated /
   too few rejected / garbage rejected).
2. Understanding dataset ts_as_text: values appear in the rendered text, history placeholder
   (1,16), token_lengths IO-free shortcut for long series == real rendered value (capped),
   meta.input_text carries no series block.
3. Forecast dataset ts_as_text: user contains history block + output instruction, supervision
   contains conclusion + Predicted values, missing-timestamp path does not crash.
4. Model llm_only (tiny Qwen2 + real tokenizer, no chronos): understanding/forecast forward
   losses finite, gradients flow only to the LLM; after add_lora only LoRA is trainable;
   generate_understanding produces text; generate_forecast produces (B, 21, H) quantiles (the tiny
   random model fails to parse => all-NaN is legal); teacher-forced is explicitly rejected.
5. run_forecast_infer llm_only path: quantile_levels falls back to CHRONOS2_QUANTILE_LEVELS, npz
   is written successfully.
"""

import json
import os
import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.ts_text import (
    CHRONOS2_QUANTILE_LEVELS,
    forecast_values_block,
    format_value,
    parse_forecast_values,
    render_forecast_history,
    render_series_text,
)
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.models.chronos_llm_model import (
    ChronosLLM, ChronosLLMConfig, _add_ts_special_tokens, add_lora,
)

LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")


def test_ts_text_unit():
    # float32 full-precision round trip: parsing back gives bit-identical float32.
    vals = np.array([0.1, -3.14159274, 1e-8, 12345.678, 0.6484375, 7.0], dtype=np.float32)
    for v in vals:
        s = format_value(v)
        assert np.float32(float(s)) == v, f"{v!r} -> {s!r} does not round-trip"
    assert format_value(np.nan) == "nan"

    # Pre-cut ellipsis marker: keep points at head and tail, state the omitted count.
    arr = np.arange(100, dtype=np.float32)
    txt = render_series_text(arr, max_points_per_side=10)
    assert "(80 values omitted)" in txt and txt.count(",") >= 20
    assert format_value(np.float32(0)) in txt and format_value(np.float32(99)) in txt
    # No pre-cut: every point present.
    txt_full = render_series_text(arr, max_points_per_side=0)
    assert "omitted" not in txt_full and format_value(np.float32(57)) in txt_full
    # C=1 and multi-channel.
    assert "Channel" not in render_series_text(arr[None, :])
    txt_mc = render_series_text(np.stack([arr, arr + 1000]), max_points_per_side=5)
    assert "Channel 1:" in txt_mc and "Channel 2:" in txt_mc and "2 channels" in txt_mc

    # Forecast history block: with / without timestamps.
    h = np.array([1.5, 2.5], dtype=np.float32)
    ht = render_forecast_history(h, ["2021-01-01", "2021-01-02"], "D")
    assert "2021-01-01: 1.5" in ht and "frequency=D" in ht
    assert "1.5" in render_forecast_history(h, None, "")

    # Value-array parsing (same protocol as baseline_timereasoner: >=horizon truncated, <horizon rejected).
    blk = forecast_values_block(np.array([1.25, -2.5, 3e-4], dtype=np.float32))
    got = parse_forecast_values(f"reasoning...\n{blk}", 3)
    assert got is not None and np.allclose(got, [1.25, -2.5, 3e-4])
    assert parse_forecast_values("Predicted values: [1, 2, 3, 4, 5]", 3).tolist() == [1, 2, 3]
    assert parse_forecast_values("Predicted values: [1, 2]", 3) is None
    assert parse_forecast_values("no numbers here", 3) is None
    assert parse_forecast_values("", 3) is None
    # Take the last array (intermediate arrays inside the reasoning do not interfere).
    two = "draft [9, 9, 9] then final answer: [1, 2, 3]"
    assert parse_forecast_values(two, 3).tolist() == [1, 2, 3]
    print("ts_text units OK")


def _tiny_llm(tok):
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
                      max_position_embeddings=8192)
    llm = Qwen2ForCausalLM(cfg)
    llm.config.use_cache = False
    return llm


def _write_understanding_jsonl(d, n_short=2, long_len=9000):
    """One short series (really rendered) + one long series (token_lengths shortcut hits the cap)
    + one multi-channel sample."""
    rows, paths = [], []
    rng = np.random.default_rng(0)
    for i in range(n_short):
        p = os.path.join(d, f"short{i}.npy")
        np.save(p, rng.normal(size=37).astype(np.float32))
        rows.append({"id": f"s{i}", "dataset_name": "toy", "task": "cls",
                     "input_text": [f"Question {i}: is it rising? Answer yes or no."],
                     "gt_text": ["yes"], "think": "Look at the slope; it increases.",
                     "input_ts": {"original": {"ori_path": p, "ori_length": 37}, "channel": 1}})
        paths.append(p)
    plong = os.path.join(d, "long.npy")
    np.save(plong, rng.normal(size=long_len).astype(np.float32))
    rows.append({"id": "L", "dataset_name": "toy", "task": "cls",
                 "input_text": ["Long question."], "gt_text": ["no"],
                 "think": "Long series reasoning.",
                 "input_ts": {"original": {"ori_path": plong, "ori_length": long_len}, "channel": 1}})
    pmc = os.path.join(d, "mc.npy")
    np.save(pmc, rng.normal(size=(25, 3)).astype(np.float32))  # (T, C) -> (C=3, T=25)
    rows.append({"id": "M", "dataset_name": "toy", "task": "cls",
                 "input_text": ["Multichannel question."], "gt_text": ["yes"],
                 "think": "Channels agree.",
                 "input_ts": {"original": {"ori_path": pmc, "ori_length": 25}, "channel": 3}})
    jp = os.path.join(d, "u.jsonl")
    with open(jp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return jp


def _write_forecast_parquet(d, with_ts=True):
    import pandas as pd

    rng = np.random.default_rng(1)
    rows = []
    for i in range(3):
        pl, fl = 24, 6
        rows.append({
            "id": f"f{i}", "dataset_name": "toy", "freq": "H",
            "past_len": pl, "future_len": fl, "total_len": pl + fl,
            "history_values": rng.normal(size=pl).astype(np.float32).tolist(),
            "history_timestamps": ([f"2021-01-01 {h:02d}:00" for h in range(pl)]
                                   if with_ts else None),
            "future_values": rng.normal(size=fl).astype(np.float32).tolist(),
            "future_timestamps": None,
            "prompt": f"## Dataset background\nToy.\n## Task\nPredict the next {fl} values.",
            "reasoning": "The series is noisy around zero.",
            "conclusion": "Future stays near zero.",
            "roi_start_idx": pl, "roi_end_idx": pl + 2, "split": "test" if i == 2 else "train",
        })
    pp = os.path.join(d, "f.parquet")
    pd.DataFrame(rows).to_parquet(pp)
    return pp


def test_understanding_ts_as_text(tok, d):
    jp = _write_understanding_jsonl(d)
    ds = UnderstandingJsonlDataset([jp], tok, max_user_tokens=256, max_tokens=1024,
                                   sample_chunks=8, ts_as_text=True)
    item = ds[0]
    assert tuple(item["history"].shape) == (1, 16), "under ts_as_text history should be the (1,16) placeholder"
    text = tok.decode(item["input_ids"])
    raw = np.load(os.path.join(d, "short0.npy"))
    assert format_value(raw[0]) in text and format_value(raw[-1]) in text, "rendered text should contain the first and last values"
    assert "Question 0" in text
    # The supervised span (CoT) should contain the reasoning and the answer.
    sup = tok.decode([t for t in item["labels"] if t != -100])
    assert "yes" in sup and "slope" in sup
    # Multi-channel rendering: under a 256-token budget middle truncation eats some channel
    # labels; check the **head label** and the **tail value of channel 3** (they sit in the
    # head/tail segments kept by truncation and must survive).
    item_mc = ds[3]
    text_mc = tok.decode(item_mc["input_ids"])
    raw_mc = np.load(os.path.join(d, "mc.npy"))  # (T=25, C=3); channel 3 = column 2
    assert "Channel 1:" in text_mc and format_value(raw_mc[-1, 2]) in text_mc
    # token_lengths: the long-series sample takes the IO-free shortcut == capped value; consistent
    # with the overall rendering accounting (all <= max_tokens).
    lens = ds.token_lengths()
    assert len(lens) == 4 and all(0 < l <= 1024 for l in lens)
    ds_ref = UnderstandingJsonlDataset([jp], tok, max_user_tokens=256, max_tokens=1024,
                                       sample_chunks=8, ts_as_text=True)
    # Long sample (idx=2): shortcut value = real rendered capped value. Compute the real rendered
    # length by hand for comparison.
    from chronos_llm.data.chat_utils import measure_llm_text_lengths
    real = measure_llm_text_lengths(
        tok, user_texts=[ds_ref._user_text_with_ts(ds_ref.data[2])],
        answer_texts=["no"], reasoning_texts=["Long series reasoning."],
        max_user_tokens=256, max_tokens=1024)
    assert lens[2] == real[0], f"shortcut {lens[2]} != real rendering {real[0]}"
    # Inference-mode meta: input_text carries no series block (keeps the output jsonl small).
    ds_inf = UnderstandingJsonlDataset([jp], tok, max_user_tokens=256, max_tokens=1024,
                                       inference_mode=True, ts_as_text=True)
    it = ds_inf[0]
    assert "Question 0" in it["meta"]["input_text"] and "Raw time series" not in it["meta"]["input_text"]
    assert format_value(raw[0]) in tok.decode(it["input_ids"]), "the inference prefix should also contain the series text"
    print("understanding ts_as_text OK")


def test_forecast_ts_as_text(tok, d):
    pp = _write_forecast_parquet(d)
    ds = ForecastParquetDataset(pp, tok, split="train", max_user_tokens=2048, max_tokens=4096,
                                ts_as_text=True)
    item = ds[0]
    text = tok.decode(item["input_ids"])
    assert "Historical time series (frequency=H)" in text and "2021-01-01 00:00" in text
    assert "Predicted values:" in text  # output instruction + value block in the supervised answer
    sup = tok.decode([t for t in item["labels"] if t != -100])
    assert "Future stays near zero." in sup and "Predicted values: [" in sup
    fut = np.asarray(ds.df.iloc[0]["future_values"], dtype=np.float32)
    assert format_value(fut[0]) in sup and format_value(fut[-1]) in sup
    # The supervised value block can be recovered by the parser.
    got = parse_forecast_values(sup, len(fut))
    assert got is not None and np.allclose(got, fut)
    # Missing-timestamps path.
    os.makedirs(d + "/nots", exist_ok=True)
    pp2 = _write_forecast_parquet(d + "/nots", with_ts=False)
    ds2 = ForecastParquetDataset(pp2, tok, split="train", max_user_tokens=2048, max_tokens=4096,
                                 ts_as_text=True)
    assert "Historical time series" in tok.decode(ds2[0]["input_ids"])
    # token_lengths has the same source as the actual rendering (including the value block).
    lens = ds.token_lengths()
    assert len(lens) == len(ds) and all(0 < l <= 4096 for l in lens)
    print("forecast ts_as_text OK")


def test_model_llm_only(tok, d):
    torch.manual_seed(0)
    llm = _tiny_llm(tok)
    cfg = ChronosLLMConfig(llm_path=LLM, llm_only=True)
    model = ChronosLLM(cfg, None, llm, tok)
    assert model.chronos is None and model.history_qformer is None
    assert model.soft_token_count(100, 4) == 0

    coll = ChronosLLMCollator(tokenizer=tok)
    jp = os.path.join(d, "u.jsonl")
    u_ds = UnderstandingJsonlDataset([jp], tok, max_user_tokens=256, max_tokens=1024,
                                     ts_as_text=True)
    batch_u = coll([u_ds[0], u_ds[3]])  # C=1 + multi-channel placeholder mixed batch
    out = model(batch_u)
    assert torch.isfinite(out["loss"]), "understanding forward loss should be finite"
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.llm.parameters())
    model.zero_grad()

    pp = os.path.join(d, "f.parquet")
    f_ds = ForecastParquetDataset(pp, tok, split="train", max_user_tokens=2048, max_tokens=4096,
                                  ts_as_text=True)
    batch_f = coll([f_ds[0], f_ds[1]])
    out_f = model(batch_f)
    assert torch.isfinite(out_f["loss"]), "forecast forward loss should be finite"
    assert "pred_loss" not in out_f, "the llm_only forecast branch should have no pred_loss"

    # add_lora: only LoRA remains trainable.
    peft = add_lora(model, r=4, alpha=8, dropout=0.0)
    trainable = [n for n, p in peft.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable), f"unexpected trainable parameters: {trainable[:5]}"

    # Generation: understanding yields text; forecast yields (B, 21, H) (tiny random model fails to parse => NaN is legal).
    base = peft.get_base_model()
    u_inf = UnderstandingJsonlDataset([jp], tok, max_user_tokens=256, max_tokens=1024,
                                      inference_mode=True, ts_as_text=True)
    gu = base.generate_understanding(coll([u_inf[0], u_inf[1]]), max_new_tokens=4)
    assert len(gu) == 2 and all(isinstance(t, str) for t in gu)
    f_inf = ForecastParquetDataset(pp, tok, split="test", max_user_tokens=2048, max_tokens=4096,
                                   ts_as_text=True, inference_mode=True)
    bf = coll([f_inf[0]])
    fl = int(bf["future"].shape[-1])
    gf = base.generate_forecast(bf, horizon=fl, max_new_tokens=4)
    assert gf["quantile_preds"].shape == (1, len(CHRONOS2_QUANTILE_LEVELS), fl)
    try:
        base.generate_forecast_teacher_forced(bf, horizon=fl)
        raise AssertionError("teacher-forced should be explicitly rejected under llm_only")
    except RuntimeError:
        pass
    # Successful-parse path: feed a text containing a valid array straight into the qp filling
    # logic (the tiny model generates blindly; it is not required to write an array).
    from chronos_llm.data.ts_text import parse_forecast_values as _p
    assert _p("Predicted values: [" + ", ".join(["1.5"] * fl) + "]", fl) is not None

    # Mixed-horizon batch: after right NaN padding of variable-length futures, a short-fl sample
    # must be parsed with **its own** fl, not with the padded max FL.
    bf2 = coll([f_inf[0], f_inf[0]])  # two copies of the same sample suffice (future is overwritten below)
    bf2["future"] = torch.full((2, 6), float("nan"))
    bf2["future"][0, :] = 1.0   # sample 0 fl=6
    bf2["future"][1, :3] = 1.0  # sample 1 fl=3 (tail NaN pad)
    _orig_bd = tok.batch_decode
    tok.batch_decode = lambda *a, **k: ["p [1, 2, 3, 4, 5, 6]", "p [7, 8, 9]"]
    try:
        gf2 = base.generate_forecast(bf2, horizon=6, max_new_tokens=4)
    finally:
        tok.batch_decode = _orig_bd
    q2 = gf2["quantile_preds"]
    assert torch.isfinite(q2[0]).all() and float(q2[0, 0, 5]) == 6.0
    assert torch.isfinite(q2[1, :, :3]).all() and float(q2[1, 0, 2]) == 9.0, \
        "the short-fl sample should parse successfully with its own horizon"
    assert torch.isnan(q2[1, :, 3:]).all(), "the pad region of the short-fl sample should stay NaN"
    print("model llm_only OK")
    return peft


def test_infer_llm_only(tok, d, peft):
    from chronos_llm.eval.infer_forecast import run_forecast_infer
    from chronos_llm.eval.infer_understanding import run_understanding_infer

    base = peft.get_base_model()
    pp = os.path.join(d, "f.parquet")
    f_inf = ForecastParquetDataset(pp, tok, split="test", max_user_tokens=2048, max_tokens=4096,
                                   ts_as_text=True, inference_mode=True)
    npz = os.path.join(d, "out", "preds.npz")
    res = run_forecast_infer(base, tok, f_inf, npz, batch_size=1, max_new_tokens=4,
                             device="cpu", num_workers=0)
    assert os.path.exists(npz)
    assert np.allclose(res["quantile_levels"], np.asarray(CHRONOS2_QUANTILE_LEVELS, np.float32)), \
        "llm_only quantile_levels should fall back to CHRONOS2_QUANTILE_LEVELS"

    jp = os.path.join(d, "u.jsonl")
    outd = os.path.join(d, "out_u")
    files = run_understanding_infer(base, tok, [jp], outd, batch_size=2, max_new_tokens=4,
                                    device="cpu", max_user_tokens=256, max_tokens=1024,
                                    num_workers=0, ts_as_text=True)
    with open(files[0]) as f:
        rows = [json.loads(l) for l in f]
    assert len(rows) == 4 and all("generated_text" in r for r in rows)
    print("infer llm_only OK")


def main():
    test_ts_text_unit()
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    with tempfile.TemporaryDirectory() as d:
        test_understanding_ts_as_text(tok, d)
        test_forecast_ts_as_text(tok, d)
        peft = test_model_llm_only(tok, d)
        test_infer_llm_only(tok, d, peft)
    print("test_llm_only: ALL OK")


if __name__ == "__main__":
    main()
