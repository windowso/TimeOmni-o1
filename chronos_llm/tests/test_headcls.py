"""End-to-end CPU test of the LLM-free understanding model on the MMTR understanding pool
(chronos2 + multi-task classification heads), with the real chronos-2.

Covers: label-map construction / candidate parsing on real HAR and Sleep question texts /
forward with C=1 and C=3, B=1 and B>1 / finite loss with gradients reaching chronos and the
heads / candidate-restricted argmax / save->load value-identical /
train_headcls.run_infer output format (generated_text recoverable by norm_answer).
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chronos_llm.data.headcls_dataset import (  # noqa: E402
    HeadClsDataset, build_label_maps, collate_headcls, norm_answer, parse_candidates,
)
from chronos_llm.models.chronos_head_classifier import ChronosHeadClassifier  # noqa: E402
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn  # noqa: E402
from chronos_llm.train_headcls import run_infer  # noqa: E402

CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")

# Question fragments identical to the real data (regression for candidate parsing)
HAR_Q = (
    "You are shown a time-series plot of accelerometer over a 2.56 second window. "
    "This data corresponds to one of two possible activities:\n    running\n    sitting\n\n"
    "    Your task is to classify the activity based on analysis of the data."
)
SLEEP_Q = (
    "You are presented with a time-series plot showing EEG data collected over a 30-second "
    "interval. This signal corresponds to one of two possible sleep stages:\n"
    "    - Non-REM stage 4\n    - Wake\n\n    Your task is to determine the correct sleep stage."
)


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_data(tmp):
    """Two tasks: har_cot (3 channels, with candidates) + veritime_RCW (single-channel binary)."""
    rng = np.random.default_rng(0)
    rows_tr, rows_te = [], []
    for i in range(8):
        p = os.path.join(tmp, f"har_{i}.npy")
        np.save(p, rng.normal(size=(128, 3)).astype(np.float32))  # stored as (T, C)
        label = "running" if i % 2 else "sitting"
        row = {
            "task": "har_cot", "scene": "HAR", "id": f"har/{i}",
            "input_text": [HAR_Q], "gt_text": [f"Answer: {label}"],
            "input_ts": {"original": {"ori_path": p, "ori_length": 128}},
        }
        (rows_tr if i < 6 else rows_te).append(row)
    for i in range(8):
        p = os.path.join(tmp, f"rcw_{i}.npy")
        np.save(p, rng.normal(size=(200,)).astype(np.float32))
        label = "Right Whale present" if i % 2 else "No Right Whale"
        row = {
            "task": "veritime_RCW", "scene": "RCW", "id": f"rcw/{i}",
            "input_text": ["Is a right whale up-call present?"],
            "gt_text": [f"Answer: {label}"],
            "input_ts": {"original": {"ori_path": p, "ori_length": 200}},
        }
        (rows_tr if i < 6 else rows_te).append(row)
    # Rows of non-applicable tasks must be filtered out
    rows_tr.append({"task": "hitsr_l2", "id": "x", "input_text": ["q"],
                    "gt_text": ["Answer: a"], "input_ts": {"original": {"ori_path": ""}}})
    tr, te = os.path.join(tmp, "train.jsonl"), os.path.join(tmp, "har_test.jsonl")
    _write_jsonl(tr, rows_tr)
    _write_jsonl(te, rows_te)
    return tr, te


def main():
    tmp = tempfile.mkdtemp(prefix="headcls_test_")
    tr, te = _make_data(tmp)

    # 1) Label maps and candidate parsing
    lm = build_label_maps([tr])
    assert set(lm) == {"har_cot", "veritime_RCW"}, f"non-applicable task not filtered: {set(lm)}"
    assert set(lm["har_cot"]["labels"]) == {"running", "sitting"}
    assert lm["veritime_RCW"]["surface"]["no right whale"] == "No Right Whale"
    assert parse_candidates(HAR_Q, {"running", "sitting", "walking"}) == ["running", "sitting"]
    assert parse_candidates(SLEEP_Q, {"non-rem stage 4", "wake", "rem sleep"}) == \
        ["non-rem stage 4", "wake"]
    assert parse_candidates("no candidates here", {"a", "b"}) is None

    # 2) Dataset and collator (C=3 and C=1 mixed batch)
    ds = HeadClsDataset([tr], lm)
    assert len(ds) == 12
    assert ds[0]["series"].shape == (3, 128) and ds[0]["cand"] is not None
    batch = collate_headcls([ds[0], ds[6]])          # har(C=3) + rcw(C=1)
    assert batch["context"].shape == (4, 200) and batch["n_channels"].tolist() == [3, 1]
    assert torch.isnan(batch["context"][0, :72]).all()   # left NaN pad

    # 3) Model forward/gradient (real chronos-2)
    chronos = load_chronos2_with_cross_attn(CHRONOS)
    model = ChronosHeadClassifier(chronos, {t: m["labels"] for t, m in lm.items()})
    model.train()
    loss, preds = model(batch["context"], batch["true_lengths"], batch["n_channels"],
                        batch["tasks"], labels=batch["labels"],
                        candidates=batch["candidates"])
    assert loss is not None and math.isfinite(float(loss)), f"loss={loss}"
    assert preds.shape == (2,) and (preds >= 0).all()
    loss.backward()
    g_chronos = next(p.grad for n, p in model.named_parameters()
                     if n.startswith("chronos.") and p.grad is not None)
    assert torch.isfinite(g_chronos).all() and g_chronos.abs().sum() > 0
    assert model.heads["har_cot"].weight.grad is not None
    assert model.heads["har_cot"].weight.grad.abs().sum() > 0

    # 4) B=1 and B>1 value-identical (single-channel samples; chronos encode has zero cross-group interaction)
    model.eval()
    with torch.no_grad():
        b1 = collate_headcls([ds[6]])
        f_single = model.pooled_features(b1["context"], b1["true_lengths"], b1["n_channels"])
        b2 = collate_headcls([ds[6], ds[7]])
        f_batch = model.pooled_features(b2["context"], b2["true_lengths"], b2["n_channels"])
    assert torch.allclose(f_single[0], f_batch[0], atol=1e-4), \
        f"B=1 vs B=2 feature drift {(f_single[0] - f_batch[0]).abs().max()}"

    # 5) Candidate-restricted argmax: restrict the candidates to a single class, the prediction must be that class
    with torch.no_grad():
        _, p_forced = model(b1["context"], b1["true_lengths"], b1["n_channels"],
                            b1["tasks"], candidates=[[1]])
    assert p_forced.item() == 1

    # 6) save -> load value-identical
    ck = os.path.join(tmp, "ckpt")
    model.save_pretrained(ck)
    m2 = ChronosHeadClassifier.from_pretrained(ck, CHRONOS)
    m2.eval()
    with torch.no_grad():
        f2 = m2.pooled_features(b1["context"], b1["true_lengths"], b1["n_channels"])
        h1 = model.trunk(model.norm(f_single))
        h2 = m2.trunk(m2.norm(f2))
    assert torch.allclose(h1, h2, atol=1e-5), "features differ after save/load"

    # 7) run_infer output: same-named jsonl, all fields present, answers recoverable by norm_answer
    inf_dir = os.path.join(tmp, "infer")
    run_infer(model, [te], lm, torch.device("cpu"), inf_dir, bs=4, workers=0)
    out = [json.loads(l) for l in open(os.path.join(inf_dir, "har_test.jsonl"))]
    assert len(out) == 4
    for r in out:
        assert set(r) >= {"id", "task", "scene", "generated_text", "ground_truth"}
        assert r["generated_text"].startswith("Answer: ")
        vocab = set(lm[r["task"]]["labels"])
        assert norm_answer(r["generated_text"]) in vocab

    print("test_headcls: OK")


if __name__ == "__main__":
    main()
