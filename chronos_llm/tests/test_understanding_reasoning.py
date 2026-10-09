"""CPU unit tests for the understanding-branch ``think`` (reasoning supervision) field.

Coverage:
1. ``UnderstandingJsonlDataset._reasoning``: missing / empty / whitespace-only think -> None;
   non-empty -> stripped.
2. Samples without think (the plain TimeOmni-v2 jsonl format) render token-for-token identically
   to the pre-change behaviour, in both training and inference (regression guard).
3. Samples with think: the supervised span contains reasoning + </think> + answer; the inference
   prefix stops at the open ``<think>`` and does not contain the reasoning itself -- mirrors the
   pure-function assertions of test_inference_prefix_alignment.py, but here at the dataset level to
   verify that the jsonl ``think`` field is actually forwarded to chat_utils.
4. Both sample kinds mixed in one jsonl render correctly and independently (the core new capability).
5. token_lengths() does not crash on a mixed-reasoning jsonl and returns sensible lengths.
6. In inference mode ``meta["gt_reasoning"]`` is filled correctly (non-empty for samples with think,
   empty string otherwise).
"""
import json
import os
import tempfile
import warnings

import numpy as np
from transformers import AutoTokenizer

from chronos_llm.data.chat_utils import build_inference_prefix_ids, build_supervised_ids
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.models.chronos_llm_model import _add_ts_special_tokens
from chronos_llm.tests.test_pretrained_peft import LLM

REASONING = "RR intervals are irregular and no P waves are visible."


def _tok():
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    return tok


def _write_jsonl(rows, d):
    p = os.path.join(d, "u.jsonl")
    with open(p, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    return p


def test_reasoning_field_parsing():
    for think, expect in [(None, None), ("", None), ("   ", None),
                           ("  hi  ", "hi"), ("real reasoning", "real reasoning")]:
        got = UnderstandingJsonlDataset._reasoning({"think": think} if think is not None else {})
        assert got == expect, f"think={think!r} -> {got!r}, expected {expect!r}"
    print("_reasoning: missing/empty/whitespace -> None, non-empty stripped OK")


def test_mixed_reasoning_dataset(tok=None):
    tok = tok or _tok()
    with tempfile.TemporaryDirectory() as d:
        ts = os.path.join(d, "ts.npy")
        np.save(ts, np.random.randn(32).astype(np.float32))
        rows = [
            {"id": "no_think", "input_text": ["plain question?"], "gt_text": ["plain answer."],
             "input_ts": {"original": {"ori_path": ts}}},
            {"id": "with_think", "input_text": ["reasoning question?"], "gt_text": ["reasoned answer."],
             "think": REASONING, "input_ts": {"original": {"ori_path": ts}}},
        ]
        p = _write_jsonl(rows, d)

        # ---- training mode ----
        ds = UnderstandingJsonlDataset([p], tok, max_user_tokens=128, max_tokens=512)
        assert len(ds) == 2

        it_plain = ds[0]
        sup_plain = tok.decode(
            [i for i, l in zip(it_plain["input_ids"], it_plain["labels"]) if l != -100],
            skip_special_tokens=False)
        assert "</think>" not in sup_plain and "<think>" not in sup_plain, (
            f"supervised span of a no-think sample must not contain think tags: {sup_plain!r}")
        # Token-for-token identical to calling chat_utils directly (reasoning_content=None),
        # confirming lossless pass-through.
        ref_ids, ref_labels = build_supervised_ids(
            tok, "plain question?", "plain answer.", reasoning_content=None,
            max_user_tokens=128, max_tokens=512)
        assert it_plain["input_ids"] == ref_ids
        assert it_plain["labels"] == ref_labels

        it_r = ds[1]
        sup_r = tok.decode(
            [i for i, l in zip(it_r["input_ids"], it_r["labels"]) if l != -100],
            skip_special_tokens=False)
        assert "</think>" in sup_r, f"supervised span of a think sample must contain </think>: {sup_r!r}"
        assert "irregular" in sup_r, f"supervised span must contain the reasoning text: {sup_r!r}"
        ref_ids_r, ref_labels_r = build_supervised_ids(
            tok, "reasoning question?", "reasoned answer.", reasoning_content=REASONING,
            max_user_tokens=128, max_tokens=512)
        assert it_r["input_ids"] == ref_ids_r
        assert it_r["labels"] == ref_labels_r

        # ---- inference mode ----
        ds_inf = UnderstandingJsonlDataset([p], tok, max_user_tokens=128, max_tokens=512,
                                            inference_mode=True)
        item_plain = ds_inf[0]
        dec_plain = tok.decode(item_plain["input_ids"], skip_special_tokens=False)
        assert "</think>" in dec_plain, f"inference prefix of a no-think sample must contain a closed empty think: {dec_plain!r}"
        assert item_plain["meta"]["gt_reasoning"] == "", "gt_reasoning of a no-think sample must be the empty string"

        item_r = ds_inf[1]
        dec_r = tok.decode(item_r["input_ids"], skip_special_tokens=False)
        assert "<think>" in dec_r and "</think>" not in dec_r, (
            f"inference prefix of a think sample must stop at the open <think>: {dec_r!r}")
        assert "irregular" not in dec_r, "inference prefix must not leak the reasoning itself (the model has to generate it)"
        assert item_r["meta"]["gt_reasoning"] == REASONING, (
            f"gt_reasoning of a think sample must carry the original text, got {item_r['meta']['gt_reasoning']!r}")

        # Prefix aligned token-for-token with the training conditioning context (same assertion
        # protocol as test_inference_prefix_alignment).
        inf_ref_plain = build_inference_prefix_ids(tok, "plain question?", reasoning=False,
                                                    max_user_tokens=128, max_tokens=512)
        assert item_plain["input_ids"] == inf_ref_plain
        inf_ref_r = build_inference_prefix_ids(tok, "reasoning question?", reasoning=True,
                                                max_user_tokens=128, max_tokens=512)
        assert item_r["input_ids"] == inf_ref_r

        # ---- token_lengths: mixed reasoning does not crash, returns sensible positive ints ----
        with tempfile.TemporaryDirectory() as cache_d:
            lens = ds.token_lengths(cache_dir=cache_d)
            assert len(lens) == 2 and all(l > 0 for l in lens), f"token_lengths abnormal: {lens}"
            lens2 = ds.token_lengths(cache_dir=cache_d)  # cache hit, identical result
            assert lens == lens2

    print("mixed think / no-think samples: supervised span, inference prefix, gt_reasoning, token_lengths all OK")


def test_no_reasoning_ablation(tok=None):
    """no-CoT control arm (``no_reasoning=True``): samples with think are treated as if they had none.

    This is the control arm of the main MMTR understanding experiment; three core assertions:

    1. **Training**: the supervised span contains neither the reasoning nor `</think>`, and is
       token-for-token identical to "the same sample with the think field removed" => apart from not
       learning the CoT span, everything else (template / time-series placeholder / answer) is
       unchanged, so the difference between the arms is attributable to the CoT itself.
    2. **Inference**: the prefix contains the complete empty `<think></think>` (the model only
       continues with the answer) rather than stopping at the open `<think>` -- training and
       inference protocols must be driven by the same switch, otherwise the no-CoT model would be
       asked to continue a span it never learned.
    3. **gt_reasoning is still filled with the original text**: the ground truth does not depend on
       the training configuration, so the "expected reasoning" can be compared afterwards.
    """
    tok = tok or _tok()
    with tempfile.TemporaryDirectory() as d:
        ts = os.path.join(d, "ts.npy")
        np.save(ts, np.random.randn(32).astype(np.float32))
        row = {"id": "with_think", "input_text": ["reasoning question?"],
               "gt_text": ["reasoned answer."], "think": REASONING,
               "input_ts": {"original": {"ori_path": ts}}}
        p = _write_jsonl([row], d)
        # Reference: the same sample, but the jsonl never had a think field.
        row_wo = {k: v for k, v in row.items() if k != "think"}
        p_wo = os.path.join(d, "u_wo.jsonl")
        with open(p_wo, "w") as f:
            f.write(json.dumps(row_wo))

        kw = dict(max_user_tokens=128, max_tokens=512)
        ds_cot = UnderstandingJsonlDataset([p], tok, **kw)
        ds_nocot = UnderstandingJsonlDataset([p], tok, no_reasoning=True, **kw)
        ds_plain = UnderstandingJsonlDataset([p_wo], tok, **kw)

        # 1. training: no-CoT == naturally no think, and != with CoT
        assert ds_nocot[0]["input_ids"] == ds_plain[0]["input_ids"], (
            "no_reasoning rendering must be token-for-token identical to a sample that never had think")
        assert ds_nocot[0]["labels"] == ds_plain[0]["labels"]
        assert ds_nocot[0]["input_ids"] != ds_cot[0]["input_ids"], (
            "no_reasoning must actually change the rendering (otherwise the switch is a no-op)")
        sup = tok.decode([i for i, l in zip(ds_nocot[0]["input_ids"], ds_nocot[0]["labels"])
                          if l != -100], skip_special_tokens=False)
        assert "irregular" not in sup, f"no-CoT supervised span must not contain the reasoning text: {sup!r}"
        assert "</think>" not in sup, f"no-CoT supervised span must not contain </think>: {sup!r}"

        # 2. inference: complete empty think (answer only), not stopping at the open tag
        inf_nocot = UnderstandingJsonlDataset([p], tok, inference_mode=True,
                                              no_reasoning=True, **kw)
        dec = tok.decode(inf_nocot[0]["input_ids"], skip_special_tokens=False)
        assert "</think>" in dec, f"no-CoT inference prefix must contain a closed empty think: {dec!r}"
        assert "irregular" not in dec, "inference prefix must not leak the reasoning"
        assert inf_nocot[0]["input_ids"] == build_inference_prefix_ids(
            tok, "reasoning question?", reasoning=False, max_user_tokens=128, max_tokens=512)

        # 3. gt_reasoning ground truth does not depend on the training configuration
        assert inf_nocot[0]["meta"]["gt_reasoning"] == REASONING, (
            "the no-CoT arm must also fill gt_reasoning with the sample's own think (for later comparison)")

        # 4. the token-cache key must distinguish the two arms: the same jsonl (same fingerprint)
        #    must not share the length array
        with tempfile.TemporaryDirectory() as cache_d:
            len_cot = ds_cot.token_lengths(cache_dir=cache_d)
            len_nocot = ds_nocot.token_lengths(cache_dir=cache_d)
            assert len_nocot[0] < len_cot[0], (
                f"no-CoT supervises one reasoning span less, so its token count must be smaller (cot={len_cot} nocot={len_nocot})"
                "; equality means the cache key lacks no_reasoning and reused the other arm's result")

    print("no-CoT control arm: supervised span / inference prefix / gt_reasoning / token cache key all OK")


def test_answer_first_explanation(tok=None):
    """answer-first mode: supervised span = `answer\\n\\nexplanation`, think left empty.

    The answer comes first and the explanation follows it. Four assertions:

    1. The supervised span is **answer then explanation**, and contains **no** `<think>`/`</think>`
       (the explanation is not wrapped in think);
    2. **The inference prefix is token-for-token identical to no-CoT** => such checkpoints are
       evaluated directly with NO_REASONING=1, the protocols align naturally, no extra switch needed;
    3. `meta["ground_truth"]` must be the **original answer** without the explanation -- otherwise
       the evaluation ground truth also carries the whole explanation and exact match fails everywhere
       (the easiest pitfall of this mode: sharing one answer variable between training and evaluation
       triggers it);
    4. The token-cache key includes this switch (same jsonl fingerprint, but the supervised span has an
       extra explanation).
    """
    tok = tok or _tok()
    with tempfile.TemporaryDirectory() as d:
        ts = os.path.join(d, "ts.npy")
        np.save(ts, np.random.randn(32).astype(np.float32))
        row = {"id": "x", "input_text": ["q?"], "gt_text": ["Answer: laptop"],
               "think": REASONING, "input_ts": {"original": {"ori_path": ts}}}
        p = _write_jsonl([row], d)
        kw = dict(max_user_tokens=128, max_tokens=512)

        af = UnderstandingJsonlDataset([p], tok, answer_first_explanation=True, **kw)[0]
        sup = tok.decode([i for i, l in zip(af["input_ids"], af["labels"]) if l != -100],
                         skip_special_tokens=False)
        # 1. answer before explanation, no think tags
        assert sup.index("laptop") < sup.index("irregular"), f"answer must precede the explanation: {sup!r}"
        assert "<think>" not in sup and "</think>" not in sup, f"explanation must not be wrapped in think: {sup!r}"
        assert REASONING in sup, f"supervised span must contain the explanation text: {sup!r}"

        # 2. inference prefix == no-CoT prefix (token-for-token)
        inf_af = UnderstandingJsonlDataset([p], tok, answer_first_explanation=True,
                                           inference_mode=True, **kw)[0]
        inf_nocot = UnderstandingJsonlDataset([p], tok, no_reasoning=True,
                                              inference_mode=True, **kw)[0]
        assert inf_af["input_ids"] == inf_nocot["input_ids"], \
            "answer-first inference prefix must be token-for-token identical to no-CoT (so evaluation can use NO_REASONING=1)"

        # 3. meta ground truth is the original answer, without the explanation
        assert inf_af["meta"]["ground_truth"] == "Answer: laptop", \
            f"ground_truth must not carry the explanation, got {inf_af['meta']['ground_truth']!r}"

        # 4. cache key distinction: answer-first supervises an extra explanation => more tokens
        ds_af = UnderstandingJsonlDataset([p], tok, answer_first_explanation=True, **kw)
        ds_plain = UnderstandingJsonlDataset([p], tok, **kw)
        with tempfile.TemporaryDirectory() as cache_d:
            l_af = ds_af.token_lengths(cache_dir=cache_d)
            l_plain = ds_plain.token_lengths(cache_dir=cache_d)
            assert l_af[0] != l_plain[0], \
                f"cache key lacks answer_first_explanation and reused the other mode's lengths ({l_af} vs {l_plain})"

        # mutual-exclusion guard
        try:
            UnderstandingJsonlDataset([p], tok, answer_first_explanation=True,
                                      no_reasoning=True, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("enabling answer_first_explanation together with no_reasoning must raise")
    print("answer-first: answer then explanation / no think wrapper / prefix==no-CoT / ground truth without explanation / cache key OK")


def test_prefix_override_switches(tok=None):
    """Two inference-time prefix override switches, both effective only in ``inference_mode``.

    ``force_reasoning``: for a test set whose samples carry no think (e.g. ST-Bench), the default
    protocol answers directly (complete empty think); with the switch on, no-think samples also
    stop at the open `<think>`.

    ``teacher_forced_reasoning``: a diagnostic protocol -- the ground-truth reasoning goes into the
    prefix and the model only continues with the answer.

    Both must **only** change the inference prefix, never the training rendering, and must be
    mutually exclusive / non-conflicting with ``no_reasoning``.
    """
    tok = tok or _tok()
    with tempfile.TemporaryDirectory() as d:
        ts = os.path.join(d, "ts.npy")
        np.save(ts, np.random.randn(32).astype(np.float32))
        # a no-think sample -- exactly the shape of the ST-Bench test set
        row_wo = {"id": "no_think", "input_text": ["plain question?"],
                  "gt_text": ["plain answer."], "input_ts": {"original": {"ori_path": ts}}}
        row_w = dict(row_wo, id="with_think", think=REASONING)
        # _write_jsonl always writes <dir>/u.jsonl => the two samples must go to **different file
        # names**, otherwise the second write overwrites the first (force_reasoning would then be
        # tested on the wrong sample)
        p_wo = _write_jsonl([row_wo], d)
        p_w = os.path.join(d, "w.jsonl")
        with open(p_w, "w") as f:
            f.write(json.dumps(row_w))

        kw = dict(max_user_tokens=128, max_tokens=512, inference_mode=True)
        base = UnderstandingJsonlDataset([p_wo], tok, **kw)[0]["input_ids"]
        forced = UnderstandingJsonlDataset([p_wo], tok, force_reasoning=True, **kw)[0]["input_ids"]
        assert forced != base, "force_reasoning must actually change the prefix of a no-think sample (otherwise the switch is useless)"
        dec_b, dec_f = (tok.decode(x, skip_special_tokens=False) for x in (base, forced))
        assert "</think>" in dec_b, f"under the default protocol a no-think sample must contain a closed empty think: {dec_b!r}"
        assert "</think>" not in dec_f, f"with force_reasoning the prefix must stop at the open think: {dec_f!r}"

        # a sample that already has think already stops at the open tag => the switch is a no-op
        # for it (token-for-token unchanged)
        assert UnderstandingJsonlDataset([p_w], tok, force_reasoning=True, **kw)[0]["input_ids"] \
            == UnderstandingJsonlDataset([p_w], tok, **kw)[0]["input_ids"], \
            "force_reasoning must not alter a sample that already has think"

        # teacher-forced: ground-truth reasoning goes into the prefix
        tf = UnderstandingJsonlDataset([p_w], tok, teacher_forced_reasoning=True, **kw)[0]
        dec_tf = tok.decode(tf["input_ids"], skip_special_tokens=False)
        assert REASONING in dec_tf and "</think>" in dec_tf, \
            f"teacher-forced prefix must contain the ground-truth reasoning and be closed: {dec_tf!r}"

        # counterfactual: inserts the reasoning of **another sample of the same task** (cyclic
        # offset), and takes precedence over teacher-forced
        OTHER = "Completely different reasoning about a flat baseline."
        rows = [dict(row_w, id="s0", task="t", think=REASONING),
                dict(row_w, id="s1", task="t", think=OTHER)]
        p_cf = os.path.join(d, "cf.jsonl")
        with open(p_cf, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
        cf = UnderstandingJsonlDataset([p_cf], tok, counterfactual_reasoning=True, **kw)
        d0 = tok.decode(cf[0]["input_ids"], skip_special_tokens=False)
        assert OTHER in d0 and REASONING not in d0, \
            f"sample 0 must receive the reasoning of sample 1 (cyclic offset), got: {d0!r}"
        d1 = tok.decode(cf[1]["input_ids"], skip_special_tokens=False)
        assert REASONING in d1 and OTHER not in d1, f"the last sample must wrap around to the first sample's reasoning: {d1!r}"
        # when both are on, counterfactual wins (both insert reasoning into the prefix; the
        # counterfactual is the stronger question)
        both = UnderstandingJsonlDataset([p_cf], tok, counterfactual_reasoning=True,
                                         teacher_forced_reasoning=True, **kw)
        assert OTHER in tok.decode(both[0]["input_ids"], skip_special_tokens=False), \
            "counterfactual must take precedence over teacher_forced"

        # guards: outside inference mode / together with no_reasoning, both must raise instead of
        # silently doing nothing
        for bad in (dict(force_reasoning=True), dict(teacher_forced_reasoning=True),
                    dict(counterfactual_reasoning=True)):
            try:
                UnderstandingJsonlDataset([p_w], tok, max_tokens=512, **bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{bad} must raise outside inference_mode")
        try:
            UnderstandingJsonlDataset([p_w], tok, force_reasoning=True, no_reasoning=True, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("enabling force_reasoning together with no_reasoning must raise (self-contradictory semantics)")

    print("force_reasoning (no-think prefix changed / think no-op) + teacher-forced + counterfactual offset pairing + guards OK")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tok = _tok()
        test_reasoning_field_parsing()
        test_mixed_reasoning_dataset(tok)
        test_no_reasoning_ablation(tok)
        test_prefix_override_switches(tok)
        test_answer_first_explanation(tok)
    print("ALL OK")
