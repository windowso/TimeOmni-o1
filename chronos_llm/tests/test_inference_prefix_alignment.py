"""CPU unit test: the inference prefix is aligned with the training conditioning context.

Core assertion: the prefix produced by ``build_inference_prefix_ids`` is token-for-token equal to
the conditioning context "before the first supervised token" inside the training
``build_supervised_ids``. I.e. the prefix the model sees at inference is **exactly** the
conditional distribution it was teacher-forced on during training, and the model continues with
precisely the first token it was taught.

- Understanding task (reasoning=None / False): the prefix must contain the complete
  ``<think>\n\n</think>\n\n`` (the closing scaffold is masked during training and the model never
  learned to generate it); the model only continues with the answer.
- Forecasting task (with reasoning / True): the prefix stops at ``<think>\n``; the model continues
  with reasoning+</think>+conclusion.
"""
from transformers import AutoTokenizer

from chronos_llm.data.chat_utils import build_supervised_ids, build_inference_prefix_ids
from chronos_llm.models.chronos_llm_model import _add_ts_special_tokens
from chronos_llm.tests.test_pretrained_peft import LLM

USER = "What is the dominant pattern in this ECG segment?"
ANSWER = "Atrial fibrillation with irregular R-R intervals."
REASONING = "RR intervals are irregular and no P waves are visible."


def _train_prefix(ids, labels):
    """Training conditioning context before the first supervised token = full[:index of first labels!=-100]."""
    k = next(i for i, l in enumerate(labels) if l != -100)
    return ids[:k]


def test_understanding_prefix_alignment():
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True); _add_ts_special_tokens(tok)
    ids, labels = build_supervised_ids(tok, USER, ANSWER, reasoning_content=None,
                                       max_user_tokens=128, max_tokens=512)
    train_pref = _train_prefix(ids, labels)
    inf = build_inference_prefix_ids(tok, USER, reasoning=False, max_user_tokens=128, max_tokens=512)
    assert inf == train_pref, (
        f"understanding inference prefix != training conditioning context\ninf : {tok.decode(inf)!r}\ntrain:{tok.decode(train_pref)!r}"
    )
    dec = tok.decode(inf, skip_special_tokens=False)
    assert "</think>" in dec, f"understanding prefix should contain the closed empty think: {dec!r}"
    # Key point: the supervised span of the understanding task does **not** contain </think> -- the
    # model never learned to generate the closing scaffold, so it must go into the prefix.
    sup_dec = tok.decode([i for i, l in zip(ids, labels) if l != -100], skip_special_tokens=False)
    assert "</think>" not in sup_dec, f"understanding supervised span should not contain </think>: {sup_dec!r}"
    assert "<think>" not in sup_dec, f"understanding supervised span should not contain <think>: {sup_dec!r}"
    print(f"understanding: inference prefix == training conditioning context (with closed empty think), supervised span={sup_dec!r} OK")


def test_forecast_prefix_alignment():
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True); _add_ts_special_tokens(tok)
    ids, labels = build_supervised_ids(tok, USER, ANSWER, reasoning_content=REASONING,
                                       max_user_tokens=128, max_tokens=512)
    train_pref = _train_prefix(ids, labels)
    inf = build_inference_prefix_ids(tok, USER, reasoning=True, max_user_tokens=128, max_tokens=512)
    assert inf == train_pref, (
        f"forecast inference prefix != training conditioning context\ninf : {tok.decode(inf)!r}\ntrain:{tok.decode(train_pref)!r}"
    )
    dec = tok.decode(inf, skip_special_tokens=False)
    # The forecast prefix stops at the open think tag: contains <think> but not the closing </think> (tokenisation stops at the <think> token).
    assert "<think>" in dec and "</think>" not in dec, f"forecast prefix should stop at the open think tag: {dec!r}"
    # Key contrast: the supervised span of the forecast task **does** contain </think> -- the model is taught to generate reasoning+close+conclusion.
    sup_dec = tok.decode([i for i, l in zip(ids, labels) if l != -100], skip_special_tokens=False)
    assert "</think>" in sup_dec, f"forecast supervised span should contain </think>: {sup_dec!r}"
    print(f"forecast: inference prefix == training conditioning context (stops at open think), supervised span contains reasoning+</think>+conclusion OK")


def test_teacher_forced_reasoning_prefix():
    """Teacher-forced diagnostic mode: the ground-truth reasoning goes into the prefix with `</think>`
    already closed; the model only continues with the answer.

    Guards three things:
    1. The prefix really contains **that ground-truth reasoning** and is closed by `</think>` --
       otherwise the model is still writing its own reasoning, the diagnostic question ("can it answer
       correctly given a good reasoning") was never actually asked, and the result table would not show it.
    2. The prefix is a **true extension** of the plain prefix (the extra part is exactly
       reasoning+close), not a separate rendering.
    3. **With `reasoning_text=None` it is token-for-token the original behaviour** -- this switch is
       off by default and must never accidentally change the official evaluation convention.
    """
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True); _add_ts_special_tokens(tok)
    kw = dict(max_user_tokens=128, max_tokens=512)

    plain = build_inference_prefix_ids(tok, USER, reasoning=True, **kw)
    tf = build_inference_prefix_ids(tok, USER, reasoning=True, reasoning_text=REASONING, **kw)

    dec = tok.decode(tf, skip_special_tokens=False)
    assert REASONING in dec, f"teacher-forced prefix should contain the ground-truth reasoning verbatim: {dec!r}"
    assert "</think>" in dec, f"teacher-forced prefix should have the think already closed: {dec!r}"
    assert len(tf) > len(plain) and tf[:len(plain)] == plain, \
        "teacher-forced prefix should be a true extension of the plain prefix"

    # Aligned with the training rendering: the tf prefix == the training conditioning context "after reasoning, before the answer"
    ids, labels = build_supervised_ids(tok, USER, ANSWER, reasoning_content=REASONING, **kw)
    dec_sup = tok.decode(ids, skip_special_tokens=False)
    assert dec_sup.startswith(dec), \
        f"teacher-forced prefix should be a prefix of the training rendering\ntf  : {dec!r}\ntrain:{dec_sup[:len(dec) + 40]!r}"

    # Token-for-token unchanged when off by default (regression)
    assert build_inference_prefix_ids(tok, USER, reasoning=True, reasoning_text=None, **kw) == plain
    assert build_inference_prefix_ids(tok, USER, reasoning=False, reasoning_text=REASONING, **kw) \
        == build_inference_prefix_ids(tok, USER, reasoning=False, **kw), \
        "with reasoning=False, reasoning_text should be ignored (there is no think span to feed)"
    print("teacher-forced: ground-truth reasoning in prefix + closed + true extension of the plain prefix; token-for-token unchanged when off OK")


def test_answer_prefix_alignment():
    """Forced answer-prefix decoding: the prefix is token-for-token the training conditioning context
    up to the first token after "Answer:" in the training rendering."""
    import json, os, tempfile
    from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset

    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True); _add_ts_special_tokens(tok)
    ans = "Answer: sitting"
    ids, labels = build_supervised_ids(tok, USER, ans, reasoning_content=None,
                                       max_user_tokens=128, max_tokens=512)
    train_pref = _train_prefix(ids, labels)
    inf = build_inference_prefix_ids(tok, USER, reasoning=False, max_user_tokens=128, max_tokens=512,
                                     answer_prefix="Answer:")
    assert len(inf) > len(train_pref) and inf == ids[:len(inf)], (
        f"the prefix with answer_prefix must be a token-for-token prefix of the training rendering\n"
        f"inf: {tok.decode(inf)!r}\ntrain: {tok.decode(ids)!r}")
    assert tok.decode(inf, skip_special_tokens=False).endswith("Answer:"), tok.decode(inf)
    rest = tok.decode(ids[len(inf):], skip_special_tokens=False)
    assert rest.startswith(" sitting"), f"the first token the model continues with should be the one that follows 'Answer:' in training: {rest!r}"
    # reasoning=True without reasoning_text: the prefix stops at the open <think>, the answer opening
    # could never enter it => must raise instead of silently having no effect
    try:
        build_inference_prefix_ids(tok, USER, reasoning=True, max_user_tokens=128, max_tokens=512,
                                   answer_prefix="Answer:")
        raise AssertionError("answer_prefix should raise with reasoning=True and no reasoning_text")
    except ValueError:
        pass
    # Teacher-forced path: answer_prefix follows the closed ground-truth reasoning, still aligned
    # token-for-token with the training rendering.
    ids2, _ = build_supervised_ids(tok, USER, ans, reasoning_content=REASONING,
                                   max_user_tokens=128, max_tokens=512)
    inf2 = build_inference_prefix_ids(tok, USER, reasoning=True, reasoning_text=REASONING,
                                      max_user_tokens=128, max_tokens=512, answer_prefix="Answer:")
    assert inf2 == ids2[:len(inf2)] and tok.decode(inf2, skip_special_tokens=False).endswith("Answer:")
    assert "</think>" in tok.decode(inf2, skip_special_tokens=False)

    # Dataset level: no_reasoning + answer_prefix => the prefix ends with "Answer:" for samples with
    # and without think; answer_prefix alone (no no_reasoning) raises on samples with a think rather
    # than being silently ineffective.
    rows = [
        {"id": "a", "input_text": [USER], "gt_text": [ans], "think": REASONING},
        {"id": "b", "input_text": [USER], "gt_text": [ans], "think": ""},
    ]
    with tempfile.TemporaryDirectory() as d:
        jp = os.path.join(d, "t.jsonl")
        with open(jp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        ds = UnderstandingJsonlDataset([jp], tok, max_user_tokens=128, max_tokens=512, inference_mode=True,
                                       no_reasoning=True, answer_prefix="Answer:")
        for i in range(2):
            it = ds[i]
            assert tok.decode(it["input_ids"], skip_special_tokens=False).endswith("Answer:"), i
            assert "</think>" in tok.decode(it["input_ids"], skip_special_tokens=False)
            assert it["meta"]["ground_truth"] == ans   # the ground truth is unaffected by the prefix
        ds_bad = UnderstandingJsonlDataset([jp], tok, max_user_tokens=128, max_tokens=512, inference_mode=True,
                                           answer_prefix="Answer:")
        try:
            ds_bad[0]
            raise AssertionError("a sample with think and without no_reasoning should raise")
        except ValueError:
            pass
        assert tok.decode(ds_bad[1]["input_ids"], skip_special_tokens=False).endswith("Answer:")
        try:
            UnderstandingJsonlDataset([jp], tok, inference_mode=True, force_reasoning=True, answer_prefix="Answer:")
            raise AssertionError("answer_prefix and force_reasoning should be mutually exclusive")
        except ValueError:
            pass
    print("forced answer prefix: prefix == training prefix (ending at 'Answer:'), dataset pass-through / exclusivity / no silent no-op OK")


if __name__ == "__main__":
    test_understanding_prefix_alignment()
    test_forecast_prefix_alignment()
    test_teacher_forced_reasoning_prefix()
    test_answer_prefix_alignment()
    print("ALL OK")
