"""CPU unit tests for the reasoning-trace judge of the understanding task (no API calls; a fake client is injected).

They guard three things that would not raise on failure but would silently produce a plausible-looking wrong table:
1. **The answer/explanation split is correct for both formats** -- the answer-first explanation comes **after** the
   answer, the CoT one **before**; splitting the wrong way scores the answer as if it were the explanation;
2. **Sampling really covers wrongly answered samples** -- the correct-vs-wrong support split depends on the
   wrong-answer group; if sampling only picks correct ones that row silently disappears or is always 0;
3. **The sign of the correct-vs-wrong support gap** -- computing the gap the wrong way round flips its sign.
"""
import json
import os
import tempfile

from chronos_llm.eval.explanation_judge import (
    parse_response, split_answer_explanation, stratified_sample, summarize,
)


def test_split_both_formats():
    """Both answer-first (explanation after) and CoT (explanation before) must split correctly."""
    af = "Answer: laptop\n\nThe baseline is low and spikes are short."
    a, e = split_answer_explanation(af)
    assert a == "laptop", a
    assert "baseline is low" in e and "laptop" not in e, e

    cot = "Step 1. The baseline is low.\nStep 2. Spikes are short.\nAnswer: laptop"
    a2, e2 = split_answer_explanation(cot)
    assert a2 == "laptop", a2
    assert "Step 1" in e2 and "Step 2" in e2, e2

    # No answer marker: the whole text counts as explanation, the answer is empty (such samples must not count as "correct")
    a3, e3 = split_answer_explanation("just rambling without a verdict")
    assert a3 == "" and "rambling" in e3
    print("answer/explanation split: answer-first after, CoT before, no marker -> all explanation OK")


def _rows(n_ok, n_bad, task="t"):
    rows = [{"id": f"ok{i}", "task": task, "ground_truth": "Answer: a",
             "generated_text": "Answer: a\n\nbecause X"} for i in range(n_ok)]
    rows += [{"id": f"bad{i}", "task": task, "ground_truth": "Answer: a",
              "generated_text": "Answer: b\n\nbecause Y"} for i in range(n_bad)]
    return rows


def test_sampling_covers_wrong_answers():
    """When correct samples vastly outnumber wrong ones, sampling must still pick wrong ones (otherwise the
    correct-vs-wrong split breaks down)."""
    picked = stratified_sample(_rows(200, 20), 20, seed=0)
    n_bad = sum(1 for _, ok in picked if not ok)
    assert n_bad >= 8, f"too few wrong-answer samples picked ({n_bad}); the correct-vs-wrong split would break down"
    assert len(picked) == 20, len(picked)
    # Determinism: the same seed gives the same result twice
    assert [r["id"] for r, _ in picked] == [r["id"] for r, _ in stratified_sample(_rows(200, 20), 20, seed=0)]
    print(f"sampling covers wrong answers ({n_bad}/20) and is deterministic OK")


def test_sampling_stratifies_by_task():
    """With several sub-tasks every sub-task must be represented; small ones must not be drowned by large ones."""
    rows = _rows(100, 100, task="big") + _rows(5, 5, task="small")
    picked = stratified_sample(rows, 20, seed=0)
    tasks = {r["task"] for r, _ in picked}
    assert tasks == {"big", "small"}, f"the small sub-task was drowned out: {tasks}"
    print("stratified by sub-task, small sub-task not drowned OK")


def test_correct_wrong_support_gap():
    """Support-score gap between the correct and wrong groups: positive when the correct group scores higher."""
    records = [{"uid": "u1", "dataset": "d", "domain": "Physiology", "correct": True},
               {"uid": "u2", "dataset": "d", "domain": "Physiology", "correct": False}]
    hi_ok = [{"uid": "u1", "support": 5.0, "consistency": 4.0},
             {"uid": "u2", "support": 2.0, "consistency": 2.0}]
    lines = "\n".join(summarize(hi_ok, records))
    assert "+3.00" in lines, f"the gap must be positive when the correct group scores higher: {lines}"

    # Both groups equally high => gap 0
    same = [{"uid": "u1", "support": 4.0, "consistency": 4.0},
            {"uid": "u2", "support": 4.0, "consistency": 4.0}]
    lines2 = "\n".join(summarize(same, records))
    assert "+0.00" in lines2, lines2
    print("sign of the correct-vs-wrong support gap correct OK")


def test_summarize_group_axis():
    """Grouped by **domain** by default; `group_key='dataset'` groups by dataset."""
    records = [{"uid": "u1", "dataset": "veritime", "domain": "Energy", "correct": True},
               {"uid": "u2", "dataset": "veritime", "domain": "Bioacoustics", "correct": False}]
    scored = [{"uid": "u1", "support": 5.0, "consistency": 5.0},
              {"uid": "u2", "support": 1.0, "consistency": 1.0}]
    dom = "\n".join(summarize(scored, records))
    assert "discipline" in dom and "Energy" in dom and "Bioacoustics" in dom, dom
    ds = "\n".join(summarize(scored, records, group_key="dataset"))
    assert "dataset" in ds and "veritime" in ds and "Energy" not in ds, ds
    print("grouping axis domain/dataset switchable OK")


def test_domain_budget_sqrt_allocation():
    """Per-domain sample size is proportional to sqrt(n) and clamped to [floor, cap] -- large domains are not wasted by
    equal sampling, small domains are not drowned by proportional sampling."""
    from chronos_llm.eval.explanation_judge import domain_budget
    assert domain_budget(50000) == 250          # 2*sqrt(50000)~447 => hits the cap
    assert domain_budget(8000) == 179           # 2*sqrt(8000)~178.9
    assert domain_budget(160) == 40             # 2*sqrt(160)~25.3 => raised to the floor
    assert domain_budget(17) == 17              # fewer rows than the floor => take all
    # Monotone non-decreasing: a larger domain can never get fewer samples
    ns = [17, 160, 200, 500, 1000, 3000, 8000, 50000]
    b = [domain_budget(n) for n in ns]
    assert b == sorted(b), b
    print("domain sampling budget (sqrt(n) + floor/cap + monotone) OK")


def test_prompt_matches_protocol():
    """The judge sees the question, the model's answer and explanation, and the reference reasoning;
    the correct answer is not shown, and only support and consistency are requested."""
    from chronos_llm.eval.explanation_judge import build_messages
    rec = {"question": "Which stage?", "gt_answer": "GOLD-ANSWER-XYZ", "pred_answer": "N2",
           "explanation": "Spindles suggest N2.", "gt_reasoning": "Reference."}
    sys_msg, user = (m["content"] for m in build_messages(rec))
    assert "GOLD-ANSWER-XYZ" not in sys_msg + user
    assert '"support"' in user and '"consistency"' in user and "hallucinated" not in sys_msg + user
    print("judge prompt hides the correct answer and asks for two axes OK")


def test_parse_response():
    assert parse_response('{"support":4,"consistency":3,"reason":"x"}') \
        == {"support": 4.0, "consistency": 3.0, "reason": "x"}
    # Surrounding text is tolerated; the JSON is still extracted
    assert parse_response('sure:\n{"support":5,"consistency":5}')["support"] == 5.0
    # Unparseable => None (the caller records an error; it must not silently count as a 0 score and drag the mean down)
    assert parse_response("no json here") is None
    assert parse_response('{"support":"bad"}') is None
    print("judge response parsing (tolerant, unparseable -> None) OK")


def test_skip_empty_explanation():
    """`--skip_empty_explanation`: drop rows whose generation contains no explanation at all **before sampling**.

    A dataset evaluated in direct-answer format has no explanations; without dropping them the judge would score
    "(empty)" with the lowest support. After dropping, the domain budget is recomputed from the **number of rows
    that have an explanation**.
    """
    import contextlib
    import io

    from chronos_llm.eval.explanation_judge import main as judge_main

    with tempfile.TemporaryDirectory() as d:
        infer = os.path.join(d, "infer")
        os.makedirs(infer)
        with open(os.path.join(infer, "sleep_test.jsonl"), "w", encoding="utf-8") as f:
            for i in range(60):                      # with explanation (Neuroscience)
                f.write(json.dumps({"id": f"s{i}", "task": "sleep_cot", "input_text": "q",
                                    "ground_truth": "Answer: N1",
                                    "generated_text": "The signal shows spindles.\nAnswer: N1",
                                    "gt_reasoning": "r"}) + "\n")
        with open(os.path.join(infer, "stbench_test.jsonl"), "w", encoding="utf-8") as f:
            for i in range(30):                      # direct answer, no explanation (Urbanism)
                f.write(json.dumps({"id": f"b{i}", "task": "stbench_correlation", "input_text": "q",
                                    "ground_truth": "Answer: A", "generated_text": "Answer: A",
                                    "gt_reasoning": ""}) + "\n")

        def _run(*extra):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                judge_main([infer, "--out", os.path.join(d, "out"), "--dry_run", *extra])
            return buf.getvalue()

        plain = _run()
        assert "Urbanism" in plain.split("sampling by domain")[1], plain   # default: rows without explanation are sampled as usual
        skipped = _run("--skip_empty_explanation")
        assert "dropped rows without an explanation 30/90" in skipped, skipped
        # The whole domain is dropped => it no longer appears in the sampling list ("Urbanism" only remains in the drop statistics)
        assert "Urbanism" not in skipped.split("sampling by domain")[1], skipped
        assert "sampled 40 rows" in skipped, skipped     # only Neuroscience remains (floor=40)
    print("--skip_empty_explanation drops rows without an explanation and recomputes the budget from the rest OK")


if __name__ == "__main__":
    test_split_both_formats()
    test_sampling_covers_wrong_answers()
    test_sampling_stratifies_by_task()
    test_correct_wrong_support_gap()
    test_summarize_group_axis()
    test_domain_budget_sqrt_allocation()
    test_skip_empty_explanation()
    test_prompt_matches_protocol()
    test_parse_response()
    print("ALL OK")
