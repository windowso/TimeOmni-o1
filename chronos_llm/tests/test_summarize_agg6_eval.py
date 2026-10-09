"""Unit tests for the pure metric functions of `summarize_agg6_eval.py` (answer extraction /
accuracy / weighted-F1).

weighted-F1 is the Time-RA protocol (support-weighted), which differs from macro; checked
against a hand computation so it cannot be written the wrong way round.
"""
from chronos_llm.scripts.utils.summarize_agg6_eval import accuracy, extract, weighted_f1


def test_extract_strips_think_and_normalizes():
    assert extract("blah\n</think>\n\nAnswer: Normal Sequence.") == "normal sequence"
    assert extract("Answer: A") == "a"
    assert extract("answer:  B ") == "b"          # case-insensitive + whitespace stripped
    assert extract("<think>reasoning not finished") == ""      # only an open think, no answer span => empty
    assert extract("") == "" and extract(None) == ""
    # Without an Answer marker fall back to the full text (consistent with the official ECG extract_answer)
    assert extract("Sudden Spike Anomaly") == "sudden spike anomaly"
    print("answer extraction (strip think / case / trailing period) OK")


def test_extract_takes_last_answer_marker():
    """If 'Answer:' appears several times in the generation (reasoning restating the question),
    the last one is the real answer."""
    assert extract("The question says Answer: X. </think> Answer: Y") == "y"
    print("last Answer marker wins when there are several OK")


def test_extract_answer_first_ignores_inline_marker():
    """In answer-first mode the answer is at the **beginning**; a mid-sentence `answer:` inside the
    explanation must not steal the boundary.

    An explanation line such as "Step 6 Summarizing the thinking process to output the answer:"
    would make "the last marker" return an empty string. The criterion is therefore **the first
    marker at the start of a line**: a mid-sentence `... the answer:` is not at a line start.
    """
    gen = ("Answer: Yes\n\nStep 1 Analyzing task intent:\n[Judgment] anomaly detection.\n"
           "Step 6 Summarizing the thinking process to output the answer:\n[Judgment] Yes")
    assert extract(gen) == "yes"
    # Only an explanation with a mid-sentence marker => falls back to the "last marker" logic
    assert extract("Some text with an inline answer: Z") == "z"
    print("answer-first line-start anchoring (not stolen by a mid-sentence answer:) OK")


def _row(gt, gen):
    return {"ground_truth": f"Answer: {gt}", "generated_text": f"Answer: {gen}"}


def test_accuracy():
    rows = [_row("A", "A"), _row("B", "B"), _row("C", "D")]
    assert abs(accuracy(rows) - 200.0 / 3) < 1e-9
    assert accuracy([]) != accuracy([])          # empty => nan
    print("accuracy OK")


def test_weighted_f1_handmade():
    """Hand computation (gt/pred = a/a, a/a, a/a, b/a):

    - class a: tp=3, **fp=1** (b predicted as a), fn=0 => prec=0.75, rec=1.0, f1=6/7~0.857143
    - class b: tp=0, fp=0, fn=1 => f1=0
    - support a=3 / b=1 => weighted = (6/7*3 + 0*1)/4 = **9/14 ~ 0.642857**

    Note macro would be (6/7+0)/2 ~ 0.4286 -- the two must differ; this test fails if the
    weighted version is implemented as macro.
    """
    rows = [_row("a", "a"), _row("a", "a"), _row("a", "a"), _row("b", "a")]
    got = weighted_f1(rows, lambda s: s)
    assert abs(got - 9 / 14) < 1e-9, got
    macro = (6 / 7 + 0) / 2
    assert abs(got - macro) > 0.2, "computed as macro"
    print(f"weighted-F1 hand check OK ({got:.6f} = 9/14; macro would be {macro:.4f})")


def test_weighted_f1_binary_collapse():
    """Time-RA Label-F1: every anomaly collapses to "anomaly", leaving only normal/anomaly."""
    norm = lambda s: "normal" if "normal" in s else "anomaly"
    rows = [_row("Normal Sequence", "Normal Sequence"),
            _row("Sudden Spike Anomaly", "Trend Drift Anomaly")]   # fine class wrong, coarse class right
    assert abs(weighted_f1(rows, norm) - 1.0) < 1e-9, "should be all correct after binary collapse"
    assert weighted_f1(rows, lambda s: s) < 1.0, "must be wrong under the fine-grained protocol"
    print("Label-F1 (binary collapse) vs Action-F1 (fine-grained) protocol distinction OK")


def test_weighted_f1_all_wrong():
    rows = [_row("a", "b"), _row("a", "b")]
    assert weighted_f1(rows, lambda s: s) == 0.0
    print("weighted-F1=0 when everything is wrong OK")


def test_extract_unfinished_cot_returns_empty():
    """No `Answer:` marker => empty; an unfinished reasoning must **not** be taken as the answer.

    For a reasoning model the inference prefix stops at the open `<think>` and that token lives in
    the **prefix**, not in the generation, so "`<think>` opened but not closed" is never seen.
    Accuracy does not care (wrong either way), but weighted-F1 would create a unique pseudo-class
    for every such sample.
    """
    # The string must exceed _BARE_ANSWER_MAX_CHARS=300 to exercise the target behaviour.
    unfinished = ("**Step 1. Analyzing task intent**: This is a classification task. "
                  "**Step 2. Selecting task-relevant key patterns**: baseline consumption level; "
                  "duration of high-consumption states; frequency of low-power states; "
                  "fluctuation amplitude. Each of these is critical because laptops typically "
                  "have lower idle power than desktops, and desktops sustain high power for "
                  "extended periods during use while laptops show shorter bursts. "
                  "**Step 3. Analyzing time series samples using selected key patterns**: "
                  "the raw signal contains non-numeric characters which must first be cleaned "
                  "before the baseline can be estim")
    assert len(unfinished) > 300, f"test string length {len(unfinished)} does not exceed the threshold; target behaviour not exercised"
    assert extract(unfinished) == "", "a generation that never reaches the answer span must be empty"
    # think closed but no answer afterwards => also empty
    assert extract("reasoning...</think>   ") == ""
    # Normal answers unaffected
    assert extract("Answer: laptop") == "laptop"
    assert extract("blah </think>\nAnswer: Desktop.") == "desktop"
    # **Bare answers without a marker must be kept** and normalised as usual
    assert extract("Sudden Spike Anomaly.") == "sudden spike anomaly"
    # answer-first mode: `Answer: X\n\n<explanation>` takes only the first line, otherwise the
    # explanation becomes part of the answer
    assert extract("Answer: laptop\n\nThe baseline is low and spikes are short.") == "laptop"
    assert extract("blah</think>\nAnswer: Desktop.\n\nBecause the load is sustained.") == "desktop"
    # Boundary: a long bare answer exactly at the threshold is still an answer, beyond it empty
    assert extract("x" * 300) == "x" * 300
    assert extract("x" * 301) == ""

    # The concrete consequence of pseudo-classes blowing up the denominator: two unfinished
    # reasonings with **different content** would each become a "predicted class" for
    # weighted-F1; once mapped to empty they merge into the same (empty) class.
    rows = [{"ground_truth": "Answer: a", "generated_text": unfinished},
            {"ground_truth": "Answer: a", "generated_text": unfinished + " different tail"},
            {"ground_truth": "Answer: a", "generated_text": "Answer: a"}]
    f1 = weighted_f1(rows, lambda s: s)
    # gt has a single class a: 1 of 3 correct => prec=1/1, rec=1/3 => f1=0.5
    assert abs(f1 - 0.5) < 1e-9, f"the two unfinished rows should merge into one empty class, got {f1}"
    print("unfinished CoT generation maps to empty (accuracy unchanged, weighted-F1 no longer creates pseudo-classes) OK")


if __name__ == "__main__":
    test_extract_unfinished_cot_returns_empty()
    test_extract_strips_think_and_normalizes()
    test_extract_takes_last_answer_marker()
    test_accuracy()
    test_weighted_f1_handmade()
    test_weighted_f1_binary_collapse()
    test_weighted_f1_all_wrong()
    print("ALL OK")
