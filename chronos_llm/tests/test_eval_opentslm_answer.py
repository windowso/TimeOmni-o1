"""Pure-function CPU unit tests for the OpenTSLM `Answer:` template extraction scorer
(chronos_llm/eval/eval_opentslm_answer.py).

Covers extract_answer/normalize_label (verbatim copies of the official evaluate_ecg_qa.py rules) and
hand-computed checks of evaluate()'s accuracy / format-compliance rate / global macro-F1.
"""
from chronos_llm.eval.eval_opentslm_answer import (
    canonicalize_sleep_label,
    detect_protocol,
    evaluate,
    extract_answer,
    extract_answer_har,
    normalize_label,
)


def test_extract_answer_matches_official_rules():
    assert extract_answer("blah blah Answer: no.") == "no"
    assert extract_answer("Answer: within the normal range") == "within the normal range"
    assert extract_answer("no Answer: marker at all here") == "marker at all here"
    assert extract_answer("only rambling, no marker") == "only rambling, no marker"
    assert extract_answer(None) == ""
    assert extract_answer("Answer: yes<|im_end|>") == "yes"
    print("extract_answer matches the official rules OK")


def test_normalize_label():
    assert normalize_label("  No.  ") == "no"
    assert normalize_label("Above the Normal Range!") == "above the normal range"
    assert normalize_label(None) == ""
    print("normalize_label OK")


def test_evaluate_accuracy_and_format_compliance():
    rows = [
        {"id": "0", "generated_text": "reasoning...\nAnswer: no", "ground_truth": "Answer: no"},
        {"id": "1", "generated_text": "reasoning...\nAnswer: yes", "ground_truth": "Answer: no"},
        {"id": "2", "generated_text": "rambling with no marker at all", "ground_truth": "Answer: no"},
    ]
    result = evaluate(rows)
    assert result["n"] == 3
    assert abs(result["accuracy"] - 1 / 3) < 1e-9  # only row 0 is correct
    assert result["n_no_answer_marker"] == 1  # row 2 has no "Answer: "
    assert abs(result["format_compliance_rate"] - 2 / 3) < 1e-9
    print("evaluate accuracy/format-compliance rate OK")


def test_evaluate_macro_f1_global():
    """The label set is discovered from the GT (as in the official sleep discover_ground_truth_labels);
    predictions must not invent classes.

    Only "no" appears in the GT => label set={no}; "yes" is OOV: it only counts as an FN for "no",
    not as an FP, and creates no new class.
      class "no": tp=1 fp=0 fn=1 -> P=1.0 R=0.5 F1=2/3
      macro_f1 = 2/3 (only 1 real class)
    If predictions could create classes, an extra "yes" class with f1=0 would pull the macro down
    to 1/3.
    """
    rows = [
        {"id": "0", "generated_text": "Answer: no", "ground_truth": "Answer: no"},
        {"id": "1", "generated_text": "Answer: yes", "ground_truth": "Answer: no"},
    ]
    result = evaluate(rows)
    assert abs(result["macro_f1_global"] - 2 / 3) < 1e-9, result["macro_f1_global"]
    assert result["n_classes"] == 1, f"an OOV prediction must not create a class; got {result['n_classes']}"
    print("evaluate macro_f1_global hand-computed check (label set discovered from GT) OK")


def test_har_official_extraction():
    """The two substantive differences of the official HAR extraction from the ECG version:
    case-insensitive + last word when there is no marker."""
    # case-insensitive (the ECG version only recognises "Answer: ")
    assert extract_answer_har("blah answer: sitting") == "sitting"
    assert extract_answer_har("blah ANSWER: Walking") == "walking"
    # without an "Answer:" marker take the last word (the ECG version returns the whole text)
    assert extract_answer_har("the subject appears to be running") == "running"
    assert extract_answer("the subject appears to be running") == "the subject appears to be running"
    # trailing punctuation stripped
    assert extract_answer_har("Answer: biking.") == "biking"
    assert extract_answer_har("") == ""
    print("HAR official extraction (case-insensitive + last-word fallback) OK")


def test_sleep_canonicalize_merges_stage4_into_stage3():
    """The official scorer merges Non-REM stage 4 into stage 3 per AASM -- without replicating this,
    GT=N4 / answer N3 would be scored as wrong."""
    assert canonicalize_sleep_label("Non-REM stage 4") == "non-rem stage 3"
    assert canonicalize_sleep_label("Non-REM stage 3") == "non-rem stage 3"
    assert canonicalize_sleep_label("non-rem stage 4") == canonicalize_sleep_label("NREM stage 3")
    assert canonicalize_sleep_label("Non-REM stage 2") == "non-rem stage 2"
    assert canonicalize_sleep_label("REM sleep") == "rem sleep"
    assert canonicalize_sleep_label("Wake") == "wake"
    assert canonicalize_sleep_label("awake") == "wake"
    print("Sleep official canonicalisation (N4->N3 merge) OK")


def test_sleep_protocol_scores_stage4_as_stage3():
    """End to end: GT=Non-REM stage 4, model answers Non-REM stage 3, the official protocol scores it correct."""
    rows = [{"id": "0", "generated_text": "Answer: Non-REM stage 3",
             "ground_truth": "Answer: Non-REM stage 4"}]
    assert evaluate(rows, protocol="sleep")["accuracy"] == 1.0, "N4/N3 merge not in effect"
    # under the ECG protocol (no merge) the same row is scored wrong
    assert evaluate(rows, protocol="ecg")["accuracy"] == 0.0
    print("Sleep protocol end to end (N4 verdict flips) OK")


def test_har_fixed_labels_vs_ecg_discovered_labels():
    """HAR uses the official fixed 8 classes; ECG/Sleep discover the label set from the GT -- the
    difference shows up when "a legal class that never appears in the GT" is predicted.

    The GT of the two rows only contains {sitting, walking}, and the model answers "running" on row 2:
    - HAR (fixed 8 classes): running is a legal label => counts as an FP and enters the class set (3 classes);
    - ECG (GT-discovered): running is not in the GT label set => treated as OOV, no FP, no new class (2 classes).
    Neither lets "completely illegal garbage output" create classes.
    """
    rows = [
        {"id": "0", "generated_text": "Answer: sitting", "ground_truth": "Answer: sitting"},
        {"id": "1", "generated_text": "Answer: running", "ground_truth": "Answer: walking"},
    ]
    har = evaluate(rows, protocol="har")
    assert har["n_oov_predictions"] == 0, "running is a legal HAR label and must not count as OOV"
    assert har["n_classes"] == 3, f"a legal non-GT prediction should count as an FP and create a class; got {har['n_classes']}"

    ecg = evaluate(rows, protocol="ecg")
    assert ecg["n_oov_predictions"] == 1, "running never appears in the GT => OOV under the dynamic label set"
    assert ecg["n_classes"] == 2

    # completely illegal garbage output (a truncation artefact) must not create classes under either protocol
    rows_junk = [
        {"id": "0", "generated_text": "Answer: sitting", "ground_truth": "Answer: sitting"},
        {"id": "1", "generated_text": "the accelerometer trace shows a gradual", "ground_truth": "Answer: walking"},
    ]
    for proto in ("har", "ecg"):
        res = evaluate(rows_junk, protocol=proto)
        assert res["n_oov_predictions"] == 1, proto
        assert res["n_classes"] == 2, f"{proto} let truncated text create a pseudo-class: {res['n_classes']}"
    print("HAR fixed label set vs ECG dynamic label set + pseudo-class guard OK")


def test_detect_protocol():
    assert detect_protocol([{"dataset_name": "opentslm_har_cot"}]) == "har"
    assert detect_protocol([{"dataset_name": "opentslm_sleep_cot"}]) == "sleep"
    assert detect_protocol([{"dataset_name": "opentslm_ecg_qa_cot"}]) == "ecg"
    assert detect_protocol([{}], "/x/har_test.jsonl") == "har"   # falls back to the file name
    assert detect_protocol([{}], "/x/test.jsonl") == "ecg"
    print("automatic protocol detection OK")


def test_answer_first_format_only_takes_answer_line():
    """`Answer: X\\n\\n<explanation>` (answer-first format) must yield only the answer line.

    The official scorer only envisaged "reasoning first, Answer last", where taking everything after
    `Answer:` is fine; for the answer-first format that would extract "answer + whole explanation",
    which never matches the GT.
    """
    af = "Answer: no\n\nThe ECG shows baseline drift and electrode artifacts, which are common."
    assert extract_answer(af) == "no", extract_answer(af)
    assert extract_answer_har("Answer: Walking\n\nThe accelerometer shows periodic peaks.") == "walking"

    # must be the **identity** on the official format (reasoning first, Answer last) -- guards against "fixing the new format breaks the old one"
    official = "Step 1 ... Step 2 ...\nAnswer: atrial fibrillation"
    assert extract_answer(official) == "atrial fibrillation"
    assert extract_answer_har("reasoning here\nAnswer: Sitting") == "sitting"
    # trailing punctuation / special tokens are still cleaned as usual
    assert extract_answer("Answer: none.\n\nbecause ...") == "none"
    print("answer-first `Answer: X\\n<explanation>` yields only the answer line and is the identity on the official format OK")


if __name__ == "__main__":
    test_answer_first_format_only_takes_answer_line()
    test_extract_answer_matches_official_rules()
    test_normalize_label()
    test_evaluate_accuracy_and_format_compliance()
    test_evaluate_macro_f1_global()
    test_har_official_extraction()
    test_sleep_canonicalize_merges_stage4_into_stage3()
    test_sleep_protocol_scores_stage4_as_stage3()
    test_har_fixed_labels_vs_ecg_discovered_labels()
    test_detect_protocol()
    print("ALL OK")
