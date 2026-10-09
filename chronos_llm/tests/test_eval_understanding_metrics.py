"""CPU unit tests for the understanding classification metrics: the macro-F1 definition of
classification_eval (pure numpy / pure functions, no GPU or model needed).

Core point: standard macro-F1 computes F1 for every class that appears in the ground truth
(support>0) and averages with equal weights; a class that is never predicted (precision denominator
0) or only ever false-positive (prec=rec=0) gets F1=0 and is counted in the average -- dropping it
would inflate macro-F1.
"""
from chronos_llm.eval.eval_understanding import classification_eval, detection_eval, strip_think

# Labels are words that are not substrings of each other (detection uses substring matching `label in generated_text`).


def _item(gt_label, gen_text):
    return {"gt_result": {"gt_class": {"default": gt_label}}, "generated_text": gen_text}


def test_macro_f1_counts_unpredicted_class_as_zero():
    # The model never outputs gamma:
    #   alpha: tp=2 fn=0 fp=1 -> prec=2/3 rec=1 -> F1=0.8
    #   beta : tp=2 fn=0 fp=0 -> prec=1   rec=1 -> F1=1.0
    #   gamma: tp=0 fn=1 fp=0 -> present in the ground truth (support=1) but never predicted (tp+fp=0)
    # standard macro-F1: gamma counts as 0 -> (0.8+1.0+0)/3 = 0.6   (dropping gamma would give 0.9)
    data = [
        _item("alpha", "alpha"),
        _item("alpha", "alpha"),
        _item("beta", "beta"),
        _item("beta", "beta"),
        _item("gamma", "alpha"),   # gamma missed, one alpha false positive
    ]
    r = classification_eval(data)
    assert abs(r["f1_score"] - 0.6) < 1e-9, r["f1_score"]
    assert abs(r["uar"] - 2.0 / 3.0) < 1e-9, r["uar"]      # UAR unaffected (gamma rec=0 was already counted)
    assert abs(r["accuracy"] - 0.8) < 1e-9, r["accuracy"]
    print(f"macro-F1 counts never-predicted class as 0 OK (f1={r['f1_score']:.4f} uar={r['uar']:.4f} acc={r['accuracy']:.4f})")


def test_macro_f1_false_positive_only_class_is_zero():
    # beta is only ever a false positive, never correct: tp=0 fp=1 -> prec=0 rec=0 -> F1=0
    #   alpha: tp=2 fn=0 fp=1 -> F1=0.8
    #   beta : tp=0 fn=1 fp=1 -> F1=0
    # macro-F1 = (0.8+0)/2 = 0.4   (dropping beta would give 0.8)
    data = [
        _item("alpha", "alpha"),
        _item("alpha", "alpha beta"),   # alpha correct + beta false positive
        _item("beta", "alpha"),         # beta missed + alpha false positive
    ]
    r = classification_eval(data)
    assert abs(r["f1_score"] - 0.4) < 1e-9, r["f1_score"]
    assert abs(r["uar"] - 0.5) < 1e-9, r["uar"]
    print(f"macro-F1 counts false-positive-only class as 0 OK (f1={r['f1_score']:.4f} uar={r['uar']:.4f})")


def test_perfect_classification_unchanged():
    # All correct: every class F1=1 -> macro-F1=1; protects the normal path from the fix.
    data = [_item("alpha", "alpha"), _item("beta", "beta")]
    r = classification_eval(data)
    assert abs(r["f1_score"] - 1.0) < 1e-9, r["f1_score"]
    assert abs(r["uar"] - 1.0) < 1e-9, r["uar"]
    assert abs(r["accuracy"] - 1.0) < 1e-9, r["accuracy"]
    print(f"perfect classification macro-F1=1 OK (f1={r['f1_score']:.4f})")


def _mcq_item(gt_letter, gen_text):
    return {"gt_result": {"answer": gt_letter}, "generated_text": gen_text}


def test_mcq_single_letter_no_word_substring_false_positive():
    # Naive substring matching would judge a row with GT=C correct whenever the generated text
    # contained "is correct." (the word "correct" contains the letter c), regardless of which option
    # the model actually picked.
    # 4 rows with GT=C where the model answers A/B/C/D: only the one answering C is correct (1/4).
    data = [_mcq_item("C", "Option A is correct.") ,
            _mcq_item("C", "Option B is correct."),
            _mcq_item("C", "Option C is correct."),
            _mcq_item("C", "Option D is correct.")]
    r = classification_eval(data)
    assert abs(r["accuracy"] - 0.25) < 1e-9, r["accuracy"]
    print(f"MCQ single-letter in-word substring false positive OK (acc={r['accuracy']:.4f}, expected 0.25)")


def test_mcq_single_letter_isolated_forms_still_match():
    # Common real output formats (bare letter / parenthesised / end of a template sentence) must not be hurt by the new boundary requirement.
    data = [_mcq_item("B", "(b)"), _mcq_item("D", "D"),
            _mcq_item("A", "The sample falls under the following category: A")]
    r = classification_eval(data)
    assert abs(r["accuracy"] - 1.0) < 1e-9, r["accuracy"]
    print(f"MCQ common isolated-letter formats still match OK (acc={r['accuracy']:.4f})")


def test_strip_think_forms():
    """Three forms: closed tag -> take what follows / only an opening tag -> treated as no answer / no markers -> returned unchanged (backward compatible)."""
    # Our actual form: the opening <think> is in the inference prefix, generation starts with the reasoning body
    assert strip_think("reasoning here</think>\n\nAnswer: yes").strip() == "Answer: yes"
    # complete pair
    assert strip_think("<think>r</think>final").strip() == "final"
    # opened but not closed = reasoning truncated, no answer span
    assert strip_think("<think>reasoning got truncated") == ""
    # legacy data without reasoning: returned unchanged, behaviour preserved
    assert strip_think("No anomaly detected") == "No anomaly detected"
    assert strip_think("") == ""
    print("strip_think three forms OK")


def test_classification_ignores_reasoning_text():
    """Other candidate labels appearing in the reasoning must not pollute the final decision."""
    # GT=beta; the reasoning mentions alpha, only the answer span says beta
    items = [_item("beta", "the signal looks like alpha at first</think> Final: beta")]
    r = classification_eval(items)
    assert r["accuracy"] == 1.0, f"alpha in the reasoning polluted the decision: {r}"
    print("classification metric strips reasoning OK")


def test_detection_ignores_reasoning_negation():
    """Substring matching for detection is most vulnerable to negations inside the reasoning -- exactly the anomaly-keyword pitfall."""
    # GT has an anomaly; the reasoning says "no obvious spike in the first half", the answer span says anomaly
    item = {
        "gt_result": {"contain": True},
        "generated_text": "there is no obvious spike early on</think> Anomaly detected.",
    }
    r = detection_eval([item], "MIMII Due", "anomaly detection")
    assert r["accuracy"] == 1.0, f"the 'no ' in the reasoning flipped the verdict: {r}"
    print("detection metric strips reasoning OK")


def test_detection_times_ignore_reasoning_numbers():
    """Event-time extraction must only look at the answer span, otherwise numbers in the reasoning are taken as the prediction."""
    item = {
        "gt_result": {"contain": True, "t": 100},
        "generated_text": "I examined samples 5, 7 and 42 closely</think> Event at 100.",
    }
    r = detection_eval([item], "STEAD", "event detection")
    # the first number in the answer span is 100 = GT => relative error 0; wrongly taking 5 from the reasoning would give 0.95
    assert r["mean_relative_error"] == 0.0, f"numbers in the reasoning polluted the event-time extraction: {r}"
    print("event time strips reasoning OK")


if __name__ == "__main__":
    test_macro_f1_counts_unpredicted_class_as_zero()
    test_macro_f1_false_positive_only_class_is_zero()
    test_perfect_classification_unchanged()
    test_mcq_single_letter_no_word_substring_false_positive()
    test_mcq_single_letter_isolated_forms_still_match()
    test_strip_think_forms()
    test_classification_ignores_reasoning_text()
    test_detection_ignores_reasoning_negation()
    test_detection_times_ignore_reasoning_numbers()
    print("ALL OK")
