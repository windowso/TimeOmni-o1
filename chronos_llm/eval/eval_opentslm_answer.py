r"""``Answer:``-template extraction evaluation for the OpenTSLM datasets (ECG-QA-CoT/HAR-CoT/Sleep-CoT),
following the official ``evaluation/baseline/evaluate_ecg_qa.py``.

Consumes the ``understanding/test.jsonl`` produced by `infer_understanding.py`/mid_eval (with
`generated_text`/`ground_truth` fields) and reports accuracy + macro-F1 + format-compliance rate.

Each dataset is dispatched to its own official logic (`--protocol auto` selects by `dataset_name`,
see `PROTOCOLS`):

- **ECG-QA-CoT**: `extract_answer`/`normalize_label` follow the official `evaluate_ecg_qa.py`; the
  accuracy criterion is plain exact match `int(pred_norm == gt_norm)` without any template candidate
  set (`template_id` is only used for informational `*_supported` flags). The official **F1** is
  per-template (macro-F1 within each `template_id` group, then averaged with equal weight per
  template); the parquet used here has no `template_id` column, so a **global** macro-F1 over the
  normalized GT answers is computed instead.
- **HAR-CoT**: the official case-insensitive extraction (regex `answer:\s*`; without a marker take
  the **last word**) and the fixed 8-class `allowed_labels`, with OOV predictions neither counted as
  FP nor added as classes.
- **Sleep-CoT**: the official `_canonicalize_label`, which merges "Non-REM stage 4" into "Non-REM
  stage 3" (for GT as well), with a label set discovered from the GT.

Usage:
    python chronos_llm/eval/eval_opentslm_answer.py \\
        --jsonl outputs/.../understanding/test.jsonl
Pure CPU, no GPU needed (the data is already generated text; no model is required).
"""
import argparse
import json
import os
import re
from collections import defaultdict

# The 8 classes of the official HARCoTQADataset.get_labels()
HAR_LABELS = {"sitting", "walking", "standing", "running",
              "walking_up", "walking_down", "biking", "lying"}

# **Line-anchored** answer marker: only a line that starts with `Answer:` counts; an occurrence in
# the middle of a sentence such as `... to output the answer:` does not. This supports outputs that
# place the answer first and an explanation afterwards, and explanation templates that contain a
# phrase such as "... to output the answer:" mid-sentence. For the "reasoning first, answer last"
# format there is a single line-anchored marker, so the result is unchanged. When no line-anchored
# marker is found, fall back to the official "last one" logic.
_ANSWER_LINE_RE = re.compile(r"^[ \t]*[Aa]nswer:[ \t]*(.*)$", re.MULTILINE)


def first_answer_line(text: str):
    """Return the rest of the line after the first **line-initial** `Answer:`; None if there is none
    (the caller falls back)."""
    if not text:
        return None
    m = _ANSWER_LINE_RE.search(text)
    return m.group(1) if m else None


def extract_answer(text: str) -> str:
    """Verbatim copy of the official evaluate_ecg_qa.py::extract_answer (ECG protocol), **plus taking
    only the answer line**.

    The official code only envisaged the "reasoning first, `Answer:` last" format, where everything
    after the marker is the answer. For an answer-first format `Answer: X\n\n<explanation>`, taking
    everything would yield "answer + whole explanation", which never matches the GT. Taking the first
    line is an identity transformation for the official format (nothing follows the answer).
    """
    if text is None:
        return ""
    if "Answer: " not in text:
        return text.strip()
    answer = first_answer_line(text)
    if answer is None:
        answer = text.split("Answer: ")[-1].split("\n")[0]
    answer = answer.strip()
    answer = re.sub(r"<\|.*?\|>|<eos>$", "", answer).strip()
    answer = re.sub(r"\.$", "", answer).strip()
    return answer


def extract_answer_har(text: str) -> str:
    """Verbatim copy of the official evaluate_har.py::extract_label_from_prediction.

    Two substantive differences from the ECG version: the regex ``answer:\\s*`` is
    **case-insensitive**; **without an "Answer:" marker it degrades to the last word** (instead of
    returning the whole text) -- the latter gives truncated samples a chance that "the last word
    happens to be the answer".

    As in ``extract_answer``, **only the first line** after the marker is taken, for compatibility
    with the answer-first `Answer: X\n\n<explanation>` format; an identity transformation for the
    official "reasoning first" format.
    """
    if text is None:
        return ""
    pred = text.strip()
    label = first_answer_line(pred)
    if label is None:
        match = list(re.finditer(r"answer:\s*", pred, re.IGNORECASE))
        if match:
            label = pred[match[-1].end():].split("\n")[0]
        else:
            label = pred.split()[-1] if pred.split() else ""
    label = label.strip()
    label = re.sub(r"[\.,;:!?]+$", "", label)
    return label.lower()


def canonicalize_sleep_label(text: str) -> str:
    """Same as the official sleep/parse_sleep_cot_data.py::_canonicalize_label (including the N4->N3
    merge).

    The official code applies this normalization to **both prediction and GT**, and stage 4 is merged
    into stage 3 per the AASM standard => the official label space has 5 classes rather than the 6 in
    our data; without replicating this step, "GT=N4 and the model answers N3" would be wrongly counted
    as incorrect.
    """
    if text is None:
        return ""
    cleaned = re.sub(r"<\|.*?\|>|<eos>$", "", str(text).strip()).strip()
    cleaned = re.sub(r"\.$", "", cleaned).strip()
    lowered = cleaned.lower()
    if "non-rem" in lowered or "nrem" in lowered:
        lowered = lowered.replace("nrem", "non-rem").replace("non rem", "non-rem")
    if "non-rem" in lowered and "stage 4" in lowered:
        return "non-rem stage 3"          # <- N4 merged into N3
    if "non-rem" in lowered and "stage 3" in lowered:
        return "non-rem stage 3"
    if "non-rem" in lowered and "stage 2" in lowered:
        return "non-rem stage 2"
    if "non-rem" in lowered and "stage 1" in lowered:
        return "non-rem stage 1"
    if "rem" in lowered and "sleep" in lowered:
        return "rem sleep"
    if lowered in {"wake", "awake"}:
        return "wake"
    if "movement" in lowered or lowered in {"mov", "mt"}:
        return "movement"
    return cleaned.lower()


def normalize_label(label: str) -> str:
    """Verbatim copy of the official normalize_label."""
    if label is None:
        return ""
    return label.lower().strip().rstrip(".,!?;:")


# Official protocol per dataset: extraction function + normalization function + fixed label set
# (None = discovered dynamically from the GT)
PROTOCOLS = {
    "ecg": {"extract": extract_answer, "norm": normalize_label, "labels": None},
    "har": {"extract": extract_answer_har, "norm": normalize_label, "labels": HAR_LABELS},
    "sleep": {"extract": extract_answer, "norm": canonicalize_sleep_label, "labels": None},
}


def detect_protocol(rows: list, filename: str = "") -> str:
    """Auto-select the protocol from the jsonl's dataset_name (written by the conversion script),
    falling back to keywords in the file name."""
    name = ""
    for r in rows:
        if r.get("dataset_name"):
            name = str(r["dataset_name"]).lower()
            break
    hay = f"{name} {os.path.basename(filename).lower()}"
    if "har" in hay:
        return "har"
    if "sleep" in hay:
        return "sleep"
    return "ecg"


def evaluate(rows: list, protocol: str = "ecg") -> dict:
    """Compute metrics under `protocol` (ecg/har/sleep) with that dataset's official extraction +
    normalization + label set.

    When `allowed_labels` is not None, replicate the official behaviour: **a prediction outside the
    label set is neither counted as FP nor added as a class** (only the GT class gets an FN), so the
    macro average runs over real classes only -- without this, OOV predictions would add a batch of
    f1=0 pseudo-classes to the denominator and systematically underestimate F1.
    """
    proto = PROTOCOLS.get(protocol, PROTOCOLS["ecg"])
    extract, norm, allowed = proto["extract"], proto["norm"], proto["labels"]
    if allowed is None:
        # This is what the official sleep discover_ground_truth_labels does: the label set is
        # **discovered from the GT**, predictions cannot invent classes. Otherwise a truncated or
        # off-topic generation passed through extract_answer becomes a whole-text "answer class",
        # unique per row, and the macro denominator fills with f1=0 pseudo-classes.
        allowed = {norm(extract(r.get("ground_truth", ""))) for r in rows}

    n = len(rows)
    n_correct = 0
    n_no_template = 0  # "Answer: " does not appear in the generated text at all (template not followed)
    n_oov = 0          # prediction outside the label set (meaningful only when allowed_labels is not None)
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_row = []

    for r in rows:
        gen_raw = r.get("generated_text", "")
        gt_raw = r.get("ground_truth", "")
        pred = norm(extract(gen_raw))
        gt = norm(extract(gt_raw))
        if not re.search(r"answer:\s*", gen_raw or "", re.IGNORECASE):
            n_no_template += 1
        pred_supported = (allowed is None) or (pred in allowed)
        if not pred_supported:
            n_oov += 1
        correct = pred == gt
        n_correct += int(correct)
        if correct:
            class_stats[gt]["tp"] += 1
        else:
            class_stats[gt]["fn"] += 1
            if pred_supported:  # official: OOV predictions are not counted as FP and do not create classes
                class_stats[pred]["fp"] += 1
        per_row.append({"id": r.get("id"), "pred": pred, "gt": gt, "correct": correct})

    class_f1 = {}
    for cls, c in class_stats.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
        rcl = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
        f1 = 2 * p * rcl / (p + rcl) if (p + rcl) > 0 else 0.0
        class_f1[cls] = f1
    macro_f1 = sum(class_f1.values()) / len(class_f1) if class_f1 else 0.0

    return {
        "n": n,
        "protocol": protocol,
        "accuracy": n_correct / n if n else 0.0,
        "macro_f1_global": macro_f1,
        "n_classes": len(class_f1),
        "format_compliance_rate": 1 - n_no_template / n if n else 0.0,
        "n_no_answer_marker": n_no_template,
        "n_oov_predictions": n_oov,
        "per_row": per_row,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="test.jsonl produced by mid_eval/infer_understanding")
    ap.add_argument("--dump_errors", type=int, default=0,
                    help="print the first N mispredicted samples for manual inspection")
    ap.add_argument("--protocol", default="auto", choices=["auto", "ecg", "har", "sleep"],
                    help="which dataset's official protocol to use (auto = decide from "
                         "dataset_name/file name)")
    args = ap.parse_args()

    rows = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    protocol = detect_protocol(rows, args.jsonl) if args.protocol == "auto" else args.protocol
    result = evaluate(rows, protocol=protocol)
    per_row = result.pop("per_row")
    f1_note = ("global macro-F1, not the official per-template one"
               if protocol == "ecg" else "official allowed_labels/normalization protocol")
    print(f"file: {args.jsonl}")
    print(f"protocol: {protocol}"
          + (" (official accuracy protocol; global F1)"
             if protocol == "ecg" else " (official protocol)"))
    print(f"samples: {result['n']}")
    print(f"accuracy: {result['accuracy']*100:.2f}%")
    print(f"macro_f1 ({f1_note}): {result['macro_f1_global']*100:.2f}")
    print(f"answer-template compliance rate ('Answer: ' present in the generated text): "
          f"{result['format_compliance_rate']*100:.2f}%"
          f" ({result['n_no_answer_marker']} non-compliant)")
    print(f"number of normalized answer classes: {result['n_classes']}"
          + (f"; {result['n_oov_predictions']} predictions outside the label set (not counted as FP, "
             f"as in the official code)"
             if PROTOCOLS[protocol]["labels"] is not None else ""))

    if args.dump_errors:
        print(f"\nFirst {args.dump_errors} error samples:")
        n_shown = 0
        for row in per_row:
            if not row["correct"]:
                print(f"  id={row['id']} pred={row['pred']!r} gt={row['gt']!r}")
                n_shown += 1
                if n_shown >= args.dump_errors:
                    break


if __name__ == "__main__":
    main()
