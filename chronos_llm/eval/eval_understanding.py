"""Understanding-task metrics: read the JSONL produced by infer_understanding.py, compute
classification / detection metrics per dataset type, and aggregate into a CSV.

Classification/QA -> accuracy / UAR / F1 (multi-label with partial credit); detection
(Event/Anomaly Detection) -> accuracy / F1 / success rate (event detection additionally reports a
relative error). The evaluation logic follows TimeOmni-v2's eval_benchmark.py (same jsonl format),
minus the <FORECAST> token-routing evaluation that only applies to TimeOmni.
"""
import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List

import numpy as np


# detect() first checks the "false" substrings (a hit => negative), then the "true" substrings
# (a hit => positive); neither => None. Substring matching is prone to the "word containment"
# trap, so every negation anchor carries a space: no->"no " (trailing space), not->" not "
# (both sides), which avoids false hits on noise/cannot/another/notable. The anomaly-type "false"
# lists use " normal" (leading space) rather than "normal" -- otherwise "abnormal" (= anomaly, the
# positive class) contains the substring normal and is misjudged negative, i.e. the wrong
# direction; "abnormal" is added to "true" accordingly so it is judged positive. MDD "true" uses
# the stem "depress" to cover depressive/depression.
DETECTION_DICT = {
    "GWOSC GW Event": {"false": ["no ", " not "], "true": []},
    "MDD": {"false": ["healthy"], "true": ["depress"]},
    "MIMII Due": {"false": [" normal", "no ", " not "], "true": ["anomal", "abnormal"]},
    "STEAD": {"false": ["no ", " not "], "true": []},
    "TIMECAP": {"false": [" not "], "true": []},
    "TS_MQA": {"false": [" normal", "no ", " not "], "true": ["anomal", "abnormal"]},
}


def strip_think(text: str) -> str:
    """Strip the think/reasoning segment from the generated text and keep only the final answer
    (mandatory before any keyword / substring matching).

    Every decision in this module is a **substring match**, so letting the reasoning segment into
    the matched range causes systematic errors -- a casual "no significant anomaly in the first
    half" inside the reasoning is enough for detect() to hit a negation anchor and flip the
    conclusion; the `extract_predicted_times` of event detection would even treat any number in the reasoning as
    a predicted time point.

    Handles the following shapes (inference-prefix semantics of `chat_utils`):
    - ``reasoning</think>answer``: **our actual shape** -- the opening ``<think>`` lives in the
      inference prefix, not in the generated text; the model continues the reasoning body and
      closes it; take everything after the last ``</think>``.
    - ``<think>reasoning</think>answer``: fully paired, likewise take what follows the closing tag.
    - **an opening ``<think>`` without a closing tag**: the reasoning was cut off by
      max_new_tokens, so **there is no answer segment at all** => return an empty string so the
      item counts as "unanswered" (detect returns None, not counted in success_rate) instead of
      gambling on a match against half a reasoning.
    - **neither marker** (legacy TimeOmni-v2-style data without reasoning): returned unchanged.
    """
    if not text:
        return text
    if "</think>" in text:
        return text.split("</think>")[-1]
    if "<think>" in text:
        return ""  # reasoning truncated, no answer segment
    return text


def _label_hit(label: str, gen: str) -> bool:
    """Decide whether label hits gen (gen already lower-cased). Single-letter labels (MCQ options
    a/b/c/d) require non-letter boundaries on both sides to avoid substring false positives --
    e.g. "correct"/"category"/"sample" contain "c"/"a"/"a" inside the word, so naive substring
    matching would mark a GT=C row correct whenever the model says "...is correct", regardless of
    which option it actually chose). Multi-letter labels
    (descriptive classification / detection words such as "depress"/"anomal") keep plain substring
    matching -- such words are long enough that false hits are unlikely, and substring matching is
    needed to cover inflections (e.g. depress covers depressive/depression)."""
    if len(label) == 1 and label.isalpha():
        return re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", gen) is not None
    return label in gen


def classification_eval(data_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Classification/QA: accuracy, UAR, F1 (single- and multi-label)."""
    if not data_list:
        return {}
    gt_result = data_list[0].get("gt_result")
    if not isinstance(gt_result, dict):  # dirty rows with "gt_result": null -- `in` on None raises TypeError
        return {}
    if "answer" in gt_result:
        all_keys, is_qa = ["answer"], True
    elif "gt_class" in gt_result:
        all_keys, is_qa = list(gt_result["gt_class"].keys()), False
    else:
        return {}

    correct_counts = {k: 0 for k in all_keys}
    total_counts = {k: 0 for k in all_keys}
    all_labels = {k: set() for k in all_keys}

    def _unified(gr):
        if not isinstance(gr, dict):
            return None
        if is_qa and "answer" in gr:
            return {"answer": gr["answer"]}
        return gr.get("gt_class")

    for item in data_list:
        ug = _unified(item.get("gt_result"))
        if not ug:
            continue
        for k, v in ug.items():
            if isinstance(v, list):
                all_labels[k].update(x.lower() for x in v)
            else:
                all_labels[k].add(str(v).lower())

    class_tp = {k: {l: 0 for l in all_labels[k]} for k in all_keys}
    class_fn = {k: {l: 0 for l in all_labels[k]} for k in all_keys}
    class_fp = {k: {l: 0 for l in all_labels[k]} for k in all_keys}

    for item in data_list:
        gen = strip_think(item.get("generated_text", "")).lower()
        ug = _unified(item.get("gt_result"))
        if not ug:
            continue
        for k, gv in ug.items():
            total_counts[k] += 1
            predicted = {l for l in all_labels[k] if _label_hit(l, gen)}
            if isinstance(gv, list):
                gset = {x.lower() for x in gv}
                correct = 0
                for l in gset:
                    if l in predicted:
                        correct += 1; class_tp[k][l] += 1
                    else:
                        class_fn[k][l] += 1
                for l in predicted - gset:
                    class_fp[k][l] += 1
                if gset:
                    correct_counts[k] += correct / len(gset)
            else:
                gl = str(gv).lower()
                if gl in predicted:
                    correct_counts[k] += 1; class_tp[k][gl] += 1
                else:
                    class_fn[k][gl] += 1
                for l in predicted:
                    if l != gl:
                        class_fp[k][l] += 1

    results = {}
    for k in all_keys:
        suffix = "" if k in ("default", "answer") else f"_{k}"
        results[f"accuracy{suffix}"] = correct_counts[k] / total_counts[k] if total_counts[k] else 0.0
        recalls, f1s = [], []
        for l in all_labels[k]:
            tp, fn, fp = class_tp[k][l], class_fn[k][l], class_fp[k][l]
            if tp + fn > 0:                                      # only classes present in the ground truth (support>0) enter the macro average
                rec = tp / (tp + fn)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # never predicted -> precision=0 (standard macro counts 0 rather than dropping it; otherwise F1 is inflated)
                recalls.append(rec)
                f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        results[f"uar{suffix}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"f1_score{suffix}"] = float(np.mean(f1s)) if f1s else 0.0
    return results


def detect(generated_text: str, dataset_name: str):
    if dataset_name not in DETECTION_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not in DETECTION_DICT")
    cfg = DETECTION_DICT[dataset_name]
    g = generated_text.lower()
    for ind in cfg.get("false", []):
        if ind.lower() in g:
            return False
    if not cfg.get("true", []):
        return True
    for ind in cfg["true"]:
        if ind.lower() in g:
            return True
    return None


def extract_predicted_times(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def detection_eval(data_list: List[Dict[str, Any]], dataset_name: str, task: str) -> Dict[str, float]:
    if not data_list:
        return {}
    correct = total = success = 0
    tp = fp = tn = fn = 0
    rel_errs = []
    for item in data_list:
        if not isinstance(item.get("gt_result"), dict):  # skip missing or null
            continue
        gt_contain = item["gt_result"].get("contain", False)
        total += 1
        if "generated_text" not in item:
            continue
        gen_ans = strip_think(item["generated_text"])
        pred = detect(gen_ans, dataset_name)
        if pred is None:
            continue
        success += 1
        if pred == gt_contain:
            correct += 1
        if gt_contain and pred:
            tp += 1
        elif not gt_contain and pred:
            fp += 1
        elif not gt_contain and not pred:
            tn += 1
        elif gt_contain and not pred:
            fn += 1
        if task == "event detection" and gt_contain and pred:
            times = extract_predicted_times(gen_ans)
            gt_times = [(k, v) for k, v in item["gt_result"].items()
                        if k != "contain" and isinstance(v, (int, float))]
            for i, (_, gt_t) in enumerate(gt_times):
                if i < len(times) and gt_t:
                    rel_errs.append(abs(times[i] - gt_t) / abs(gt_t))
    out = {"accuracy": correct / total if total else 0.0}
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    out["f1_score"] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out["success_rate"] = success / total if total else 0.0
    if task == "event detection":
        out["mean_relative_error"] = float(np.mean(rel_errs)) if rel_errs else 0.0
        out["median_relative_error"] = float(np.median(rel_errs)) if rel_errs else 0.0
    return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_results_to_csv(results: List[Dict[str, Any]], output_csv: str):
    if not results:
        print("No results to save."); return
    results.sort(key=lambda x: x["filename"])
    all_cols = set()
    for r in results:
        all_cols.update(r.keys())
    cols = ["filename"] + sorted(c for c in all_cols if c != "filename")
    avg = {"filename": "AVERAGE"}
    for c in cols:
        if c == "filename":
            continue
        vals = [r[c] for r in results if r.get(c, "") not in ("", None)]
        if not vals:
            avg[c] = ""
        elif all(isinstance(v, int) for v in vals):
            avg[c] = sum(vals)
        else:
            try:
                fl = [float(v) for v in vals]
                avg[c] = f"{sum(fl) / len(fl):.2f}"
            except (TypeError, ValueError):
                avg[c] = ""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(results); w.writerow(avg)
    print(f"\nWrote {output_csv} ({len(results)} files)")


def evaluate_all_files(input_folder: str, output_csv: str):
    results, results_suffix = [], []
    for fn in os.listdir(input_folder):
        if not fn.endswith(".jsonl"):
            continue
        data = load_jsonl(os.path.join(input_folder, fn))
        if not data:
            print(f"  skipping empty file {fn}"); continue
        first = data[0]
        dataset_name = first.get("dataset_name", "")
        task = first.get("task", "").lower()
        row = {"filename": fn}
        # A failure on one file must not abort the whole evaluation: mid-training evaluation runs
        # inside the training process on rank 0 -- an unknown dataset_name (detect() raises) or a
        # dirty gt_result would otherwise kill the training run while the other ranks hang on the barrier.
        try:
            if task in ("classification", "qa"):
                ev = classification_eval(data)
            else:
                ev = detection_eval(data, dataset_name, task)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: metric computation failed for {fn} ({type(e).__name__}: {e}), skipping this file")
            continue
        for k, v in ev.items():
            row[k] = f"{v * 100:.2f}"
        (results if "accuracy" in row else results_suffix).append(row)
    if output_csv:
        save_results_to_csv(results, output_csv)
        if results_suffix:
            base, ext = os.path.splitext(output_csv)
            save_results_to_csv(results_suffix, f"{base}_suffix{ext}")
    else:
        print("--output not given, skipping save.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input_folder", help="directory of JSONL files produced by infer_understanding.py")
    ap.add_argument("--output", "-o", default=None, help="path of the aggregated CSV")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.input_folder):
        print(f"directory does not exist: {args.input_folder}"); raise SystemExit(1)
    evaluate_all_files(args.input_folder, args.output)


if __name__ == "__main__":
    main()
