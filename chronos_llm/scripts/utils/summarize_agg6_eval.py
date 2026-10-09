"""Evaluation summary for the MMTR understanding test pool ("agg6") -- **each dataset is scored
with the native metric of its own source paper**, and placed side by side with numbers transcribed
from those papers.

Consumes the output directory of `infer_understanding.py` (one jsonl per test set, same file
name, containing `generated_text`/`ground_truth`/`task`/`scene`) and emits a markdown
comparison table.

    python chronos_llm/scripts/utils/summarize_agg6_eval.py outputs/eval/<run>/infer

**Protocol notes per source**:

- **ECG-QA / HAR / Sleep (OpenTSLM datasets)**: scored with the official-protocol scorer in
  `eval/eval_opentslm_answer.py`. ECG accuracy follows the official protocol (our input lists no
  per-template candidate answers); ECG F1 is computed globally (the official one is per-template and
  needs `template_id`). For HAR/Sleep the official loader lists all classes in the prompt, whereas the
  passed-through CSV `prompt` field is the two-way prompt of the CoT annotation stage, so the task
  input differs and the transcribed numbers are reference only.
- **ST-Bench**: MCQ accuracy on T1/T2/T3 of ST-Test. The T4 forecasting subtask is not part of the
  understanding branch (it goes through the forecasting branch).
- **VeriTime**: accuracy on the three Scenario subtasks and the four Knowledge datasets of the
  officially released test files (question text verbatim, candidate set inside the question).
- **Time-RA**: two-level weighted F1 -- **Label F1** (binary: Normal Sequence vs any anomaly) and
  **Action F1** (fine-grained type). The source paper's protocol is a text-only LLM with few-shot
  exemplars + CoT, so the transcribed numbers are reference only.
- **HiTSR**: L2/L3 accuracy on HiTSR's own test split; the source paper's main table uses an OOD
  sample of other benchmarks, so the transcribed numbers are reference only.
- **TelecomTS**: `QnA.{network,anomalies}` expanded into QA pairs, exact-match answer accuracy; the
  source paper reports per-subtask metrics at a different granularity, so no transcribed number is
  shown.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

# Share the single definition of the "line-initial Answer: marker" with the official-protocol
# extractor (the two must agree, otherwise the same generations would yield different answers
# on the two evaluation paths).
from chronos_llm.eval.eval_opentslm_answer import first_answer_line  # noqa: E402


# --------------------------- generic answer extraction ---------------------------
# Without an `Answer:` marker, anything longer than this is judged "cannot be an answer; it is an
# unfinished reasoning trace" (ground-truth answers are far shorter, unfinished reasoning far longer).
_BARE_ANSWER_MAX_CHARS = 300
def extract(text):
    """Take the content after `Answer:` and normalise it; strip the reasoning segment first (the
    answer comes after `</think>`).

    **Two "no answer was written" checks**, both required:

    1. `<think>` was opened but never closed => generation was truncated and there is no answer
       segment at all.
    2. **There is no `Answer:` marker at all** => the answer segment was likewise never reached.
       Check 1 cannot fire for a reasoning model whose inference prefix stops at the **opening**
       `<think>` (that token lives in the prefix, not in `generated_text`), so an unfinished
       reasoning trace would otherwise be taken wholesale as an "answer"; weighted F1 (Time-RA)
       would then mint a unique pseudo-class for every such sample.

    Check 2 **carries a length criterion**: a **bare answer** without `Answer:` (e.g.
    "Sudden Spike Anomaly") is legitimate input and is kept; only "no marker **and**
    implausibly long for an answer" (> `_BARE_ANSWER_MAX_CHARS`) is treated as empty.
    """
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        return ""
    line = first_answer_line(text)
    parts = re.split(r"[Aa]nswer:\s*", text)
    if line is not None:
        # **First line-anchored marker + only that line**: for an answer-first segment
        # `Answer: X\n\n<explanation>` taking the whole segment would treat the explanation as the
        # answer, and taking the **last** `Answer:` can be fooled by mid-sentence markers inside an
        # explanation (e.g. "... to output the answer:").
        s = line
    elif len(parts) > 1:
        s = parts[-1].split("\n")[0]      # no line-initial marker: fall back to the last marker
    elif len(text) > _BARE_ANSWER_MAX_CHARS:
        return ""
    else:
        s = text          # bare answer: still goes through the same normalisation below (strip special tokens / trailing dots / lowercase)
    s = re.sub(r"<\|.*?\|>|<eos>$", "", s).strip()
    return re.sub(r"[.\s]+$", "", s).strip().lower()


def accuracy(rows):
    if not rows:
        return float("nan")
    return 100.0 * sum(extract(r["ground_truth"]) == extract(r["generated_text"]) for r in rows) / len(rows)


def weighted_f1(rows, keyfn):
    """Support-weighted F1 (the Time-RA paper's metric). ``keyfn`` maps a sample to a class label."""
    tp, fp, fn, sup = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        g, p = keyfn(extract(r["ground_truth"])), keyfn(extract(r["generated_text"]))
        sup[g] += 1
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1
    total = sum(sup.values())
    if not total:
        return float("nan")
    acc = 0.0
    for c, n in sup.items():
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc += f1 * n
    return acc / total


def load(infer_dir, name):
    p = os.path.join(infer_dir, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


def by_task(rows):
    d = defaultdict(list)
    for r in rows:
        d[r.get("task")].append(r)
    return d


# --------------------- baselines (transcribed from the respective papers) ---------------------
# Each entry = (description of the comparison target, value) for protocol-matched targets.
BASELINES = {
    "ecg":        ("OpenTSLM best Flamingo/Llama3.2-3B", 46.25),
    # har/sleep live in REFERENCE_ONLY: the official HARCoTQADataset/SleepEDFCoTQADataset list
    # all classes as candidates in the prompt, while our input_text is the two-way prompt of the
    # CoT annotation stage.
    "st_t1":      ("STReasoner-8B (SOTA)", 95.65),
    "st_t2":      ("STReasoner-8B (SOTA)", 75.71),
    "st_t3":      ("STReasoner-8B (SOTA)", 87.12),
    "vt_anomaly": ("VeriTime-4B SFT+RL (SOTA)", 91.11),
    "vt_attr":    ("VeriTime-4B SFT+RL (SOTA)", 87.50),
    "vt_infer":   ("VeriTime-4B SFT+RL (SOTA)", 77.14),
    "vt_ctu":     ("VeriTime-4B (SOTA)", 67.50),
    "vt_ecg":     ("VeriTime-4B (SOTA)", 30.30),
    "vt_emg":     ("VeriTime-3B (SOTA)", 64.96),
    "vt_rcw":     ("VeriTime-3B (SOTA)", 64.89),
}
# Targets with a different protocol -- order-of-magnitude reference only
REFERENCE_ONLY = {
    "har":            ("OpenTSLM best SoftPrompt/Llama3.2-1B (all-class listing setting, not our two-way prompt)", 71.48),
    "sleep":          ("OpenTSLM best SoftPrompt/Llama3.2-1B (all-class listing setting, not our two-way prompt)", 81.08),
    "ra_uni_label":   ("Time-RA uni best Label-F1: Qwen2.5-3B zero-shot", 0.9000),
    "ra_uni_action":  ("Time-RA uni best Action-F1: Llama-3-8B SFT", 0.1511),
    "ra_multi_label": ("Time-RA multi best Label-F1: Qwen2.5-7B SFT", 0.8544),
    "ra_multi_action": ("Time-RA multi best Action-F1: Phi-4-mini SFT", 0.4372),
    "hitsr_l2":       ("LLaTiSA L2-Local (OOD sample, different questions)", 75.6),
    "hitsr_l3":       ("LLaTiSA L3 (OOD sample, different questions)", 67.0),
}


def fmt(v, nd=2):
    return "—" if v != v else f"{v:.{nd}f}"


def delta(ours, base, nd=2):
    if ours != ours:
        return "—"
    d = ours - base
    return f"{'+' if d >= 0 else ''}{d:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("infer_dir")
    ap.add_argument("--out", default=None, help="markdown output path (default: print only)")
    args = ap.parse_args(argv)
    D = args.infer_dir
    L = []

    def w(s=""):
        L.append(s)

    # --- OpenTSLM three subsets: reuse the official-protocol scorer ---
    from chronos_llm.eval.eval_opentslm_answer import detect_protocol, evaluate as ots_eval
    w("## 1. ECG-QA / HAR / Sleep (official-protocol scoring; HAR/Sleep use a different task input †)")
    w()
    w("| Subset | N | Ours Acc | Paper best Acc | Δ | Ours F1 | Paper best F1 |")
    w("|---|---|---|---|---|---|---|")
    for key, fname, f1_base in [("ecg", "test.jsonl", 40.25),
                                ("har", "har_test.jsonl", 65.44),
                                ("sleep", "sleep_test.jsonl", 69.88)]:
        rr = load(D, fname)
        if not rr:
            continue
        m = ots_eval(rr, protocol=detect_protocol(rr, fname))
        acc = 100.0 * m["accuracy"]                 # evaluate() returns a fraction in 0~1
        f1 = 100.0 * m["macro_f1_global"]
        if key == "ecg":
            who, base = BASELINES[key]
            w(f"| ECG | {m['n']} | **{fmt(acc)}** | {fmt(base)} | **{delta(acc, base)}** | "
              f"{fmt(f1)} | not comparable |")
        else:
            who, base = REFERENCE_ONLY[key]
            w(f"| {key.upper()} † | {m['n']} | **{fmt(acc)}** | {fmt(base)} † | not comparable † | "
              f"{fmt(f1)} | {fmt(f1_base)} † |")
    w()
    w("> ECG accuracy follows the official protocol; our input_text has no per-template candidate "
      "answers. ECG F1 is computed globally (the official one is per-template).")
    w(">")
    w("> † HAR/Sleep: the official loaders list all classes as candidates in the prompt, while our "
      "input_text is the two-way prompt of the CoT annotation stage; the transcribed numbers are "
      "reference only.")
    w()

    # --- ST-Bench ---
    rows = load(D, "stbench_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 2. ST-Bench / STReasoner (ST-Test)")
        w()
        w("| Subtask | N | Ours Acc | STReasoner-8B (SOTA) | Δ | Qwen3-8B+SFT | ChatTS-8B | GPT-5.2(Text) |")
        w("|---|---|---|---|---|---|---|---|")
        for tk, label, key, sft, chatts, gpt in [
                ("stbench_etiological", "T1 etiological", "st_t1", 82.13, 56.52, 83.09),
                ("stbench_entity", "T2 entity", "st_t2", 62.65, 19.51, 38.78),
                ("stbench_correlation", "T3 correlation/causality", "st_t3", 79.84, 41.08, 58.79)]:
            a = accuracy(t.get(tk, []))
            who, base = BASELINES[key]
            w(f"| {label} | {len(t.get(tk, []))} | **{fmt(a)}** | {fmt(base)} | **{delta(a, base)}** | "
              f"{fmt(sft)} | {fmt(chatts)} | {fmt(gpt)} |")
        w()

    # --- VeriTime ---
    rows = load(D, "veritime_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 3. VeriTime / TSRBench (official released test, verbatim questions)")
        w()
        w("| Family | Subtask | N | Ours Acc | VeriTime SOTA | Δ | ChatTS(SFT) | GPT-4o-mini |")
        w("|---|---|---|---|---|---|---|---|")
        for tk, fam, label, key, chatts, gpt in [
                ("veritime_Anomaly_detection", "Scenario", "Anomaly detection", "vt_anomaly", 89.44, 82.12),
                ("veritime_Scenario_attribution", "Scenario", "Scenario attribution", "vt_attr", 80.68, 61.36),
                ("veritime_Inferential_calculation", "Scenario", "Inferential calculation", "vt_infer", 72.38, 65.71)]:
            a = accuracy(t.get(tk, []))
            who, base = BASELINES[key]
            w(f"| {fam} | {label} | {len(t.get(tk, []))} | **{fmt(a)}** | {fmt(base)} | "
              f"**{delta(a, base)}** | {fmt(chatts)} | {fmt(gpt)} |")
        for tk, label, key, cls in [("veritime_CTU", "CTU", "vt_ctu", 67.20),
                                    ("veritime_ECG", "ECG", "vt_ecg", 28.39),
                                    ("veritime_EMG", "EMG", "vt_emg", 73.33),
                                    ("veritime_RCW", "RCW", "vt_rcw", 62.28)]:
            a = accuracy(t.get(tk, []))
            who, base = BASELINES[key]
            w(f"| Knowledge | {label} | {len(t.get(tk, []))} | **{fmt(a)}** | {fmt(base)} | "
              f"**{delta(a, base)}** | classical best {fmt(cls)} | — |")
        w()
        w("> The Knowledge family's \"classical best\" is the best per-task fully supervised classical "
          "model reported by the source paper.")
        w()

    # --- Time-RA ---
    rows = load(D, "time_ra_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 4. Time-RA / RATs40K (different protocol, reference only)")
        w()
        w("| Subset | N | Ours Label-F1 | Paper best Label-F1 | Ours Action-F1 | Paper best Action-F1 |")
        w("|---|---|---|---|---|---|")
        norm = lambda s: "normal" if "normal" in s else "anomaly"
        for tk, label, lk, ak in [("time_ra_uni", "Uni", "ra_uni_label", "ra_uni_action"),
                                  ("time_ra_multi", "Multi", "ra_multi_label", "ra_multi_action")]:
            rr = t.get(tk, [])
            lf = weighted_f1(rr, norm)
            af = weighted_f1(rr, lambda s: s)
            w(f"| {label} | {len(rr)} | **{fmt(lf, 4)}** | {REFERENCE_ONLY[lk][1]:.4f} | "
              f"**{fmt(af, 4)}** | {REFERENCE_ONLY[ak][1]:.4f} |")
        w()
        w("> The source paper's protocol is a text-only LLM with few-shot exemplars + CoT; the "
          "transcribed numbers are reference only.")
        w()

    # --- HiTSR ---
    rows = load(D, "hitsr_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 5. HiTSR / LLaTiSA (HiTSR's own test split; the transcribed columns use an OOD sample)")
        w()
        w("| Level | N | Ours Acc | LLaTiSA(OOD) | ChatTS(OOD) | GPT-4o Text(OOD) |")
        w("|---|---|---|---|---|---|")
        for tk, label, key, chatts, gpt in [("hitsr_l2", "L2", "hitsr_l2", 57.0, 47.6),
                                            ("hitsr_l3", "L3", "hitsr_l3", 59.0, 43.0)]:
            a = accuracy(t.get(tk, []))
            w(f"| {label} | {len(t.get(tk, []))} | **{fmt(a)}** | {REFERENCE_ONLY[key][1]:.1f} | "
              f"{fmt(chatts, 1)} | {fmt(gpt, 1)} |")
        w()
        w("> The transcribed columns are OOD-sampled questions from other benchmarks, not HiTSR's "
          "own test; reference only.")
        w()

    # --- TelecomTS ---
    rows = load(D, "telecomts_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 6. TelecomTS (exact-match QA accuracy)")
        w()
        w("| Subtask | N | Ours Acc | # distinct GT answers |")
        w("|---|---|---|---|")
        for tk, label in [("telecomts_network", "network QA"), ("telecomts_anomalies", "anomalies QA")]:
            rr = t.get(tk, [])
            n_gt = len({extract(r["ground_truth"]) for r in rr})
            flag = "  **degenerate**" if n_gt <= 1 else ""
            w(f"| {label} | {len(rr)} | **{fmt(accuracy(rr))}**{flag} | {n_gt} |")
        w()
        w("> A subtask whose test ground truth has a single distinct answer is flagged as degenerate. "
          "The source paper reports per-subtask metrics at a different granularity; we expand "
          "`QnA.{network,anomalies}` into QA and score exact match.")
        w()

    text = "\n".join(L)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n-> wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
