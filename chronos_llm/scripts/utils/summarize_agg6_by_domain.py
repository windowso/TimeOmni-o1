r"""Summarize the agg6 (MMTR understanding test pool) evaluation **per domain**, with the domain
axis aligned to the **discipline taxonomy of SciTS/TimeOmni (2510.03255)** plus extensions, giving
the 12 domains of the understanding task.

    python chronos_llm/scripts/utils/summarize_agg6_by_domain.py \
        runA=outputs/eval/runA/infer runB=outputs/eval/runB/infer \
        --detail --list

Why per domain: the source datasets are mixtures of collection sources -- e.g. VeriTime spans
physiology (ECG/EMG), energy (household electricity), bioacoustics (right-whale calls) and many
business scenarios, and Time-RA's `scene` field labels several sources. Reporting per dataset does
not reveal whether a model is strong on "physiological signals" or "IT operations".

--- Domain axis: SciTS disciplines + extensions ---

SciTS encodes the discipline in the second segment of its Release_v1 test-set file names
(`Name-Domain-TaskType-...`, see `chronos_llm/configs/understanding_test_jsonl.txt` and
`paper_understanding_domains.py`): Astronomy / Bioacoustics / Earth_Science / Economics /
Manufacturing / Meteorology / Neuroscience / Physiology / Radar / Urbanism. Two conventions are
followed from it: Sleep belongs to **Neuroscience** (`Sleep-Neuroscience-Classification-Sleep_stage`),
and HAR/activity recognition belongs to **Physiology** (`TS_MQA-Physiology-Classification-Activity`).

SciTS is a taxonomy of *scientific* disciplines and has no IT / telecom / energy engineering
domains, so extra categories are added and marked as extensions:

- `Information_Technology` (IT operations / cloud services)
- `Telecommunications` (telecom / 5G), kept separate from IT
- `Energy` (energy / power)
- `Cross_domain_Synthetic` (cross-domain / synthetic) -- HiTSR: L2 is pure shape/number description
  matching with no domain; in L3 each option describes a scenario from a different field, so a
  single sample spans multiple domains. Time-RA's `Artificial` scene falls into the same class.

--- Where each sample's domain comes from ---

1. **Time-RA**: the `scene` field, mapped to a discipline via SCENE_DOMAIN.
2. **VeriTime Scenario family** (anomaly detection / scenario attribution / reasoning computation):
   the domain is written in the question ("...collected from **Education** with length of 256");
   a regex extracts it, mapped via VT_DOMAIN. Questions without this sentence are labelled
   `Unlabeled` and excluded from the average.
3. **All other sub-tasks**: the whole sub-task has a single source, so TASK_DOMAIN is hard-coded,
   with the justification given in the per-line comments.

ST-Bench is assigned to Urbanism as a whole: its questions are "nodes + graph structure" without a
physical domain, and it is spatio-temporal graph reasoning over urban/environmental sensor networks
(same nature as SciTS's `TS_MQA-Urbanism-Anomaly_Detection-Traffic_flow`).

--- Metric convention ---

Each (sub-task x domain) cell is scored with that sub-task's **native protocol** (the ECG-QA / HAR /
Sleep subsets use their official protocol, Time-RA uses support-weighted F1, everything else uses
exact-match acc), then aggregated to the domain by sample-count weighting. Heterogeneous samples are
never pooled and re-scored, since the protocols differ. Hence a domain row is a **count-weighted
average of scores**, not a single re-computed metric. Avg = the mean of the domain scores.

The `n` column is the **distinct sample count**: one group of Time-RA samples produces two
cells (Label-F1 and Action-F1), so summing over cells would count it twice.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from chronos_llm.eval.eval_opentslm_answer import evaluate as ots_eval  # noqa: E402
from chronos_llm.scripts.utils.summarize_agg6_eval import (  # noqa: E402
    accuracy, fmt, load, weighted_f1,
)

FILES = ["test.jsonl", "har_test.jsonl", "sleep_test.jsonl", "stbench_test.jsonl",
         "veritime_test.jsonl",
         # Cleaned VeriTime test file. **Mutually exclusive** with `veritime_test.jsonl` -- an
         # infer directory only ever contains one of the two.
         "veritime_test_vtfix.jsonl",
         "hitsr_test.jsonl", "telecomts_test.jsonl", "time_ra_test.jsonl"]

PHYS, NEURO, BIO = "Physiology", "Neuroscience", "Bioacoustics"
EARTH, METEO, URBAN = "Earth_Science", "Meteorology", "Urbanism"
ECON, MANUF = "Economics", "Manufacturing"
IT, TELECOM, ENERGY, CROSS = ("Information_Technology", "Telecommunications",
                              "Energy", "Cross_domain_Synthetic")

# discipline -> (human-readable description, whether it is a native SciTS discipline)
DOMAIN_META = {
    PHYS:    ("Physiology (ECG/EMG/activity recognition)", True),
    NEURO:   ("Neuroscience (sleep staging)", True),
    BIO:     ("Bioacoustics (animal calls)", True),
    EARTH:   ("Earth science / environmental monitoring", True),
    METEO:   ("Meteorology", True),
    URBAN:   ("Urbanism (traffic / urban sensor networks)", True),
    ECON:    ("Economics and business", True),
    MANUF:   ("Manufacturing", True),
    IT:      ("IT operations / cloud services", False),
    TELECOM: ("Telecommunications / 5G", False),
    ENERGY:  ("Energy / power", False),
    CROSS:   ("Cross-domain / synthetic (no single source domain)", False),
    "Unlabeled": ("Question gives no source", False),
}
ORDER = [PHYS, NEURO, BIO, EARTH, METEO, URBAN, ECON, MANUF, IT, TELECOM, ENERGY, CROSS, "Unlabeled"]

# Single-source sub-tasks -> discipline (justification in the comments)
TASK_DOMAIN = {
    "ecg_qa_cot":   PHYS,     # OpenTSLM ECG-QA; SciTS: PTB_XL-Physiology-Classification-ECG
    "har_cot":      PHYS,     # activity recognition; SciTS: TS_MQA-Physiology-Classification-Activity
    "veritime_ECG": PHYS,     # AliveCor single-lead ECG (stated in the question)
    "veritime_EMG": PHYS,     # tibialis anterior needle-electrode EMG (stated in the question)
    "sleep_cot":    NEURO,    # SciTS: Sleep-Neuroscience-Classification-Sleep_stage
    "veritime_RCW": BIO,      # right-whale up-call audio (stated in the question); same class as SciTS's three bioacoustics sets
    "veritime_CTU": ENERGY,   # UK household electricity "Powering the Nation" (stated in the question)
    "telecomts_network":   TELECOM,
    "telecomts_anomalies": TELECOM,
    # ST-Bench: no physical domain in the question; assigned to Urbanism by judgement
    # (spatio-temporal graph reasoning over urban/environmental sensor networks), see module docstring
    "stbench_entity":      URBAN,
    "stbench_correlation": URBAN,
    "stbench_etiological": URBAN,
    "hitsr_l2": CROSS,        # pure shape/number description matching, no domain
    "hitsr_l3": CROSS,        # each of the four options describes a scenario from a different field; distractors are deliberately cross-domain
}

# Time-RA scene -> discipline
SCENE_DOMAIN = {
    "Healthcare-ECG": PHYS, "Medical-ECG": PHYS, "Sensor-Patient Behaviour Recognition": PHYS,
    "Server-YAHOO": IT, "Server-IOPS": IT, "AIOps-Amazon_CloudWatch": IT, "IoT-Utilization": IT,
    "Machine-System": MANUF,
    "Environment-Sensor": EARTH, "IoT-Drinking Water Quality": EARTH,
    "Ocean-Tropical Atmosphere": METEO,          # TAO tropical-atmosphere buoy array
    "Traffic-Traffic_Flow": URBAN,
    "Finance-Exchange": ECON, "Twitter-Ticker_Symbol": ECON,
    "Artificial": CROSS,
}

# The business scenarios in VeriTime Scenario-family questions -> discipline
VT_DOMAIN = {
    "Healthcare": PHYS,
    "Application Performance": IT, "Microservices": IT, "Kubernetes Cluster": IT,
    "Oracle Database": IT, "Redis Database": IT, "Web Servers": IT, "Network Infrastructure": IT,
    "Energy": ENERGY,
    "Weather Forecasting": METEO,
    "Environmental": EARTH, "Agriculture": EARTH,
    "Traffic and Transportation": URBAN,
    "Finance": ECON, "Retail": ECON, "Marketing and Sales": ECON, "Advertising": ECON,
    "Media and Entertainment": ECON, "Sports Analytics": ECON, "Social Media": ECON,
    "Education": ECON,
    "Manufacturing": MANUF,
}

_VT_RE = re.compile(r"collected from ([A-Za-z /&\-]+?) with length")

# Sub-task -> upstream dataset it belongs to (used by `--list`; VeriTime/Time-RA span several
# domains within one dataset, which is precisely the point)
SOURCE_OF = {
    "ecg_qa_cot": "OpenTSLM ECG-QA", "har_cot": "OpenTSLM HAR", "sleep_cot": "OpenTSLM Sleep",
    "stbench_entity": "ST-Bench", "stbench_correlation": "ST-Bench", "stbench_etiological": "ST-Bench",
    "hitsr_l2": "HiTSR (LLaTiSA)", "hitsr_l3": "HiTSR (LLaTiSA)",
    "telecomts_network": "TelecomTS", "telecomts_anomalies": "TelecomTS",
    "time_ra_uni": "Time-RA (RATs40K)", "time_ra_multi": "Time-RA (RATs40K)",
}


def source_of(task):
    return SOURCE_OF.get(task, "VeriTime (TSRBench)" if task.startswith("veritime_") else task)


def domain_of(row):
    task = row.get("task") or ""
    if task in TASK_DOMAIN:
        return TASK_DOMAIN[task]
    if task.startswith("time_ra"):
        return SCENE_DOMAIN.get(row.get("scene"), "Unlabeled")
    if task.startswith("veritime_"):           # Scenario family: the domain is in the question
        m = _VT_RE.search(row.get("input_text") or "")
        return VT_DOMAIN.get(m.group(1).strip(), "Unlabeled") if m else "Unlabeled"
    return "Unlabeled"


def cell_score(task, rows):
    """Score of one (sub-task x domain) cell, computed with the sub-task's **native protocol**.
    -> [(metric name, score)]"""
    if task in ("ecg_qa_cot", "har_cot", "sleep_cot"):
        proto = {"ecg_qa_cot": "ecg", "har_cot": "har", "sleep_cot": "sleep"}[task]
        return [("official-protocol acc", 100.0 * ots_eval(rows, proto)["accuracy"])]
    if task.startswith("time_ra"):
        # Label = binary (normal vs any anomaly); Action = fine-grained type. Same convention as the main table.
        # `weighted_f1` returns a **0-1 fraction**, whereas accuracy / the official protocol are
        # 0-100, so it is multiplied by 100 before pooling.
        return [("Label-F1", 100.0 * weighted_f1(rows, lambda s: "normal" if "normal" in s else "anomaly")),
                ("Action-F1", 100.0 * weighted_f1(rows, lambda s: s))]
    return [("acc", accuracy(rows))]


def collect(infer_dir, exclude_tasks=(), exclude_domains=()):
    """-> (cells, n_samples, sources); sub-tasks in exclude_tasks and domains in exclude_domains
    are dropped entirely.
    cells    : {domain: {(task, metric): (score, n)}}
    n_samples: {domain: distinct sample count} (must not be summed over cells -- one group of
               Time-RA samples produces two cells)
    sources  : {domain: {upstream dataset: sample count}}
    """
    cells = defaultdict(dict)
    n_samples = defaultdict(int)
    sources = defaultdict(lambda: defaultdict(int))
    for fn in FILES:
        rows = load(infer_dir, fn)
        if not rows:
            continue
        buckets = defaultdict(list)
        for r in rows:
            if r.get("task") in exclude_tasks:
                continue
            dom = domain_of(r)
            if dom in exclude_domains:
                continue
            buckets[(r.get("task"), dom)].append(r)
        for (task, dom), rs in buckets.items():
            n_samples[dom] += len(rs)
            sources[dom][source_of(task)] += len(rs)
            for metric, score in cell_score(task, rs):
                cells[dom][(task, metric)] = (score, len(rs))
    return cells, n_samples, sources


def domain_summary(cells_of_dom):
    """Domain score = **count-weighted average** of the cell scores (cells use different metrics,
    so this is an average of scores, not a re-computed metric)."""
    num = den = 0
    for score, n in cells_of_dom.values():
        if score == score:
            num += score * n
            den += n
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=infer_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--detail", action="store_true", help="print the per-cell breakdown of every domain")
    ap.add_argument("--list", action="store_true", help="print the domain -> upstream dataset / sub-task listing")
    ap.add_argument("--exclude-domains", default="",
                    help="comma-separated disciplines to drop entirely from the statistics (e.g. Energy)")
    ap.add_argument("--exclude-tasks", default="",
                    help="comma-separated sub-task names to drop entirely from the statistics (e.g. "
                         "veritime_ECG,veritime_EMG,veritime_RCW,veritime_CTU)")
    args = ap.parse_args()

    ex = tuple(t.strip() for t in args.exclude_tasks.split(",") if t.strip())
    exd = tuple(d.strip() for d in args.exclude_domains.split(",") if d.strip())
    if ex or exd:
        print(f"WARNING: excluded sub-tasks: {', '.join(ex) or '-'}; excluded domains: {', '.join(exd) or '-'}\n")
    runs = [(spec.partition("=")[0], collect(spec.partition("=")[2], ex, exd)) for spec in args.runs]
    tags = [t for t, _ in runs]
    ref_cells, ref_n, ref_src = runs[0][1]

    L = ["# agg6 evaluation summarized **per domain** (domain axis aligned to the SciTS/TimeOmni discipline taxonomy)\n"]
    L.append("SciTS is a taxonomy of scientific disciplines without IT / telecom / energy, so extra "
             "categories are added and marked `*`. "
             "Two SciTS conventions are followed: **Sleep -> Neuroscience, HAR -> Physiology**.\n")
    L.append("Each cell is scored with the sub-task's native protocol (official ECG-QA/HAR/Sleep protocol / "
             "Time-RA support-weighted F1 / exact-match acc otherwise); a domain row is the "
             "**count-weighted average** of those scores -- an average of scores, not a single metric "
             "re-computed over pooled heterogeneous samples. `n` is the **distinct sample count**.\n")

    L.append("| discipline | description | n | " + " | ".join(tags) + " |")
    L.append("|---|---|---|" + "---|" * len(tags))
    for dom in ORDER:
        if not ref_n.get(dom):
            continue
        desc, native = DOMAIN_META[dom]
        vals = [domain_summary(c.get(dom, {})) for _, (c, _, _) in runs]
        best = max((v for v in vals if v == v), default=None)
        cells = [f"**{fmt(v)}**" if best is not None and v == best else fmt(v) for v in vals]
        L.append(f"| {dom}{'' if native else ' *'} | {desc} | {ref_n[dom]} | " + " | ".join(cells) + " |")
    # Avg = the **simple mean over domain columns** ("Unlabeled" excluded; not count-weighted).
    # When excluding sub-tasks empties a whole domain the denominator changes, so comparisons must
    # use the same set of domains (the row label prints how many domains were counted).
    doms = [d for d in ORDER if ref_n.get(d) and d != "Unlabeled"]
    avgs = []
    for _, (c, _, _) in runs:
        vs = [domain_summary(c.get(d, {})) for d in doms]
        vs = [v for v in vs if v == v]
        avgs.append(sum(vs) / len(vs) if vs else float("nan"))
    best = max((v for v in avgs if v == v), default=None)
    cells = [f"**{fmt(v)}**" if best is not None and v == best else fmt(v) for v in avgs]
    L.append(f"| **Avg ({len(doms)} disciplines, equal weight)** | | | " + " | ".join(cells) + " |")
    L.append("\n`*` = domain not covered by SciTS, added by this project.")

    if args.list:
        L.append("\n## Which datasets / sub-tasks make up each domain\n")
        L.append("| discipline | upstream datasets (count) | sub-tasks (count) |")
        L.append("|---|---|---|")
        for dom in ORDER:
            if not ref_n.get(dom):
                continue
            src = ", ".join(f"{k}({v})" for k, v in sorted(ref_src[dom].items(), key=lambda x: -x[1]))
            tasks = sorted({(t, n) for (t, _), (_, n) in ref_cells[dom].items()}, key=lambda x: -x[1])
            tk = ", ".join(f"`{t}`({n})" for t, n in tasks)
            L.append(f"| {dom} | {src} | {tk} |")

    if args.detail:
        L.append("\n## Per-domain breakdown (sub-task x metric x count)\n")
        for dom in ORDER:
            if not ref_n.get(dom):
                continue
            L.append(f"\n### {dom} ({DOMAIN_META[dom][0]})\n")
            L.append("| sub-task (metric) | n | " + " | ".join(tags) + " |")
            L.append("|---|---|" + "---|" * len(tags))
            for k in sorted({k for _, (c, _, _) in runs for k in c.get(dom, {})}):
                n = next((c[dom][k][1] for _, (c, _, _) in runs if k in c.get(dom, {})), 0)
                row = [fmt(c.get(dom, {}).get(k, (float("nan"), 0))[0]) for _, (c, _, _) in runs]
                L.append(f"| {k[0]} ({k[1]}) | {n} | " + " | ".join(row) + " |")

    text = "\n".join(L)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(text + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
