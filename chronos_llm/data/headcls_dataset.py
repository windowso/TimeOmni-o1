"""Data side of the no-LLM ablation on the MMTR understanding pool (chronos2 + classification
heads): list of applicable subtasks, label maps, candidate-set parsing.

Applicability criterion: **fixed question + closed label set + test vocabulary fully covered by
train** => the series alone is decidable and a classification head applies. The other subtasks
(per-question ECG-QA questions, TelecomTS observation summaries embedded in the question,
per-question free-text options of HiTSR/ST-Bench/VeriTime-Scenario, numeric answers of
Inferential_calculation) structurally depend on a text interface and are not part of this
ablation.

The official HAR/Sleep protocol is a **per-sample two-way choice** (random baseline 50%): the
candidate set is enumerated line by line in the question, and at inference a mechanical regex
extracts the candidates for a constrained argmax -- no semantic parsing of the question, no LLM
needed, but comparable with the LLM arms under the same protocol.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .understanding_dataset import _load_ts_2d

# The applicable subtasks
APPLICABLE_TASKS = (
    "har_cot", "sleep_cot",
    "time_ra_uni", "time_ra_multi",
    "veritime_Anomaly_detection", "veritime_CTU", "veritime_ECG",
    "veritime_EMG", "veritime_RCW",
)
# Subtasks whose question enumerates a per-sample candidate set (official OpenTSLM per-sample
# two-way protocol)
CANDIDATE_TASKS = ("har_cot", "sleep_cot")

_ANS_RE = re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE)


def norm_answer(text) -> str:
    if isinstance(text, list):
        text = text[0] if text else ""
    return _ANS_RE.sub("", str(text)).strip().rstrip(".").lower()


def surface_answer(text) -> str:
    """Surface form of the answer with the ``Answer:`` prefix removed and original casing kept (used
    to render generated text)."""
    if isinstance(text, list):
        text = text[0] if text else ""
    return _ANS_RE.sub("", str(text)).strip()


def build_label_maps(train_files) -> dict:
    """Stream over the train jsonl -> {task: {"labels": [norm...], "surface": {norm: original form}}}.

    labels are sorted by (frequency desc, lexical) for determinism; surface is the most frequent
    original spelling under that normalized form.
    """
    freq = defaultdict(Counter)
    surf = defaultdict(lambda: defaultdict(Counter))
    for fp in train_files:
        with open(fp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task = d.get("task")
                if task not in APPLICABLE_TASKS:
                    continue
                n = norm_answer(d.get("gt_text", ""))
                if not n:
                    continue
                freq[task][n] += 1
                surf[task][n][surface_answer(d.get("gt_text", ""))] += 1
    maps = {}
    for task, cnt in freq.items():
        labels = sorted(cnt, key=lambda a: (-cnt[a], a))
        maps[task] = {
            "labels": labels,
            "surface": {n: surf[task][n].most_common(1)[0][0] for n in labels},
        }
    return maps


def parse_candidates(question: str, labels_norm: set) -> list | None:
    """Lines of the question that are **exactly a label** => the sample's candidate set (>=2 needed
    for the parse to count as successful).

    Whole-line exact matching (allowing a leading ``-``/whitespace) prevents a label word mentioned
    incidentally inside a sentence from being picked up.
    """
    cands = []
    for line in question.splitlines():
        s = line.strip().lstrip("-").strip()
        n = s.rstrip(".").lower()
        if n in labels_norm and n not in cands:
            cands.append(n)
    return cands if len(cands) >= 2 else None


class HeadClsDataset(Dataset):
    """Reads the understanding jsonl (applicable-subtask rows only) and yields per sample a (C,T) series +
    label id + candidate ids.

    ``label_maps`` must come from the **train** files (test reuses the same mapping, closed-set
    prediction); answers in test never seen in train (defensive fallback) get label -100 and are excluded from the loss.
    """

    def __init__(self, jsonl_files, label_maps: dict, max_context: int = 8192,
                 limit: int = 0):
        self.label_maps = label_maps
        self.max_context = int(max_context)
        self.rows = []
        for fp in jsonl_files:
            with open(fp) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    task = d.get("task")
                    if task not in label_maps:
                        continue
                    lm = label_maps[task]
                    n = norm_answer(d.get("gt_text", ""))
                    label = lm["labels"].index(n) if n in lm["labels"] else -100
                    cand = None
                    if task in CANDIDATE_TASKS:
                        it = d.get("input_text")
                        q = it[0] if isinstance(it, list) else str(it)
                        names = parse_candidates(q, set(lm["labels"]))
                        if names:
                            cand = [lm["labels"].index(x) for x in names]
                    self.rows.append({
                        "task": task,
                        "label": label,
                        "cand": cand,
                        "path": (d.get("input_ts") or {}).get("original", {}).get("ori_path", ""),
                        "id": d.get("id", ""),
                        "scene": d.get("scene", ""),
                        "gt": (d.get("gt_text") or [""])[0] if isinstance(d.get("gt_text"), list)
                              else str(d.get("gt_text", "")),
                        "src": str(fp),
                    })
            if limit and len(self.rows) >= limit:
                self.rows = self.rows[:limit]
                break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        arr = _load_ts_2d(r["path"]) if r["path"] else np.zeros((1, 16), dtype=np.float32)
        arr = arr[:, -self.max_context:]                       # left-truncate, keep the most recent
        return {
            "series": torch.from_numpy(np.ascontiguousarray(arr)),
            "task": r["task"], "label": r["label"], "cand": r["cand"],
            "id": r["id"], "scene": r["scene"], "gt": r["gt"], "src": r["src"],
        }


def collate_headcls(batch):
    """Fold into ``(sum C, L)`` with left NaN padding + ``n_channels``/``true_lengths`` (same
    convention as the understanding branch)."""
    L = max(x["series"].shape[1] for x in batch)
    rows, tls, ncs = [], [], []
    for x in batch:
        s = x["series"]
        C, T = s.shape
        pad = torch.full((C, L), float("nan"))
        pad[:, L - T:] = s
        rows.append(pad)
        tls.extend([T] * C)
        ncs.append(C)
    return {
        "context": torch.cat(rows, dim=0),
        "true_lengths": torch.tensor(tls, dtype=torch.long),
        "n_channels": torch.tensor(ncs, dtype=torch.long),
        "tasks": [x["task"] for x in batch],
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "candidates": [x["cand"] for x in batch],
        "meta": [{k: x[k] for k in ("id", "task", "scene", "gt", "src")} for x in batch],
    }
