"""Convert the forecasting (T4) task of ST-Bench (STReasoner, arXiv:2601.03248) into the parquet
format the forecast branch of this project trains on (trained on its training split, scored with
the official MAE).

Data notes:
- ST-CoT/forecasting_cot.jsonl is the only split with real
  `<think>...</think><answer>[...]</answer>` reasoning chains, but the `<answer>` values in its
  `output` are the annotating LLM's own predictions (low-error candidates kept by rejection
  sampling), **not exact ground truth**, so they are not used as regression targets.
- The `idx` field of each ST-CoT row does not index ST-SFT/forecasting_finetune.jsonl. Ground truth
  is instead recovered by exact matching on the `input` text against ST-SFT, which yields the
  corresponding ground-truth `output` (a JSON list of numbers).
- `timeseries` contains only the history (truncated at the forecast origin), no future; the ground
  truth lives only in `output`.
- Within one row the multi-node histories all have identical length (no padding alignment needed),
  but the horizon K and the history length T vary per sample (K in [2,62], T in [32,400]); this
  project's `_masked_pinball` / dynamic batching support variable lengths natively.

Multi-channel mapping: `_forecast_layout` in this project hard-codes "channel row order = [targets,
known-future covariates, past-only covariates]" (the target must be the first channel row). The
original ST-Bench node numbering does not guarantee that the target node is Node 0, so this script
physically moves the target node to channel 0 and **simultaneously relabels every "Node X" reference
in the prompt/reasoning text** to "Node 0 = target, Node 1..N-1 = the remaining nodes (original
relative order preserved)" -- guaranteeing that node numbers in the text and physical array
positions are consistent in every sample (otherwise the model cannot learn a stable
number <-> physical-channel mapping).

Splits:
- split=train <- 650 ST-CoT rows (real reasoning; ground truth recovered by joining ST-SFT on the
  input text).
- split=test  <- 280 ST-Test rows (official held-out; reasoning/conclusion left empty --
  inference_mode evaluation does not need these two columns; `output` is already pure ground truth
  and is used directly as future_values).
ROI: ST-Bench has no notion of "a sub-interval of interest inside the forecast window", so the ROI
covers the entire future window (roi_loss degenerates to whole-window supervision equivalent to
pred_loss).

Usage:
  python chronos_llm/scripts/utils/build_stbench_forecast_parquet.py \
      --out data/external/stbench_t4_forecast.parquet
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ST_BENCH_DIR = Path(
    "data/raw/st-bench/ST-Bench"
)

NODE_RE = re.compile(r"Node (\d+) time series with length of (\d+)")
GRAPH_RE = re.compile(r"Graph Structure:\s*(.*?),\s*please analyze")
CONTEXT_RE = re.compile(r"Given the context (.*?), predict the value of node")
TARGET_RE = re.compile(r"predict the value of node (\d+)")
K_RE = re.compile(r"next (\d+) steps")
WINDOW_RE = re.compile(r"Historical observation window:\s*(\d+)-(\d+)")
THINK_RE = re.compile(r"<think(?:ing)?>\s*(.*?)\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_meta(input_text: str) -> dict:
    nodes = NODE_RE.findall(input_text)
    node_ids = [int(i) for i, _ in nodes]
    node_lens = [int(l) for _, l in nodes]
    m = GRAPH_RE.search(input_text)
    graph_text = m.group(1).strip() if m else ""
    m = CONTEXT_RE.search(input_text)
    context = m.group(1).strip() if m else ""
    m = TARGET_RE.search(input_text)
    if m is None:
        raise ValueError("target node not found")
    target_id = int(m.group(1))
    m = K_RE.search(input_text)
    if m is None:
        raise ValueError("forecast horizon K not found")
    K = int(m.group(1))
    m = WINDOW_RE.search(input_text)
    window = (int(m.group(1)), int(m.group(2))) if m else None
    return dict(node_ids=node_ids, node_lens=node_lens, graph_text=graph_text,
                context=context, target_id=target_id, K=K, window=window)


def relabel_node_text(text: str, remap: dict) -> str:
    """Replace 'Node <old_id>' in the text with 'Node <new_id>' (remap: old->new).
    A two-hop placeholder replacement avoids double substitution in swap maps like {0:2, 2:0}."""
    tmp = text
    for old_id in remap:
        tmp = re.sub(rf"\bNode {old_id}\b", f"__NODE_{old_id}__", tmp)
    for old_id, new_id in remap.items():
        tmp = tmp.replace(f"__NODE_{old_id}__", f"Node {new_id}")
    return tmp


def build_plain_prompt(N: int, T: int, K: int, window, context: str, graph_text_relabeled: str) -> str:
    """For the no-reasoning setting (forecast_prompt_only): built from the same variables as build_prompt,
    keeping all factual information (background / scenario / graph structure / series metadata) and
    dropping only the "reason step by step + <think> format" instruction section. The rendered prompt
    text is not re-parsed with regexes (the generic scripts/utils/build_plain_prompt.py regexes are
    written for the three-section `## Dataset background/Event/Series metadata` layout of the main
    corpus; applied to this script's own `## Scenario`/`## Graph structure` template they would drop
    the scenario and graph structure entirely, hence direct construction from the same source).
    """
    aux_desc = f"Node 1..{N - 1}" if N > 2 else "Node 1"
    if window is not None:
        win_txt = (f"indices {window[0]}-{window[1]} were flagged in the original annotation as "
                   "a recent sub-window most relevant to the near-term trend")
    else:
        win_txt = "not specified"
    parts = [
        f"A networked system of {N} time series (nodes) is linked by a directed influence graph; "
        f"a separate time-series encoder feeds you the HISTORY window values of all {N} nodes "
        f"directly, in the same Node 0..{N - 1} order.",
        f"This is synthetic spatio-temporal simulation data (network-SDE generated): nodes "
        f"influence each other through a directed, time-varying adjacency graph with propagation "
        f"delays and stochastic noise. Node 0 is the forecast target; {aux_desc} are auxiliary "
        f"nodes whose own future values are not requested but whose recent behavior, combined "
        f"with the graph structure, may inform the target's near-term dynamics.",
        f"Scenario: {context}",
        f"Graph structure (relabeled so Node 0 is always the forecast target): {graph_text_relabeled}",
        f"All {N} nodes have history length {T} steps (indices 0-{T - 1}); Node 0 (target) has a "
        f"forecast horizon of the next {K} steps (indices {T}-{T + K - 1}); {win_txt}.",
    ]
    return "\n\n".join(parts)


def build_prompt(N: int, T: int, K: int, window, context: str, graph_text_relabeled: str) -> str:
    aux_desc = f"Node 1..{N - 1}" if N > 2 else "Node 1"
    if window is not None:
        win_txt = (f"indices {window[0]}-{window[1]} (a recent sub-window the original "
                   "annotation flagged as most relevant to the near-term trend)")
    else:
        win_txt = "not specified"
    return f"""Your task is to analyze the historical dynamics of a networked system of {N} time series (nodes) linked by a directed influence graph, and forecast how the FUTURE window of the target node (Node 0) evolves. A separate time-series encoder feeds you the HISTORY window values of all {N} nodes directly, in the same Node 0..{N - 1} order used below; you are also given the graph structure and a short scenario context.

## Dataset background
This is synthetic spatio-temporal simulation data (network-SDE generated): nodes influence each other through a directed, time-varying adjacency graph with propagation delays and stochastic noise. Node 0 is the forecast target; {aux_desc} are auxiliary nodes whose own future values are not requested but whose recent behavior, combined with the graph structure, may inform the target's near-term dynamics.

## Scenario
{context}

## Graph structure (relabeled so Node 0 is always the forecast target)
{graph_text_relabeled}

## Series metadata
- All {N} nodes: history length {T} steps (indices 0-{T - 1})
- Node 0 (target): forecast horizon is the next {K} steps (indices {T}-{T + K - 1})
- Recent focus window highlighted in the original annotation: {win_txt}

## Task
Reason step by step about how the target node's own recent dynamics, the auxiliary nodes' recent behavior, and the graph's influence structure jointly determine the future window, then commit to a conclusion. Output your reasoning wrapped in <think> ... </think>, immediately followed by your final predicted values as a JSON array of exactly {K} numbers, e.g. [v1, v2, ..., v{K}]."""


def convert_row(raw: dict, *, split: str, row_id: str, gt_future: list) -> dict:
    meta = parse_meta(raw["input"])
    N = len(meta["node_ids"])
    target_id = meta["target_id"]
    if target_id not in meta["node_ids"]:
        raise ValueError(f"target_id={target_id} not in node_ids={meta['node_ids']}")
    order = [target_id] + [i for i in meta["node_ids"] if i != target_id]
    remap = {old: new for new, old in enumerate(order)}  # old_id -> new_id (new 0 is always the target)

    pos_of = {nid: k for k, nid in enumerate(meta["node_ids"])}
    ts = raw["timeseries"]
    history_new = [ts[pos_of[old_id]] for old_id in order]
    lens = {len(h) for h in history_new}
    if len(lens) != 1:
        raise ValueError(f"inconsistent node history lengths within a row: {lens}")
    T = lens.pop()

    # The ground-truth length is taken from the actual output length (the number parsed by K_RE
    # differs from the truth by 1 in rare cases, an end-of-series truncation edge case in the official
    # generation script `_compute_forecast_windows`).
    K = len(gt_future)

    graph_relabeled = relabel_node_text(meta["graph_text"], remap)
    prompt = build_prompt(N, T, K, meta["window"], meta["context"], graph_relabeled)
    plain_prompt = build_plain_prompt(N, T, K, meta["window"], meta["context"], graph_relabeled)

    reasoning, conclusion = "", ""
    if split == "train":
        m_think = THINK_RE.search(raw["output"])
        m_answer = ANSWER_RE.search(raw["output"])
        if not m_think or not m_answer:
            raise ValueError("ST-CoT row lacks <think>/<answer> tags")
        reasoning = relabel_node_text(m_think.group(1).strip(), remap)
        conclusion = relabel_node_text(m_answer.group(1).strip(), remap)

    return {
        "id": row_id,
        "dataset_name": "ST-Bench/forecasting",
        "batch": f"st_bench_{split}",
        "annotator": "Claude-4.5-Sonnet" if split == "train" else "",
        "freq": "1 step (synthetic network-SDE simulation)",
        "past_len": T,
        "future_len": K,
        "total_len": T + K,
        "history_values": [[round(float(v), 6) for v in node] for node in history_new],
        "future_values": [round(float(v), 6) for v in gt_future],
        "background": "",
        "event": "",
        "prompt": prompt,
        "plain_prompt": plain_prompt,
        "reasoning": reasoning,
        "conclusion": conclusion,
        "roi_start_idx": T,
        "roi_end_idx": T + K,
        "roi_shape": "full",
        "split": split,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--st_bench_dir", default=str(ST_BENCH_DIR))
    args = ap.parse_args()

    d = Path(args.st_bench_dir)
    sft = load_jsonl(d / "ST-SFT" / "forecasting_finetune.jsonl")
    cot = load_jsonl(d / "ST-CoT" / "forecasting_cot.jsonl")
    test = load_jsonl(d / "ST-Test" / "forecasting_test.jsonl")

    sft_input_to_output = {row["input"]: row["output"] for row in sft}

    rows = []
    n_join_fail = 0
    for i, raw in enumerate(cot):
        gt_raw = sft_input_to_output.get(raw["input"])
        if gt_raw is None:
            n_join_fail += 1
            continue
        gt = json.loads(gt_raw)
        rows.append(convert_row(raw, split="train", row_id=f"stbench_cot_{i}", gt_future=gt))

    n_test_fail = 0
    for i, raw in enumerate(test):
        try:
            gt = json.loads(raw["output"])
            rows.append(convert_row(raw, split="test", row_id=f"stbench_test_{i}", gt_future=gt))
        except Exception as e:
            n_test_fail += 1
            print(f"[warn] test row {i} failed to convert: {e}")

    print(f"ST-CoT: {len(cot)} rows, {n_join_fail} join failures")
    print(f"ST-Test: {len(test)} rows, {n_test_fail} conversion failures")

    df = pd.DataFrame(rows)
    n_train = (df["split"] == "train").sum()
    n_test = (df["split"] == "test").sum()
    print(f"final parquet: {len(df)} rows (train={n_train} / test={n_test})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
