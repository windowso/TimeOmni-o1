"""Training logs -> png plots of each metric against step.

Two usages:
- Called automatically at the end of training (train.py, rank 0):
  ``plot_log_history(trainer.state.log_history, out_dir)``.
- Offline plotting of an existing run: ``python -m chronos_llm.scripts.utils.plot_train_logs <run_dir>``
  -- prefers ``trainer_state.json`` (output_dir or the latest checkpoint), otherwise falls back to
  parsing the ``{'loss': ...}`` dict lines in the train_*.log text (runs killed midway without a
  trainer_state can still be plotted).

One png per numeric metric (loss/text_loss/pred_loss/roi_loss/grad_norm/learning_rate/
gate_tanh_*/cuda_mem_gb/batch_* ...), written to ``{run_dir}/plots/``.
"""
import ast
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SKIP_KEYS = {"step", "epoch"}  # epoch is monotone and near-linear, not worth its own plot; step is the x axis


def plot_log_history(log_history: list, out_dir: str) -> list[str]:
    """log_history (a list of dicts shaped like trainer.state.log_history) -> one png per metric."""
    series: dict[str, list] = {}
    for rec in log_history:
        step = rec.get("step")
        if step is None:
            continue
        for k, v in rec.items():
            if k in _SKIP_KEYS or not isinstance(v, (int, float)):
                continue
            series.setdefault(k, []).append((step, v))
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for k, pts in sorted(series.items()):
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, lw=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel(k)
        ax.set_title(k)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{re.sub(r'[^A-Za-z0-9_.-]', '_', k)}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def _history_from_trainer_state(run_dir: str):
    cands = [os.path.join(run_dir, "trainer_state.json")]
    cands += sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")),
                    key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)), reverse=True)
    for p in cands:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f).get("log_history", []), p
    return None, None


_DICT_RE = re.compile(r"\{'loss':.*?\}")


def _history_from_text_logs(run_dir: str):
    """Fallback: parse the {'loss': ...} lines of train_*.log (values are strings, converted to float).
    Multiple logs (reruns / resumes) are merged in file-name time order; step cannot be recovered
    from the in-line epoch, so the line ordinal is used instead."""
    hist = []
    for lf in sorted(glob.glob(os.path.join(run_dir, "train_2*_node0.log"))):
        for line in open(lf, errors="ignore"):
            m = _DICT_RE.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError):
                continue
            rec = {}
            for k, v in d.items():
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if rec:
                rec["step"] = len(hist) + 1  # text logs have no step field; approximate by log-line order
                hist.append(rec)
    return hist


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    hist, src = _history_from_trainer_state(run_dir)
    if not hist:
        hist, src = _history_from_text_logs(run_dir), "train_*.log text fallback (step = line ordinal)"
    if not hist:
        raise SystemExit(f"no trainer_state.json or parseable train_*.log found under {run_dir}")
    out = plot_log_history(hist, os.path.join(run_dir, "plots"))
    print(f"source: {src}\nwrote {len(out)} plots -> {os.path.join(run_dir, 'plots')}")
    for p in out:
        print(" ", os.path.basename(p))


if __name__ == "__main__":
    main()
