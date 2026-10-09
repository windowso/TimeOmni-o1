"""Training + inference entry point of the no-LLM ablation arm on the MMTR understanding pool:
chronos2 + multi-task classification heads (see models/chronos_head_classifier.py).

A single GPU suffices (only the fixed-label subtasks apply; series <= 5000 points). At the
end of every epoch the weights are saved and closed-set inference is run on the test list; the
predictions are rendered as ``Answer: <surface>`` and written to `infer_ep{E}/<same name>.jsonl`
(fields generated_text/ground_truth/task/scene/id), scored with the same native protocol as the
other arms via `scripts/utils/summarize_agg6_eval.py` -- the classification-head arm and the LLM arms
stay on one axis throughout.

Usage (see also scripts/train_headcls.sh):
  python -m chronos_llm.train_headcls \
    --train_list chronos_llm/configs/understanding_train_mmtr.txt \
    --test_list  chronos_llm/configs/understanding_test_mmtr.txt \
    --output_dir outputs/baseline_headcls
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chronos_llm.data.headcls_dataset import (
    HeadClsDataset, build_label_maps, collate_headcls, norm_answer,
)
from chronos_llm.models.chronos_head_classifier import ChronosHeadClassifier
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn

DEFAULT_CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")


def read_list(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


@torch.no_grad()
def run_infer(model, test_files, label_maps, device, out_dir, bs, workers, limit=0):
    """Closed-set inference file by file -> write a jsonl of the same name (only rows of applicable subtasks, original row order kept)."""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    acc = {}
    for fp in test_files:
        ds = HeadClsDataset([fp], label_maps, limit=limit)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=bs, collate_fn=collate_headcls,
                            num_workers=workers)
        rows_out, hit, tot = [], 0, 0
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                _, preds = model(
                    batch["context"].to(device), batch["true_lengths"].to(device),
                    batch["n_channels"].to(device), batch["tasks"],
                    labels=None, candidates=batch["candidates"],
                )
            for m, p in zip(batch["meta"], preds.tolist()):
                lm = label_maps[m["task"]]
                surface = lm["surface"][lm["labels"][p]]
                rows_out.append({
                    "id": m["id"], "task": m["task"], "scene": m["scene"],
                    "generated_text": f"Answer: {surface}",
                    "ground_truth": m["gt"],
                })
                tot += 1
                hit += int(norm_answer(surface) == norm_answer(m["gt"]))
        name = os.path.basename(fp)
        with open(os.path.join(out_dir, name), "w") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        acc[name] = (hit, tot)
        print(f"[infer] {name}: {len(rows_out)} rows, rough acc={hit / max(1, tot):.4f}")
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_list", required=True)
    p.add_argument("--test_list", required=True)
    p.add_argument("--chronos_path", default=DEFAULT_CHRONOS)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=96)
    p.add_argument("--infer_bs", type=int, default=256)
    p.add_argument("--lr_chronos", type=float, default=2e-5)   # lr of the Chronos-2 encoder
    p.add_argument("--lr_head", type=float, default=1e-4)      # lr of the from-scratch classification heads
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--no_stats", action="store_true",
                   help="disable the 6-dim scale statistics features (the LLM arms enable history_stats_token; default aligned = on)")
    p.add_argument("--max_steps", type=int, default=0, help="for smoke tests: >0 truncates the number of steps per epoch")
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--limit_test", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_files, test_files = read_list(args.train_list), read_list(args.test_list)
    lm_path = os.path.join(args.output_dir, "label_maps.json")
    if os.path.exists(lm_path):
        label_maps = json.load(open(lm_path))
        print(f"[data] reusing existing label maps {lm_path}")
    else:
        label_maps = build_label_maps(train_files)
        json.dump(label_maps, open(lm_path, "w"), ensure_ascii=False, indent=1)
    for t, lm in sorted(label_maps.items()):
        print(f"[label] {t}: {len(lm['labels'])} classes")

    train_ds = HeadClsDataset(train_files, label_maps, limit=args.limit_train)
    print(f"[data] train={len(train_ds)} rows / {len(label_maps)} tasks")
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                        collate_fn=collate_headcls, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)

    chronos = load_chronos2_with_cross_attn(args.chronos_path)
    model = ChronosHeadClassifier(
        chronos, {t: lm["labels"] for t, lm in label_maps.items()},
        use_stats=not args.no_stats,
    ).to(device)
    model.chronos.train()

    decay, no_decay, head = [], [], []
    for n, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        if not n.startswith("chronos."):
            head.append(prm)
        elif prm.ndim <= 1:
            no_decay.append(prm)
        else:
            decay.append(prm)
    opt = torch.optim.AdamW([
        {"params": decay, "lr": args.lr_chronos, "weight_decay": args.weight_decay},
        {"params": no_decay, "lr": args.lr_chronos, "weight_decay": 0.0},
        {"params": head, "lr": args.lr_head, "weight_decay": args.weight_decay},
    ])
    steps_per_epoch = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warm = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warm:
            return step / max(1, warm)
        t = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    gstep = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, run_hit, run_tot = 0.0, 0, 0
        for it, batch in enumerate(loader):
            if args.max_steps and it >= args.max_steps:
                break
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss, preds = model(
                    batch["context"].to(device), batch["true_lengths"].to(device),
                    batch["n_channels"].to(device), batch["tasks"],
                    labels=labels, candidates=None,   # no candidate restriction in training: CE over all classes
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gstep += 1
            run_loss += float(loss)
            keep = labels != -100
            run_hit += int((preds[keep] == labels[keep]).sum())
            run_tot += int(keep.sum())
            if gstep % 50 == 0:
                print(f"[ep{ep} step{gstep}] loss={run_loss / max(1, it + 1):.4f} "
                      f"acc={run_hit / max(1, run_tot):.4f} lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)
        print(f"[ep{ep}] train loss={run_loss / max(1, steps_per_epoch):.4f} "
              f"acc={run_hit / max(1, run_tot):.4f}", flush=True)
        model.save_pretrained(os.path.join(args.output_dir, f"epoch_{ep}"))
        run_infer(model, test_files, label_maps, device,
                  os.path.join(args.output_dir, f"infer_ep{ep}"),
                  args.infer_bs, args.num_workers, limit=args.limit_test)
    print(f"[done] score with: python chronos_llm/scripts/utils/summarize_agg6_eval.py "
          f"{args.output_dir}/infer_ep{args.epochs}")


if __name__ == "__main__":
    main()
