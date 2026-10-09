"""GRPO utilities shared by the GRPO entry points (``_freeze_to_llm_lora``, ``_save_ckpt``), plus a
generic GRPO loop for the forecasting branch (built on grpo_trainer.grpo_step).

Only the LLM LoRA is updated (chronos / the two Q-formers stay frozen in their SFT state); the data
uses inference_mode (inference prefix with reasoning=True + meta.gt_conclusion for the reward). The
adapter is saved every --save_every_steps.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.models.chronos_llm_model import ChronosLLM
from chronos_llm.rl.grpo_trainer import grpo_step


def _save_ckpt(model, path):
    """Save the PEFT adapter + additionally the ChronosLLMConfig (config.json) -- otherwise from_pretrained
    cannot read chronos_ckpt/llm_path at evaluation time and cannot rebuild the base (same reasoning as trainer._save)."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.config.save_pretrained(path)


def _freeze_to_llm_lora(model):
    """Unfreeze only the LLM's LoRA parameters (the GRPO policy); freeze chronos/qformer/base. Returns the number of trainable parameters."""
    n_train = 0
    for n, p in model.named_parameters():
        keep = ("lora_" in n) and ("llm." in n)
        p.requires_grad_(keep)
        if keep:
            n_train += p.numel()
    return n_train


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", required=True, help="forecasting checkpoint (adapter + config)")
    ap.add_argument("--parquet", required=True, help="forecasting parquet (the reward uses its gt_conclusion)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--w_roi", type=float, default=0.3)
    ap.add_argument("--w_mag", type=float, default=0.3)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_user_tokens", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_every_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 limits the number of steps (smoke tests / short runs)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    print(f"loading checkpoint {args.init_ckpt} ...")
    model = ChronosLLM.from_pretrained(args.init_ckpt, merge=False, is_trainable=True)
    model = model.to(args.device)
    tok = model.tokenizer
    n_train = _freeze_to_llm_lora(model)
    print(f"GRPO trains only the LLM LoRA: {n_train/1e6:.2f}M trainable parameters")

    ds = ForecastParquetDataset(args.parquet, tok, split="train", inference_mode=True,
                                max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens,
                                emit_meta=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=ChronosLLMCollator(tokenizer=tok))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            future = batch.get("future")
            if future is None or future.shape[0] == 0:
                continue
            horizon = future.shape[-1]
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            m = grpo_step(model, batch, opt, horizon=horizon, group_size=args.group_size,
                          temperature=args.temperature, w_roi=args.w_roi, w_mag=args.w_mag,
                          max_new_tokens=args.max_new_tokens)
            step += 1
            if step % args.log_every == 0:
                print(f"ep{epoch} step{step} | " + " ".join(f"{k}={v:.4f}" for k, v in m.items()),
                      flush=True)
            if step % args.save_every_steps == 0:
                p = os.path.join(args.output_dir, f"checkpoint-{step}")
                _save_ckpt(model, p)
                print(f"saved -> {p}", flush=True)
            if args.max_steps and step >= args.max_steps:
                print(f"reached max_steps={args.max_steps}; stopping early", flush=True)
                _save_ckpt(model, os.path.join(args.output_dir, "final"))
                return
    _save_ckpt(model, os.path.join(args.output_dir, "final"))
    print("GRPO training finished")


if __name__ == "__main__":
    main()
