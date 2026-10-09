"""GRPO training entry point for the understanding branch: continue training the LLM LoRA
from the SFT checkpoint with an exact-match answer reward.

Usage (via scripts/train_grpo_understanding.sh):
  python -m chronos_llm.rl.train_grpo_understanding \
    --init_ckpt outputs/<sft_run>/<checkpoint> \
    --focus_jsonl <focus pool files> --rest_jsonl <rest pool files> --pool_balance 1 \
    --output_dir outputs/grpo_understanding_run/...

Data mixing: questions are drawn from two pools of per-domain jsonl files (see
configs/grpo_focus_pools.txt / grpo_rest_pools.txt); the pool is chosen per step by a Bernoulli
draw with probability --focus_ratio, and with --pool_balance=1 every file within a pool is sampled
uniformly, so smaller domains are drawn at a higher rate than their share of the rows.
Only the LLM LoRA is updated (reusing _freeze_to_llm_lora); the data are loaded in
inference_mode (the inference prefix stops at the open <think> tag, and meta.ground_truth feeds
the reward). The adapter is saved every --save_every_steps.
"""
import argparse
import json
import os
import random

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.eval.dist_utils import barrier, init_distributed
from chronos_llm.models.chronos_llm_model import ChronosLLM, add_lora
from chronos_llm.rl.grpo_understanding import grpo_understanding_step
from chronos_llm.rl.train_grpo import _freeze_to_llm_lora, _save_ckpt


def init_policy(init_ckpt, device, *, init_merge_reopen=False, lora_r=0, lora_alpha=0):
    """Build the initial GRPO policy from the SFT checkpoint; two options:

    - **continue (default)**: `from_pretrained(merge=False, is_trainable=True)` -- keep
      training the same LoRA as SFT, so the update stays in the neighbourhood of the low-rank
      parameters SFT already settled on.
    - **init_merge_reopen** (same flag name as in the forecasting-side train.py): first merge
      the SFT LoRA into the LLM (in memory), then `add_lora` a brand-new zero-initialised LoRA --
      the update starts from zero in a fresh r-dimensional subspace (effective total capacity
      2r). Because the fresh B=0, **both options start from numerically identical policies**; the
      only difference is the parameterisation of the update. r/alpha default to the SFT
      adapter_config.json.
      Persistence goes through the `config.init_merged_from` **lineage replay** (the same
      mechanism as the forecasting side): when a saved checkpoint is reloaded,
      _maybe_replay_merged_init at the end of from_config automatically replays the SFT merge
      onto the bare base, so the new adapter is always stacked on the same base and the base
      weights are never written to disk.
      Only one level of lineage is supported -- an init_ckpt that already carries
      init_merged_from is rejected outright.
    """
    if not init_merge_reopen:
        return ChronosLLM.from_pretrained(init_ckpt, merge=False, is_trainable=True).to(device)
    from chronos_llm.models.chronos_llm_model import ChronosLLMConfig
    if getattr(ChronosLLMConfig.from_pretrained(init_ckpt), "init_merged_from", None):
        raise ValueError(f"init_merge_reopen does not support nested lineage: {init_ckpt} already carries init_merged_from")
    model = ChronosLLM.from_pretrained(init_ckpt, merge=True)
    model.config.init_merged_from = os.path.abspath(init_ckpt)
    if not (lora_r and lora_alpha):
        ac = json.load(open(os.path.join(init_ckpt, "adapter_config.json")))
        lora_r = lora_r or int(ac["r"])
        lora_alpha = lora_alpha or int(ac["lora_alpha"])
    print(f"init_merge_reopen: fresh LoRA r={lora_r} alpha={lora_alpha} (zero-initialised B => same starting point as continuing the SFT LoRA; "
          f"lineage init_merged_from={model.config.init_merged_from})")
    return add_lora(model, r=lora_r, alpha=lora_alpha, dropout=0.0).to(device)


def _loader(files, tok, args, seed, no_reasoning=False):
    """no_reasoning=True is used for the direct pool: the rollout prefix is the complete empty
    <think></think> (answer directly), matching the evaluation convention of datasets that are
    answered without a trace (e.g. ST-Bench). The reward is still exact answer match.

    --pool_balance=1: sample uniformly **by dataset** within the pool (each row weighted
    1/rows-in-its-file, with replacement). The default (0), concatenating several files and
    shuffling, follows the natural **row-count** proportion, so small datasets are rarely drawn.
    RL has sampling semantics, so repeating small datasets with replacement is harmless."""
    dkw = dict(base_dir=args.base_dir, inference_mode=True,
               max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens,
               sample_chunks=args.sample_chunks, no_reasoning=no_reasoning)
    g = torch.Generator(); g.manual_seed(seed)
    coll = ChronosLLMCollator(tokenizer=tok)
    if getattr(args, "pool_balance", 0) and len(files) > 1:
        subs = [UnderstandingJsonlDataset([p], tok, **dkw) for p in files]
        ds = ConcatDataset(subs)
        w = torch.cat([torch.full((len(s),), 1.0 / max(len(s), 1)) for s in subs])
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=g)
        return DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=0,
                          collate_fn=coll)
    ds = UnderstandingJsonlDataset(files, tok, **dkw)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                      collate_fn=coll, generator=g)


def _cycle(dl):
    while True:
        for b in dl:
            yield b


def pick_source(r, *, has_rest, has_direct, focus_ratio, direct_ratio):
    """Three-pool source selection (r in [0,1) comes from an rng seeded identically on every
    rank => synchronised selection): direct is carved off first with direct_ratio; the remainder
    is split between focus/rest by focus_ratio. Without rest, anything that is not direct is
    focus (backward compatible with single/dual pools)."""
    if has_direct and r < direct_ratio:
        return "direct"
    if not has_rest:
        return "focus"
    thr = direct_ratio + (1 - direct_ratio) * focus_ratio if has_direct else focus_ratio
    return "focus" if r < thr else "rest"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", required=True, help="SFT checkpoint (adapter + config)")
    ap.add_argument("--focus_jsonl", nargs="+", required=True, help="jsonl files of the focus pool")
    ap.add_argument("--rest_jsonl", nargs="*", default=[], help="jsonl files of the rest pool")
    ap.add_argument("--direct_jsonl", nargs="*", default=[],
                    help="direct-answer prefix pool (no_reasoning rollout; for datasets such as "
                         "ST-Bench that are evaluated under the direct-answer convention)")
    ap.add_argument("--focus_ratio", type=float, default=0.5, help="probability of drawing a focus batch at each step")
    ap.add_argument("--pool_balance", type=int, default=0,
                    help="1=sample uniformly by dataset within each pool (row weight=1/rows in file, with replacement); "
                         "0=natural row-count proportion (in a multi-file pool large datasets swamp small ones, see _loader)")
    ap.add_argument("--direct_ratio", type=float, default=0.0,
                    help="probability of drawing a direct batch at each step (decided before focus; the remainder is split by focus_ratio)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout sampling temperature")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--logits_chunk", type=int, default=512)
    ap.add_argument("--logp_micro_bs", type=int, default=2,
                    help="micro-batch size of rl_logp along the batch dim (0=whole batch; bs8 whole-batch needs ~120GB and OOMs)")
    ap.add_argument("--init_merge_reopen", type=int, default=0,
                    help="1=merge the SFT LoRA into the LLM first, then open a brand-new LoRA "
                         "(see init_policy; same flag name as the forecasting-side train.py)")
    ap.add_argument("--rollout_merge", type=int, default=1,
                    help="1=concatenate the G groups into one large batch and generate once (under the "
                         "incremental-decoding bandwidth bottleneck, xG parallelism is a free speed-up; "
                         "distribution-equivalent); 0=generate group by group (memory fallback)")
    ap.add_argument("--algo", default="grpo", choices=["grpo", "drgrpo", "rft"],
                    help="policy update algorithm: grpo=within-group standardised advantage; drgrpo="
                         "removes the length/sigma normalisation biases; rft=rejection-sampling SFT "
                         "(maximum likelihood on positives only, no negative gradient)")
    ap.add_argument("--lora_r", type=int, default=0, help="fresh LoRA r (0=reuse the SFT adapter_config)")
    ap.add_argument("--lora_alpha", type=int, default=0)
    ap.add_argument("--max_user_tokens", type=int, default=1500)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--sample_chunks", type=int, default=15, help="must match the SFT training setting (15 for the MMTR understanding pool)")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_every_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    # Multi-GPU data parallelism (active when launched with torchrun; single-process behaviour is
    # numerically unchanged): each rank draws its own batches with a different shuffle seed (RL has
    # sampling semantics, not epoch semantics, so near-independent sampling is fine), while the
    # per-step source selection (focus/rest) uses the same seed on every rank => synchronised
    # selection (avoids the waste of a step's wall time being set by the slowest rank when ranks mix
    # focus/rest differently); gradients are allreduce-averaged inside grpo_understanding_step
    # (gloo, no NCCL watchdog -- uneven rollout times are the norm); save/step logs only on rank0.
    # Effective global batch = bs x world.
    rank, world, local_rank = init_distributed()
    if world > 1 and args.device == "cuda":
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    print(f"Loading SFT checkpoint {args.init_ckpt} ... (init: "
          f"{'init_merge_reopen' if args.init_merge_reopen else 'continue the same LoRA'})")
    model = init_policy(args.init_ckpt, args.device,
                        init_merge_reopen=bool(args.init_merge_reopen),
                        lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    tok = model.tokenizer
    n_train = _freeze_to_llm_lora(model)
    print(f"GRPO trains only the LLM LoRA: trainable params {n_train/1e6:.2f}M")
    # Memory plan = **micro-batched** backward of rl_logp along the batch dim (--logp_micro_bs, see
    # grpo_understanding _step): with bs8 x (prefix + 2048) full sequences and no mitigation the
    # activations do not fit in memory. gradient_checkpointing is not used: in train mode it also
    # wraps the forward of generate (use_cache is forcibly disabled and the mask-slicing path
    # produces a non-contiguous bias, so the rollout fails with "SDPA (*bias): last dimension must
    # be contiguous"); micro-batching does not touch the attention internals and avoids the
    # recomputation overhead.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Zero all dropout: rollout sampling (train mode) and the rl_logp evaluation must be **the same
    # policy**, and dropout would make them disagree (TRL's GRPO likewise defaults to disable_dropout);
    # it also saves the per-layer (B,T,H) float32 dropout masks.
    for m in base.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    it_focus = _cycle(_loader(args.focus_jsonl, tok, args, args.seed + rank * 100))
    it_rest = (_cycle(_loader(args.rest_jsonl, tok, args, args.seed + 1 + rank * 100))
               if args.rest_jsonl else None)
    it_direct = (_cycle(_loader(args.direct_jsonl, tok, args, args.seed + 2 + rank * 100,
                                no_reasoning=True))
                 if args.direct_jsonl else None)
    rng = random.Random(args.seed)   # no rank offset: every rank picks the same source each step

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    iters = {"focus": it_focus, "rest": it_rest, "direct": it_direct}
    for step in range(1, args.max_steps + 1):
        src = pick_source(rng.random(), has_rest=it_rest is not None,
                          has_direct=it_direct is not None,
                          focus_ratio=args.focus_ratio, direct_ratio=args.direct_ratio)
        batch = next(iters[src])
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        m = grpo_understanding_step(
            model, batch, opt, group_size=args.group_size, temperature=args.temperature,
            top_p=args.top_p, max_new_tokens=args.max_new_tokens,
            logits_chunk=args.logits_chunk, logp_micro_bs=args.logp_micro_bs,
            merge_rollout=bool(args.rollout_merge), algo=args.algo)
        if rank == 0 and step % args.log_every == 0:
            print(f"step{step} [{src}] | " + " ".join(f"{k}={v:.4f}" for k, v in m.items()),
                  flush=True)   # with multiple GPUs these are rank0's own batch metrics (gradients are globally averaged, metrics are not aggregated)
        if rank == 0 and step % args.save_every_steps == 0:
            p = os.path.join(args.output_dir, f"checkpoint-{step}")
            _save_ckpt(model, p)
            print(f"saved -> {p}", flush=True)
    if rank == 0:
        _save_ckpt(model, os.path.join(args.output_dir, "final"))
        print("GRPO understanding-branch training finished")
    barrier()


if __name__ == "__main__":
    main()
