"""GRPO for the understanding branch with an exact-match answer reward.

The update follows the shared GRPO utilities of grpo_trainer; only the reward differs:
- rollout (model.rollout_understanding_rl): sample G groups of reasoning + answer with
  T=1.0 / top_p=1.0 / top_k=0.
- reward[g,b] = 1{extract(gen)==extract(gt)} -- exact answer match (same answer extraction as
  `summarize_agg6_eval.extract`, i.e. the same protocol as the official scoring); the reasoning
  receives no direct supervision. If no answer segment was written (truncated / not closed),
  extract returns empty and the reward is naturally 0.
- in-group advantage = (r - mean_G)/(std_G+eps); all-correct / all-wrong groups get advantage=0
  and no gradient.
- only the LLM LoRA is updated (train_grpo._freeze_to_llm_lora); chronos / the two Q-formers stay
  frozen.
- gen_len_mean is logged: with an answer-only reward, a sudden drop in the generated length
  indicates that the policy is shortening or dropping its reasoning.
"""
import numpy as np
import torch
import torch.distributed as dist

from chronos_llm.scripts.utils.summarize_agg6_eval import extract


def _allreduce_grads(model):
    """Gradient synchronisation for multi-GPU data parallelism: average the accumulated gradients
    of the trainable parameters across ranks.

    A manual allreduce instead of the DDP wrapper -- each step performs G x (number of micro-batches)
    backward accumulations, which does not fit DDP's per-backward automatic synchronisation
    semantics (it would need no_sync workarounds); the only trainable parameters are the LLM LoRA
    (21.64M, ~87MB/step), so even gloo takes milliseconds. Parameter traversal order = module
    definition order and all ranks share the architecture, so the collective calls line up
    naturally; a parameter whose grad is None gets a zero placeholder to avoid mismatched
    collective calls across ranks. No-op when no process group is initialised (single-GPU
    behaviour unchanged)."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = dist.get_world_size()
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad)
        p.grad.div_(world)


def compute_answer_rewards(texts, batch):
    """texts: list[G] of list[str]; reward = exact match of the extracted answers. Returns a (G,B) numpy array.

    gt is taken from ground_truth in batch['meta'] (the dataset must use inference_mode=True so
    that the collator emits meta). When the extracted gt is empty (dirty data) the whole column is
    0 -- no in-group variance => no gradient, equivalent to skipping that sample."""
    metas = batch.get("meta") or []
    gts = [extract(str(m.get("ground_truth") or "")) for m in metas]
    G, B = len(texts), len(gts)
    R = np.zeros((G, B), dtype=np.float64)
    for g in range(G):
        for b in range(B):
            if gts[b]:
                R[g, b] = 1.0 if extract((texts[g][b] or "").strip()) == gts[b] else 0.0
    return R


def grpo_understanding_step(model, batch, optimizer, *, group_size=8, temperature=1.0,
                            top_p=1.0, grad_clip=1.0, max_new_tokens=2048,
                            logits_chunk=512, logp_micro_bs=0, merge_rollout=True,
                            algo="grpo"):
    """One policy-update step. Returns a metrics dict. The model must already have its non-LLM-LoRA
    parameters frozen and be set to train() by the caller.

    Three ``algo`` variants (rollout / reward / micro-batching are fully shared; only the advantage
    and the logp aggregation differ):
    - **grpo** (default): in-group standardised advantage=(R-mu)/(sigma+eps), per-sequence
      length-normalised logp.
    - **drgrpo** (Dr.GRPO): adv=R-mu (no sigma normalisation -- samples of different difficulty are
      no longer rescaled by sigma to equal weight); the summed logp is divided by the **constant**
      max_new_tokens (removing the per-sequence 1/|o| length bias, which penalises long wrong
      trajectories less and systematically encourages verbosity).
    - **rft** (rejection-sampling SFT): adv=R (0/1) -- maximum likelihood on correct trajectories
      only, no negative gradient.

    With ``logp_micro_bs>0`` the gradient forward is further split along the **batch** dimension
    into micro-batches with per-chunk backward: understanding-branch sequences are long
    (prefix+2048) and a full-batch gradient forward at bs8 keeps ~120GB of activations across the
    48 layers, a guaranteed OOM; rows are independent and share the same left padding, so chunking
    is **numerically equivalent** to the full batch (loss normalisation uses the full B*G, the
    accumulated gradient is identical). 0 = whole batch."""
    prefix_e, prefix_a, gen_ids, texts = model.rollout_understanding_rl(
        batch, group_size=group_size, temperature=temperature,
        max_new_tokens=max_new_tokens, top_p=top_p, merge_groups=merge_rollout)
    R = compute_answer_rewards(texts, batch)                     # (G, B)
    dev = prefix_e.device
    Rt = torch.as_tensor(R, dtype=torch.float32, device=dev)
    if algo == "grpo":       # in-group (across G, per sample) standardised advantage -- the GRPO core, no critic
        adv = (Rt - Rt.mean(0, keepdim=True)) / (Rt.std(0, keepdim=True) + 1e-6)  # (G,B)
    elif algo == "drgrpo":
        adv = Rt - Rt.mean(0, keepdim=True)
    elif algo == "rft":
        adv = Rt
    else:
        raise ValueError(f"unknown algo: {algo}")

    optimizer.zero_grad()
    total_loss = 0.0
    B = prefix_e.shape[0]
    mb = logp_micro_bs if logp_micro_bs and logp_micro_bs > 0 else B
    # **Per group x per micro-batch** gradient forward + backward (gradient accumulation): only one
    # micro-batch's computation graph is alive at any time.
    for g in range(group_size):
        for s in range(0, B, mb):
            e = min(s + mb, B)
            logp_sb = model.rl_logp(prefix_e[s:e], prefix_a[s:e], gen_ids[g][s:e],
                                    logits_chunk=logits_chunk,
                                    length_norm=(algo != "drgrpo"))
            if algo == "drgrpo":
                logp_sb = logp_sb / max_new_tokens   # constant normalisation (sample-independent, no length bias)
            loss_sb = -(adv[g, s:e].detach() * logp_sb).sum() / (group_size * B)
            loss_sb.backward()
            total_loss += float(loss_sb.detach())
    _allreduce_grads(model)   # multi-GPU: average gradients across ranks (single-GPU no-op); clip after the sync
    trainable = [p for p in model.parameters() if p.requires_grad]
    gn = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
    optimizer.step()

    pad_id = model.tokenizer.pad_token_id
    lens = [float((s != pad_id).sum(-1).float().mean()) if pad_id is not None
            else float(s.shape[1]) for s in gen_ids]
    return {
        "loss": total_loss,
        "reward": float(R.mean()),                          # = group-average answer accuracy
        "reward_std_in_group": float(np.mean(R.std(0))),    # zero in-group variance => no learning signal
        "mixed_frac": float(np.mean((R.sum(0) > 0) & (R.sum(0) < len(texts)))),
        "gen_len_mean": float(np.mean(lens)),               # a sudden drop = shortened reasoning
        "grad_norm": float(gn),
    }
