"""Training entry point for Chronos-2 + Qwen3.5 (torchrun + HF Trainer + DeepSpeed ZeRO-2).

See chronos_llm/scripts/train_understanding.sh and chronos_llm/scripts/train_forecast.sh for usage.
"""

import argparse
import importlib.util
import os

from transformers import TrainingArguments

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import ConcatBranchDataset
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.eval.mid_eval import MidTrainEvalCallback
from chronos_llm.models.chronos_llm_model import (ChronosLLM, ChronosLLMConfig, add_lora,
                                                  freeze_ts_modules)
from chronos_llm.trainer import ChronosLLMTrainer


def parse_args():
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--chronos_ckpt", required=True)
    p.add_argument("--llm_path", required=True)
    p.add_argument("--forecast_parquet", default=None)
    p.add_argument("--understanding_jsonl", nargs="*", default=[])
    p.add_argument("--understanding_base_dir", default=None)
    p.add_argument("--understanding_no_reasoning", action="store_true",
                   help="no-CoT control arm: ignore the jsonl `think` field and supervise only the answer "
                        "(no <think>reasoning</think>). The mid-training eval inference prefix switches "
                        "accordingly to 'complete empty think, continue with the answer only', so both sides "
                        "stay consistent. When evaluating a trained no-CoT model, pass the same flag to "
                        "infer_understanding.py.")
    p.add_argument("--understanding_answer_first", action="store_true",
                   help="answer-first mode: supervised segment = `answer\\n\\nexplanation` -- "
                        "the explanation is appended **after** the answer, not wrapped in <think>; the think "
                        "segment stays empty. The inference prefix is token-identical to no-CoT, so "
                        "**evaluate with NO_REASONING=1 directly** (no extra flag); metrics look only at the "
                        "first line after `Answer:`. Mutually exclusive with --understanding_no_reasoning.")
    p.add_argument("--output_dir", required=True)
    # Model / Q-former
    # Soft prompt budget (current scheme = log-linear token budget, see SlidingWindowQFormer.plan):
    #   total tokens = global G + local nw*k, where nw is solved from
    #   T = clamp(token_min + b*log2(P x C), token_min, token_max) (b anchored by token_pc_ref).
    #   The token count grows logarithmically with history length and is capped at token_max (default 200);
    #   knobs: --sw_token_min/max/pc_ref + --sw_queries_per_window (k) + --sw_global_queries (G).
    p.add_argument("--sw_queries_per_window", type=int, default=8, help="Number of query tokens k each local window is compressed into")
    # NOTE: the next two no longer affect window planning; kept only for backward compatibility with old checkpoint
    # configs (plan() no longer reads them; changing them changes no output):
    p.add_argument("--sw_target_windows", type=int, default=8,
                   help="[DEPRECATED, backward compat only] window-count cap of the old ladder scheme; the token budget is now controlled by --sw_token_max")
    p.add_argument("--sw_min_windows", type=int, default=4,
                   help="[DEPRECATED, backward compat only] window-count floor of the old ladder scheme; plan() no longer reads it")
    p.add_argument("--sw_global_queries", type=int, default=16,
                   help="Number of global queries G: cross-attend over the patches of the whole history to add cross-window global context, prepended to the local tokens; 0=off")
    p.add_argument("--sw_token_min", type=int, default=40, help="Lower bound on the soft prompt token count (clamp floor of the log-linear budget)")
    p.add_argument("--sw_token_max", type=int, default=200, help="Upper bound on the soft prompt token count (cap)")
    p.add_argument("--sw_token_pc_ref", type=int, default=32768,
                   help="P x C at which the token count reaches token_max (slope anchor of the log-linear budget)")
    p.add_argument("--chronos_window", type=int, default=0, help="Chronos-2 sliding-window chunk size; 0=use its context_length")
    p.add_argument("--encode_max_rows", type=int, default=58,
                   help="Max rows per encode call in sliding-window encoding (samples x channels x chunks; group-attn cost is O(R^2) -> linear after bucketing); 0=no bucketing")
    # Short-window detection boosters (see ChronosLLMConfig). Off by default; aimed at detection sets such as
    # Weather ('single channel, 16 points, 1 patch').
    p.add_argument("--history_stats_token", action="store_true",
                   help="Project per-history (per-channel) window-scale statistics [loc, scale, range, max|x|, ...] through a small MLP into a statistics token appended to the soft prompt "
                        "(restores the scale/range discriminative signal that instance_norm removes)")
    p.add_argument("--short_min_patches", type=int, default=0,
                   help="Upsample ultra-short single-chunk histories (fewer patches than this) to this many patches so Chronos-2/Q-former see temporal structure; 0=off")
    p.add_argument("--fb_num_query_tokens", type=int, default=16)
    p.add_argument("--qformer_num_heads", type=int, default=8)
    p.add_argument("--qformer_num_layers", type=int, default=2)
    p.add_argument("--qformer_hidden_dim", type=int, default=768,
                   help="Q-former internal working width (BLIP-2 style: low-dim compression + final linear projection to the target dim; "
                        "768 = Chronos-2 d_model, the information-bottleneck width of the KV source)")
    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--train_embed_tokens", action="store_true",
                   help="Also add embed_tokens to modules_to_save (new-token embeddings become trainable; uses more memory)")
    # Loss weights
    p.add_argument("--text_loss_weight", type=float, default=1.0)
    p.add_argument("--pred_loss_weight", type=float, default=1.0)
    p.add_argument("--roi_loss_weight", type=float, default=1.0)
    p.add_argument("--analysis_loss_weight", type=float, default=1.0)
    # Data / sampling
    p.add_argument("--understanding_max_context", type=int, default=240_000, help="History cap for the understanding branch (left-truncate, keep the most recent)")
    p.add_argument("--forecast_max_context", type=int, default=8192, help="History cap for the forecasting branch (= Chronos-2 context_length, left-truncate)")
    p.add_argument("--encode_sample_chunks", type=int, default=15,
                   help="Number of windows K for uniform window sampling of ultra-long understanding histories (0=off, plain truncation; >=2 enables it, triggered when L > K*chronos_window)")
    p.add_argument("--encode_overview_chunk", type=int, default=1,
                   help="Whether to append a whole-span downsampled overview chunk when sampling triggers (1/0)")
    p.add_argument("--max_user_tokens", type=int, default=1500)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--understanding_bs", type=int, default=900)
    p.add_argument("--forecast_bs", type=int, default=150)
    p.add_argument("--length_pool_factor", type=int, default=0,
                   help="Length-bucketing pool size = bs x this value (batches of similar cost, less padding); 0=off (fully random batches)")
    # Dynamic batch size under a token budget (off by default = fixed bs); calibrate the budget with a GPU smoke run before enabling
    p.add_argument("--understanding_token_budget", type=int, default=0,
                   help="Per-batch budget on the sum of LLM tokens for the understanding branch (short samples automatically form larger batches); 0=fixed bs")
    p.add_argument("--forecast_token_budget", type=int, default=0,
                   help="Per-batch budget on the sum of LLM tokens for the forecasting branch; 0=fixed bs")
    p.add_argument("--patch_budget", type=int, default=0,
                   help="Secondary budget: per-batch cap on the sum of Chronos-2 patches (P x C), keeps short-text x long-history batches from turning Chronos-2 into the bottleneck; 0=off")
    p.add_argument("--patch_token_weight", type=float, default=0.0,
                   help="patch->token conversion weight lambda: dynamic batching/sorting charges the unified cost tok + lambda x patch against the token budget "
                        "(calibrated by memory cost so patch-heavy and text-heavy batches use similar memory); 0=legacy behaviour (patches are only hard-capped by patch_budget)")
    p.add_argument("--max_dynamic_bs", type=int, default=1000, help="Cap on the dynamic batch size (safety rail)")
    p.add_argument("--dynamic_bs_ladder", default="",
                   help="Quantization ladder for the dynamic batch size (comma-separated, e.g. 1,2,3,4,6,8,12,16,24,32,48,64): "
                        "a full batch is cut down to the largest rung <= its size and the leftover samples roll over to the next batch. "
                        "fla's TileLang kernels are specialized per batch size at compile time, so quantization reduces the "
                        "tens-of-seconds JIT compile for every newly seen bs to at most one compile per rung. Empty=off (any integer bs, behaviour unchanged)")
    p.add_argument("--token_cache_dir", default="./.cache/chronos_llm",
                   help="On-disk cache dir for text token lengths (the first build takes minutes, later reads take seconds)")
    p.add_argument("--understanding_epoch_repeats", type=int, default=1,
                   help="How many full passes over the understanding data per epoch (>=1)")
    p.add_argument("--forecast_epoch_repeats", type=int, default=1,
                   help="How many full passes over the forecasting data per epoch (>=1); "
                        "each pass is shuffled/batched independently and interleaved evenly")
    # Training (grouped learning rates: learning_rate is the base lr for full fine-tuning of Chronos-2; the other groups are in trainer._param_group_of)
    p.add_argument("--learning_rate", type=float, default=1e-5, help="lr for full fine-tuning of Chronos-2 (base group)")
    p.add_argument("--lr_lora", type=float, default=1e-4, help="lr for the LLM LoRA (+ optional embed_tokens)")
    p.add_argument("--lr_qformer", type=float, default=1e-4, help="lr for the two from-scratch Q-formers")
    p.add_argument("--lr_gate", type=float, default=1e-3,
                   help="lr for the gated cross-attn gate (a zero-initialised scalar; needs a large lr to open the feedback path)")
    p.add_argument("--num_train_epochs", type=float, default=10.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=2,
                   help="Keep the most recent N checkpoints; <=0 => None = keep all")
    p.add_argument("--save_only_model", action="store_true",
                   help="Save only model weights, no optimizer/scheduler/rng (each checkpoint is just the adapter; "
                        "training cannot be resumed from such checkpoints)")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Resume training from a checkpoint dir; pass 'true' to auto-pick the latest checkpoint under output_dir")
    p.add_argument("--init_from_checkpoint", default=None,
                   help="Load weights from an already-trained PEFT checkpoint dir as initialisation "
                        "(ChronosLLM.from_pretrained(merge=False, is_trainable=True)), "
                        "skipping the from_config+add_lora cold start and not reusing the old optimizer/scheduler state "
                        "-- unlike the exact resume of --resume_from_checkpoint (which requires unchanged hyper-parameters/param groups); "
                        "used to continue training with a changed recipe (e.g. with --freeze_llm, or changed --ss_*_ratio)")
    p.add_argument("--init_merge_reopen", action="store_true",
                   help="Change the continuation semantics of --init_from_checkpoint: instead of continuing the same adapter, "
                        "merge the checkpoint's adapter into the base (the old LLM deltas are baked in and frozen) and then "
                        "add_lora a fresh zero-delta LoRA -- the old deltas are no longer exposed to the new task's gradients. Architecture and forward-behaviour fields are inherited entirely from the checkpoint "
                        "config (same semantics as merge=False continuation; CLI architecture args are ignored); gate_init/"
                        "zero_cross_out_proj are applied as on the cold-start path. The lineage is persisted via config.init_merged_from, "
                        "and from_pretrained replays the merge automatically on reload (nesting two levels is not supported)")
    p.add_argument("--deepspeed", default=None)
    # When running plain DDP instead of DeepSpeed: on the understanding branch the gate/feedback_qformer/Chronos-2
    # cross-attn receive no gradient (unused), and DDP by default raises "Expected to mark a variable ready only once"
    # or hangs, so this must be enabled. The DeepSpeed ZeRO-2 path does not need it (DeepSpeed handles it), hence default False.
    p.add_argument("--ddp_find_unused_parameters", action="store_true")
    # Gate warm start: cross_attn.gate is zero-initialised by default (the starting point is exactly the zero-shot
    # identity), but the feedback path has a cold-start deadlock (gate~0 => cross-attn/feedback_qformer gradients are
    # scaled away by tanh(gate) => the feedback path cannot learn => the gate has no payoff and never opens).
    # Setting >0 (e.g. 0.5) starts with the gate open so the feedback path gets useful gradients to co-adapt from the
    # first step. Default 0 keeps the original behaviour.
    p.add_argument("--gate_init", type=float, default=0.0)
    # Zero-initialise the cross-attn residual branch: zero the weights/bias of cross_attn.attn.out_proj so the feedback
    # branch outputs exactly 0 at the start (unlike gate zero-init: once the gate drifts off 0, the large output of a
    # random out_proj times a small gate still perturbs Chronos-2 strongly). A zero-initialised out_proj lets the feedback contribution grow smoothly from 0, the standard
    # 'zero-init residual branch'. Combine with --gate_init 1.0 so out_proj receives full gradients and the
    # gate x out_proj double deadlock is avoided.
    p.add_argument("--zero_cross_out_proj", action="store_true")
    # Forecasting-branch ablation switches (mutually exclusive; all off by default = the current model)
    # -- architecture ablations (all defaults = the current model) --
    p.add_argument("--history_compressor", default="qformer", choices=["qformer", "pool"],
                   help="A1: history-side compressor. pool = drop the learned queries/cross-attn and instead split each window into k equal "
                        "time segments, masked-mean each and apply a linear projection; token count/positions/channel emb/in-window PE are "
                        "identical to the qformer arm, only the compression operator changes (params ~60M -> ~6M).")
    p.add_argument("--feedback_compressor", default="qformer", choices=["qformer", "pool"],
                   help="B1: feedback-side compressor. pool = split by valid length into M equal segments, masked-mean + linear projection; "
                        "**the injection scheme is unchanged** (still per-layer gated cross-attn), unlike the single-point-injection ablation.")
    p.add_argument("--history_encode_layer", type=int, default=0,
                   help="A4: which Chronos-2 encoder layer the soft prompt is taken from (1-indexed; early exit saves compute); "
                        "0 = last layer + final_layer_norm (current).")
    p.add_argument("--feedback_llm_layer", type=int, default=0,
                   help="B2: which LLM layer's hidden states are fed back (1-indexed); 0 = last layer (current).")
    p.add_argument("--cross_attn_layers", default="all",
                   help="B3: which Chronos-2 encoder blocks receive the feedback: all|lastK|firstK|comma-separated layer indices; "
                        "unselected layers keep their cross_attn weights (checkpoint structure unchanged) but never receive cross_states.")
    p.add_argument("--chronos_only", action="store_true",
                   help="Ablation 2 (chronos_only arm): the forecasting branch fine-tunes only Chronos-2 and never runs the LLM/soft prompt/feedback")
    p.add_argument("--freeze_llm", action="store_true",
                   help="Freeze the LLM (incl. LoRA) and train only Chronos-2 + the two Q-formers + gate -- "
                        "combine with --init_from_checkpoint to continue training only the forecasting modules on top of a trained LLM "
                        "(--ss_start_ratio/--ss_end_ratio set the share of self-generated text)")
    p.add_argument("--freeze_ts", action="store_true",
                   help="Freeze Chronos-2 + the two Q-formers and train only the LLM LoRA "
                        "(mutually exclusive with --freeze_llm/--chronos_only)")
    p.add_argument("--feedback_scope", default="all", choices=["all", "conclusion"],
                   help="Ablation 1: which segment of the LLM hidden states is fed back to Chronos-2 (all = whole sequence; conclusion = only after </think>)")
    p.add_argument("--forecast_prompt_only", action="store_true",
                   help="Ablation 3 (prompt_only arm): do not generate reasoning; feed back the hidden states of the plain_prompt text directly (the parquet must contain a plain_prompt column)")
    p.add_argument("--chronos_random_init", action="store_true",
                   help="Ablation 4: does TSFM pre-training matter -- skip loading the Chronos-2 pre-trained weights and train from random initialisation "
                        "(freely combinable with --chronos_only or the FULL protocol)")
    p.add_argument("--tsfm_backbone", default="chronos2", choices=["chronos2", "timesfm3"],
                   help="Backbone ablation: swap the TSFM backbone. With timesfm3, --chronos_ckpt points at the "
                        "TimesFM-3.0 weight directory instead (the timesfm_backbone adapter wraps it into the "
                        "chronos interface shape)")
    p.add_argument("--llm_only", action="store_true",
                   help="LLM-only baseline (the symmetric ablation that removes the TSFM): do not load Chronos-2/the two Q-formers; "
                        "the dataset serialises the series as full-precision text (ts_as_text) fed straight into the LLM; both understanding and forecasting have only the "
                        "text CE (forecast supervision = <think>reasoning</think>conclusion + future values as text), "
                        "trainable params = LLM LoRA (same LoRA config and lr_lora). Mutually exclusive with chronos_only/"
                        "forecast_prompt_only/freeze_llm/SS")
    p.add_argument("--ss_end_ratio", type=float, default=1.0,
                   help="In-training scheduled sampling: the teacher ratio anneals linearly from --ss_start_ratio to this value "
                        "(enabled only if <1.0; e.g. 0.7 = by the end of training 30%% of forecast batches feed back a self-generated conclusion)")
    p.add_argument("--ss_start_ratio", type=float, default=1.0)
    p.add_argument("--ss_max_new_tokens", type=int, default=288,
                   help="max_new_tokens for self-generation in SS batches (should cover the longest generated segment of the training data)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=-1)
    # Automatic mid-training evaluation (every N epochs): the evaluated tasks follow the trained branches -- mixed
    # training evaluates both, single-branch training only its own task. Outputs go to {output_dir}/mid_eval/epoch_{E}/.
    p.add_argument("--eval_every_epochs", type=int, default=0,
                   help="Run mid-training evaluation every N epochs (0=off)")
    p.add_argument("--eval_understanding_jsonl", nargs="*", default=[],
                   help="Test jsonl(s) for the understanding-branch evaluation (the training jsonl is the train split; a test set must be given separately)")
    p.add_argument("--eval_understanding_base_dir", default=None,
                   help="Prefix for relative ori_path values inside the eval jsonl (same as media_root)")
    p.add_argument("--eval_forecast_parquet", default=None,
                   help="Evaluation parquet for the forecasting branch (split=test); defaults to --forecast_parquet")
    p.add_argument("--eval_understanding_limit", type=int, default=0,
                   help="Evaluate only the first N rows of each test jsonl (0=all)")
    p.add_argument("--eval_forecast_limit", type=int, default=0)
    p.add_argument("--eval_understanding_bs", type=int, default=900, help="Evaluation batch size per rank")
    p.add_argument("--eval_understanding_max_new_tokens", type=int, default=200,
                   help="max_new_tokens for understanding-branch generation in mid_eval -- the default 200 suits "
                        "short direct-answer test sets without a reasoning segment; **datasets supervised with reasoning must raise it**, "
                        "otherwise generation is truncated before the final 'Answer: X' and template-based accuracy extraction is underestimated. "
                        "Same kind of knob as --eval_forecast_max_new_tokens on the forecasting branch")
    p.add_argument("--eval_forecast_bs", type=int, default=150)
    p.add_argument("--eval_forecast_teacher_forced", type=int, default=1,
                   help="Whether the forecasting-branch mid-training eval also runs teacher_forced (_tf.csv, to see exposure bias) in addition to generate; 1 = both")
    p.add_argument("--eval_forecast_max_new_tokens", type=int, default=320,
                   help="max_new_tokens for generative mid_eval -- the default 320 suits short reasoning; "
                        "datasets with clearly longer reasoning (e.g. ST-Bench multi-node descriptions) must raise it, otherwise generation is hard-truncated before "
                        "the conclusion is written (<answer> never appears) -- Chronos-2's quantile forecast does not parse the "
                        "generated text and will not become nan because of it, but a truncated reasoning used as the cross-attn feedback condition is "
                        "systematically biased toward 'half-finished, not yet converged to a conclusion' hidden states, degrading the conditioning quality overall")
    return p.parse_args()


def _apply_gate_init(model, gate_init: float):
    """Fill every gated cross-attn gate with gate_init; return (count, max |gate| before the overwrite).

    Suffix matching hits both the bare ChronosLLM and the PEFT wrapper (the modules_to_save copy plus the
    original_module snapshot); filling both copies matches the copy semantics of the cold-start path
    ('fill before add_lora').
    """
    import torch as _torch
    n_gate, prev_max = 0, 0.0
    with _torch.no_grad():
        for n, prm in model.named_parameters():
            if n.endswith(".cross_attn.gate"):
                prev_max = max(prev_max, float(prm.abs().max()))
                prm.fill_(gate_init); n_gate += 1
    return n_gate, prev_max


def _apply_zero_cross_out_proj(model):
    """Zero the cross-attn residual-branch out_proj (the feedback contribution learns smoothly from 0); return the number of zeroed tensors."""
    import torch as _torch
    n_op = 0
    with _torch.no_grad():
        for n, prm in model.named_parameters():
            if ".cross_attn.attn.out_proj." in ("." + n):
                prm.zero_(); n_op += 1
    return n_op


def main():
    args = parse_args()
    if args.llm_only:
        bad = [f for f in ("chronos_only", "forecast_prompt_only", "freeze_llm",
                           "chronos_random_init") if getattr(args, f)]
        if bad or args.ss_end_ratio < 1.0 or args.ss_start_ratio < 1.0:
            raise ValueError(f"--llm_only is mutually exclusive with {bad or 'SS'} (there is no Chronos-2/feedback path)")
        if args.eval_forecast_teacher_forced:
            # llm_only has no teacher-forced forecast evaluation (the values come from the generated text itself);
            # silently disable it instead of raising: the training scripts enable tf by default and the llm_only
            # branch should not require every caller to remember to turn it off.
            args.eval_forecast_teacher_forced = 0
    mcfg = ChronosLLMConfig(
        chronos_ckpt=args.chronos_ckpt,
        llm_path=args.llm_path,
        sw_queries_per_window=args.sw_queries_per_window,
        # sw_target_windows / sw_min_windows are passed through only for backward compatibility; plan() no longer
        # reads them (see SlidingWindowQFormer). The actual token budget is set by the three sw_token_* below +
        # sw_queries_per_window + sw_global_queries.
        sw_target_windows=args.sw_target_windows,
        sw_min_windows=args.sw_min_windows,
        sw_global_queries=args.sw_global_queries,
        sw_token_min=args.sw_token_min,
        sw_token_max=args.sw_token_max,
        sw_token_pc_ref=args.sw_token_pc_ref,
        chronos_window=args.chronos_window,
        encode_max_rows=args.encode_max_rows,
        history_stats_token=args.history_stats_token,
        short_min_patches=args.short_min_patches,
        fb_num_query_tokens=args.fb_num_query_tokens,
        qformer_num_heads=args.qformer_num_heads,
        qformer_num_layers=args.qformer_num_layers,
        qformer_hidden_dim=args.qformer_hidden_dim,
        text_loss_weight=args.text_loss_weight,
        pred_loss_weight=args.pred_loss_weight,
        roi_loss_weight=args.roi_loss_weight,
        analysis_loss_weight=args.analysis_loss_weight,
        llm_dtype="bfloat16" if args.bf16 else "float32",
        history_compressor=args.history_compressor,
        feedback_compressor=args.feedback_compressor,
        history_encode_layer=args.history_encode_layer,
        feedback_llm_layer=args.feedback_llm_layer,
        cross_attn_layers=args.cross_attn_layers,
        chronos_only=args.chronos_only,
        feedback_scope=args.feedback_scope,
        forecast_prompt_only=args.forecast_prompt_only,
        chronos_random_init=args.chronos_random_init,
        tsfm_backbone=args.tsfm_backbone,
        llm_only=args.llm_only,
    )
    if args.init_merge_reopen and not args.init_from_checkpoint:
        raise ValueError("--init_merge_reopen must be used together with --init_from_checkpoint")
    if args.init_from_checkpoint and not args.init_merge_reopen:
        # Continue training with a changed recipe (e.g. --freeze_llm / changed --ss_*_ratio): load the trained weights
        # as initialisation -- no cold start, no reuse of the old optimizer state (unlike the exact resume of
        # --resume_from_checkpoint). add_lora is skipped (the LoRA is already loaded with the checkpoint via
        # adapter_config.json). Gate warm start / zero_cross_out_proj are left alone by default but **applied after
        # loading when passed explicitly** -- intended for continuing forecast training from a checkpoint whose feedback
        # path was never trained (e.g. a pure understanding-branch product: gate stuck at 0, cross-attn out_proj still
        # randomly initialised, feedback_qformer never received a gradient); there the feedback path is equivalent to a
        # cold start and skipping this would replay the gate=0 deadlock plus the random-out_proj CRPS blow-up. Do NOT
        # pass them when continuing from a forecast checkpoint whose feedback path was already trained (they would wipe
        # the trained gate/out_proj -- the log prints max |gate| before the overwrite for checking; it should be ~0).
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[train] loading weights from checkpoint for continued training (fresh optimizer): {args.init_from_checkpoint}",
                  flush=True)
        model = ChronosLLM.from_pretrained(args.init_from_checkpoint, merge=False, is_trainable=True)
        if args.gate_init != 0.0:
            n_gate, prev_max = _apply_gate_init(model, args.gate_init)
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[train] gate warm start (applied after init_from_checkpoint): {n_gate} cross_attn.gate tensors "
                      f"set to {args.gate_init} (max |gate| before overwrite={prev_max:.6f}, expected ~0)", flush=True)
        if args.zero_cross_out_proj:
            n_op = _apply_zero_cross_out_proj(model)
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[train] cross-attn out_proj zero-init (applied after init_from_checkpoint): {n_op} tensors zeroed",
                      flush=True)
    else:
        if args.init_merge_reopen:
            # Fresh LoRA after merge: architecture/forward-behaviour fields are inherited entirely from the checkpoint
            # config (same semantics as merge=False continuation -- otherwise replaying the adapter fails on a structure
            # mismatch, e.g. the history_stats_token branch); init_merged_from records the lineage => from_config replays
            # the merge, and from_pretrained reloads along the same path.
            mcfg = ChronosLLMConfig.from_pretrained(args.init_from_checkpoint)
            mcfg.init_merged_from = args.init_from_checkpoint
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[train] merge + fresh-LoRA continuation: merging the adapter of {args.init_from_checkpoint} "
                      "into the base, then add_lora again (old LLM deltas baked in and frozen, new LoRA starts from zero delta)",
                      flush=True)
        base = ChronosLLM.from_config(mcfg)
        # Gate warm start (filled before add_lora; the value is copied into the PEFT wrapper via modules_to_save and updated normally during training).
        if args.gate_init != 0.0:
            n_gate, _ = _apply_gate_init(base, args.gate_init)
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[train] gate warm start: {n_gate} cross_attn.gate tensors set to {args.gate_init}", flush=True)
        # Zero-init the cross-attn out_proj (the feedback branch outputs exactly 0 at the start; likewise filled before add_lora and copied via modules_to_save).
        if args.zero_cross_out_proj:
            n_op = _apply_zero_cross_out_proj(base)
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[train] cross-attn out_proj zero-init: {n_op} tensors zeroed", flush=True)
        # LoRA is added at training-assembly time: only the LLM gets LoRA; Chronos-2/the two Q-formers are trainable via modules_to_save.
        model = add_lora(base, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                         train_embed_tokens=args.train_embed_tokens)
    if args.chronos_only:
        # Ablation 2: train only Chronos-2's native params; freeze LLM/LoRA/the two Q-formers/the injected cross_attn
        # (the forward pass never touches them; leaving them trainable would hand DeepSpeed requires_grad=True params
        # with no gradient). cross_states=None => cross_attn does not run.
        n_tr = 0
        for n, prm in model.named_parameters():
            keep = ("chronos" in n) and (".cross_attn." not in n)
            prm.requires_grad_(keep)
            n_tr += int(keep)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[train] chronos_only: only {n_tr} Chronos-2 parameter tensors are trainable (everything else frozen)", flush=True)
    if args.freeze_llm:
        # Freeze the LLM (incl. LoRA), leaving only Chronos-2 + the two Q-formers (+ their gate; the cross_attn-named
        # parts already belong to the qformer group) trainable.
        n_tr = 0
        for n, prm in model.named_parameters():
            keep = ("chronos" in n) or ("feedback_qformer" in n) or ("history_qformer" in n)
            prm.requires_grad_(keep)
            n_tr += int(keep)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[train] freeze_llm: {n_tr} Chronos-2/Q-former parameter tensors trainable, LLM (incl. LoRA) frozen",
                  flush=True)
    if args.freeze_ts:
        if args.freeze_llm or args.chronos_only:
            raise ValueError("--freeze_ts is mutually exclusive with --freeze_llm/--chronos_only")
        n_fr = freeze_ts_modules(model)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[train] freeze_ts: froze {n_fr} Chronos-2/Q-former parameter tensors, training only the LLM LoRA",
                  flush=True)
    if int(os.environ.get("RANK", "0")) == 0:
        print("Trainable parameters:\n" + model.get_base_model().trainable_parameter_summary(), flush=True)

    tok = model.get_base_model().tokenizer
    understanding_ds = (
        UnderstandingJsonlDataset(
            args.understanding_jsonl, tok, base_dir=args.understanding_base_dir,
            max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens,
            sample_chunks=args.encode_sample_chunks,
            overview_chunk=bool(args.encode_overview_chunk),
            sample_window=args.chronos_window or 8192,
            no_reasoning=args.understanding_no_reasoning,
            answer_first_explanation=args.understanding_answer_first,
            ts_as_text=args.llm_only,
        )
        if args.understanding_jsonl
        else None
    )
    forecast_ds = (
        ForecastParquetDataset(
            args.forecast_parquet, tok, split="train",
            max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens,
            forecast_prompt_only=args.forecast_prompt_only,
            emit_ss_prefix=(args.ss_end_ratio < 1.0),
            ts_as_text=args.llm_only,
        )
        if args.forecast_parquet
        else None
    )
    assert understanding_ds is not None or forecast_ds is not None, "Provide data for at least one branch"
    concat = ConcatBranchDataset(understanding_ds, forecast_ds)

    # Mid-training evaluation: tasks follow the trained branches -- whichever branch is trained gets evaluated.
    callbacks = []

    if args.ss_end_ratio < 1.0:
        # Linear annealing of the teacher ratio for in-training scheduled sampling: start->end over max_steps.
        # The attributes live on the base model (the ChronosLLM under the PEFT wrapper); forward_forecast reads them via getattr.
        from transformers import TrainerCallback as _TCB2

        class _SSAnnealCallback(_TCB2):
            def __init__(self, mdl, r0, r1, max_new):
                self.mdl, self.r0, self.r1 = mdl, r0, r1
                self.mdl.ss_max_new_tokens = max_new
                self.mdl.ss_teacher_ratio = r0
            def on_step_begin(self, targs, state, control, **kw):
                t = min(1.0, state.global_step / max(1, state.max_steps))
                r = self.r0 + (self.r1 - self.r0) * t
                self.mdl.ss_teacher_ratio = r
                # The trigger is decided by a deterministic RNG keyed on global_step -- every rank makes the same
                # decision at the same step (otherwise under ZeRO-2 one rank generates while the others idle at the
                # allreduce, and the load imbalance slows everything down).
                import random as _rnd
                self.mdl.ss_force_gen = _rnd.Random(9173 + state.global_step).random() >= r
                return control
            def on_log(self, targs, state, control, logs=None, **kw):
                if logs is not None:
                    logs["ss_teacher_ratio"] = round(float(self.mdl.ss_teacher_ratio), 4)
                return control
        callbacks.append(_SSAnnealCallback(model.get_base_model(), args.ss_start_ratio,
                                           args.ss_end_ratio, args.ss_max_new_tokens))
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[train] SS annealing enabled: teacher {args.ss_start_ratio}->{args.ss_end_ratio}, "
                  f"gen max_new_tokens={args.ss_max_new_tokens}", flush=True)

    # Memory-snapshot diagnostics (only when MEM_SNAPSHOT_PATH is set; never triggers in normal training):
    # enable _record_memory_history before step 1, dump a snapshot after N steps and stop training. The snapshot
    # contains the full device_traces event history -> replay it offline to get the allocation call stacks at the
    # peak moment and locate patch-memory regressions.
    if os.environ.get("MEM_SNAPSHOT_PATH"):
        import torch as _torch
        from transformers import TrainerCallback as _TCB

        class _MemSnapCallback(_TCB):
            def __init__(self, path, steps):
                self.path = path; self.steps = steps; self.n = 0
            def on_train_begin(self, a, s, c, **kw):
                _torch.cuda.memory._record_memory_history(
                    enabled="all", context="all", stacks="all", max_entries=400000)
            def on_step_end(self, a, s, c, **kw):
                self.n += 1
                if self.n >= self.steps:
                    p = f"{self.path}.rank{os.environ.get('RANK','0')}.pickle"
                    _torch.cuda.memory._dump_snapshot(p)
                    print(f"[MEM_SNAPSHOT] dumped {p} after {self.n} steps", flush=True)
                    c.should_training_stop = True
                return c
        callbacks.append(_MemSnapCallback(os.environ["MEM_SNAPSHOT_PATH"],
                                          int(os.environ.get("MEM_SNAPSHOT_STEPS", "1"))))
    if args.eval_every_epochs > 0:
        eval_u_files = args.eval_understanding_jsonl if understanding_ds is not None else []
        eval_f_parquet = ((args.eval_forecast_parquet or args.forecast_parquet)
                          if forecast_ds is not None else None)
        if understanding_ds is not None and not eval_u_files \
                and int(os.environ.get("RANK", "0")) == 0:
            print("WARNING: the understanding branch is being trained but --eval_understanding_jsonl was not given; mid-training evaluation will skip the understanding task", flush=True)
        if eval_u_files or eval_f_parquet:
            callbacks.append(MidTrainEvalCallback(
                tokenizer=tok, output_dir=args.output_dir,
                every_epochs=args.eval_every_epochs,
                understanding_jsonl=eval_u_files,
                understanding_base_dir=args.eval_understanding_base_dir,
                understanding_limit=args.eval_understanding_limit,
                understanding_bs=args.eval_understanding_bs,
                understanding_max_new_tokens=args.eval_understanding_max_new_tokens,
                # answer-first's inference prefix is token-identical to no-CoT (the dataset sets reasoning to None
                # under answer_first), and the standalone evaluation (NO_REASONING=1) uses the same convention --
                # mid_eval must match, otherwise mid_eval of an answer-first run would use the 'continue the reasoning'
                # prefix and make the model write a <think> segment it never learned.
                understanding_no_reasoning=(args.understanding_no_reasoning
                                            or args.understanding_answer_first),
                forecast_parquet=eval_f_parquet,
                forecast_limit=args.eval_forecast_limit,
                forecast_bs=args.eval_forecast_bs,
                forecast_max_new_tokens=args.eval_forecast_max_new_tokens,
                forecast_teacher_forced=bool(args.eval_forecast_teacher_forced),
                max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens,
                sample_chunks=args.encode_sample_chunks,
                overview_chunk=bool(args.encode_overview_chunk),
                sample_window=args.chronos_window or 8192,
            ))

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.forecast_bs,  # placeholder; the real batch is decided by the batch_sampler
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="epoch",   # both branches checkpoint per epoch (aligned with the mid_eval epoch cadence)
        save_total_limit=(args.save_total_limit if args.save_total_limit > 0 else None),  # <=0 => keep all
        save_only_model=args.save_only_model,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        # tensorboard: custom log keys (text_loss/gate/cuda_mem_gb/batch_* etc.) all flow into tb via on_log.
        # The runtime image may lack tensorboard (TensorBoardCallback would crash with RuntimeError) -> fall back to
        # report_to=none; the console/log file still carry every metric (monitoring does not depend on tb).
        report_to=(["tensorboard"] if (importlib.util.find_spec("tensorboard")
                                       or importlib.util.find_spec("tensorboardX")) else ["none"]),
        logging_dir=os.path.join(args.output_dir, "tb"),
        seed=args.seed,
    )

    trainer = ChronosLLMTrainer(
        model=model,
        args=targs,
        train_dataset=concat,
        processing_class=tok,
        callbacks=callbacks or None,
        understanding_dataset=understanding_ds,
        forecast_dataset=forecast_ds,
        understanding_bs=args.understanding_bs,
        forecast_bs=args.forecast_bs,
        understanding_max_context=args.understanding_max_context,
        forecast_max_context=args.forecast_max_context,
        sampler_seed=args.seed,
        lr_lora=args.lr_lora,
        lr_qformer=args.lr_qformer,
        lr_gate=args.lr_gate,
        length_pool_factor=args.length_pool_factor,
        understanding_token_budget=args.understanding_token_budget,
        forecast_token_budget=args.forecast_token_budget,
        patch_budget=args.patch_budget,
        patch_token_weight=args.patch_token_weight,
        understanding_epoch_repeats=args.understanding_epoch_repeats,
        forecast_epoch_repeats=args.forecast_epoch_repeats,
        max_dynamic_bs=args.max_dynamic_bs,
        dynamic_bs_ladder=(tuple(int(v) for v in args.dynamic_bs_ladder.split(",") if v.strip())
                           or None) if args.dynamic_bs_ladder else None,
        token_cache_dir=args.token_cache_dir,
    )
    # resume: 'true' -> True (auto-pick the latest checkpoint under output_dir); otherwise resume from the given dir.
    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() in ("true", "1", "yes"):
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    # After training, rank0 plots every numeric metric of log_history as png ({output_dir}/plots/).
    # The runtime image may lack matplotlib: training + saving are already done, so a plotting failure must not make
    # the whole job exit non-zero or trigger a rerun.
    if trainer.args.process_index == 0:
        try:
            from chronos_llm.scripts.utils.plot_train_logs import plot_log_history
            plot_log_history(trainer.state.log_history, os.path.join(args.output_dir, "plots"))
        except Exception as e:
            print(f"[train] post-training plotting skipped (missing dependency/error; the saved model and metrics are unaffected): {e}")


if __name__ == "__main__":
    main()
