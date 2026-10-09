"""ChronosLLMTrainer: an HF Trainer subclass.

- ``compute_loss``: dispatches each batch to one of the model's two forwards according to its
  branch (the model consumes the dict directly).
- ``get_train_dataloader``: uses DualBranchBatchSampler to produce homogeneous batches + a custom
  collator, sharded by DDP world_size/rank.
- ``create_optimizer``: **grouped learning rates** -- gate (a zero-initialised scalar; needs a large
  lr to open the feedback path), the two Q-formers trained from scratch, LoRA, and full fine-tuning
  of chronos2, one group each (see _param_group_of).
  Note: grouped lrs require the DeepSpeed config to **not** define optimizer/scheduler (otherwise
  DeepSpeed builds its own single-lr optimizer and this method is never called).
- Logging: text/pred/roi component losses (window means) + gate opening tanh(gate) mean/max
  (answers "is the LLM feedback path actually used?") + peak memory cuda_mem_gb + GPU power
  gpu_power_w/max_w (per-step samples taken in compute_loss over the logging window, mean/peak;
  low = data starvation / idling).
"""

import torch
from torch.utils.data import DataLoader
from transformers import Trainer

from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.data.sampler import DualBranchBatchSampler


class _EpochAwareDataLoader(DataLoader):
    """DataLoader with ``set_epoch``: forwards to the batch_sampler for per-epoch reshuffling.

    transformers' Trainer only calls ``set_epoch`` if the dataloader **itself** has it (a bare
    DataLoader does not -> DualBranchBatchSampler.epoch stays 0 -> identical data order every
    epoch). ``accelerator.prepare`` cannot be used instead: it would shard the batch_sampler by rank
    a second time, conflicting with the manual sharding inside our sampler."""

    def set_epoch(self, epoch: int):
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)


class _FaultTolerantTensorBoard:
    """Wraps HF's TensorBoardCallback so that a failed disk write never takes down training.

    A transient filesystem error (e.g. a quota overrun) raised inside the async writer thread would
    otherwise bubble up through on_log->flush and kill the whole multi-GPU job. TensorBoard is pure
    telemetry (the metrics are also in the text log), so any OSError should warn and skip: drop the current writer; on the next
    on_log TensorBoardCallback's own `tb_writer is None -> _init_summary_writer` rebuilds it -- if the
    quota recovered, logging resumes, otherwise it is skipped again. Composition rather than
    inheritance: every callback event is forwarded."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr) or not name.startswith("on_"):
            return attr

        def _safe(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except OSError as e:
                print(f"[tensorboard] warning: event write failed ({e}); skipping this one, training continues", flush=True)
                try:
                    if getattr(self._inner, "tb_writer", None) is not None:
                        self._inner.tb_writer.close()
                except Exception:
                    pass
                self._inner.tb_writer = None

        return _safe


class ChronosLLMTrainer(Trainer):
    def __init__(
        self,
        *args,
        understanding_dataset=None,
        forecast_dataset=None,
        understanding_bs: int = 8,
        forecast_bs: int = 2,
        understanding_max_context: int = 240_000,
        forecast_max_context: int = 8192,
        sampler_seed: int = 0,
        lr_lora: float = 1e-4,
        lr_qformer: float = 1e-4,
        lr_gate: float = 1e-3,
        length_pool_factor: int = 0,
        understanding_token_budget: int = 0,
        forecast_token_budget: int = 0,
        patch_budget: int = 0,
        max_dynamic_bs: int = 0,
        dynamic_bs_ladder=None,
        patch_token_weight: float = 0.0,
        token_cache_dir=None,
        understanding_epoch_repeats: int = 1,
        forecast_epoch_repeats: int = 1,
        **kwargs,
    ):
        self._understanding_dataset = understanding_dataset
        self._forecast_dataset = forecast_dataset
        self._understanding_bs = understanding_bs
        self._forecast_bs = forecast_bs
        self._understanding_max_context = understanding_max_context
        self._forecast_max_context = forecast_max_context
        self._sampler_seed = sampler_seed
        self._lr_lora = lr_lora
        self._lr_qformer = lr_qformer
        self._lr_gate = lr_gate
        self._length_pool_factor = length_pool_factor
        self._u_token_budget = understanding_token_budget
        self._f_token_budget = forecast_token_budget
        self._patch_budget = patch_budget
        self._max_dynamic_bs = max_dynamic_bs
        self._dynamic_bs_ladder = dynamic_bs_ladder
        self._patch_token_weight = patch_token_weight
        self._token_cache_dir = token_cache_dir
        self._u_epoch_repeats = understanding_epoch_repeats
        self._f_epoch_repeats = forecast_epoch_repeats
        self._comp_sums: dict = {}
        self._comp_counts: dict = {}
        super().__init__(*args, **kwargs)
        # Swap the TensorBoard callback for the fault-tolerant wrapper (see _FaultTolerantTensorBoard);
        # when tensorboard is not installed the callback is absent and this is silently skipped.
        try:
            from transformers.integrations.integration_utils import TensorBoardCallback
            for cb in list(self.callback_handler.callbacks):
                if isinstance(cb, TensorBoardCallback):
                    self.callback_handler.callbacks.remove(cb)
                    self.callback_handler.callbacks.append(_FaultTolerantTensorBoard(cb))
        except ImportError:
            pass

    # ----------------------------------------------------- grouped learning rates / gate monitoring
    @staticmethod
    def _param_group_of(name: str) -> str:
        """Trainable parameter name -> lr group. Works for names before and after PEFT wrapping
        (a leading dot is prepended before matching).

        - gate: zero-initialised + tanh => a serial gradient bottleneck (while the gate stays at 0
          the cross-attn weight gradients are identically 0); a single scalar per layer, so give it
          a large lr (default 1e-3) to open the feedback path quickly.
        - qformer: randomly initialised from scratch, the usual 1e-4 scale. The **newly injected
          cross-attn projections/LN** inside chronos are likewise trained from scratch and belong to
          this group -- with the small chronos lr the feedback branch converges much more slowly once
          the gate opens.
        - llm: only LoRA is trainable (+ optional embed_tokens modules_to_save), 1e-4 scale.
        - chronos: full fine-tuning of pretrained weights, uses args.learning_rate (small, default 1e-5).
        """
        n = "." + name
        if n.endswith(".cross_attn.gate"):
            return "gate"
        if (".cross_attn." in n or ".history_qformer." in n or ".feedback_qformer." in n
                or ".history_stats_proj." in n):
            return "qformer"
        if ".llm." in n:
            return "llm"
        return "chronos"

    def create_optimizer(self, model=None):
        opt_model = self.model if model is None else model
        if self.optimizer is None:
            lrs = {
                "gate": self._lr_gate,
                "qformer": self._lr_qformer,
                "llm": self._lr_lora,
                "chronos": self.args.learning_rate,
            }
            decay_names = set(self.get_decay_parameter_names(opt_model))
            buckets: dict = {}
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                grp = self._param_group_of(n)
                # The gate is forcibly exempted from weight decay: it is neither a bias nor a
                # LayerNorm, so by default it lands in the decay set, and decoupled decay would keep
                # pulling the zero-initialised gate back towards 0 at rate lr_gate (the largest lr in
                # the model) x wd -- directly fighting the design goal of the gate leaving 0 quickly
                # to open the feedback path.
                wd = 0.0 if grp == "gate" else (self.args.weight_decay if n in decay_names else 0.0)
                buckets.setdefault((grp, wd), []).append(p)
            grouped = [
                {"params": ps, "lr": lrs[g], "weight_decay": wd}
                for (g, wd), ps in sorted(buckets.items())
            ]
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            optimizer_kwargs.pop("params", None)
            optimizer_kwargs.pop("model", None)
            self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        return self.optimizer

    def _gate_params(self):
        if not hasattr(self, "_gate_param_cache"):
            self._gate_param_cache = [
                p for n, p in self.model.named_parameters()
                if n.endswith("cross_attn.gate") and p.requires_grad
            ]
        return self._gate_param_cache

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        # Accumulate component losses; ``log`` takes the window mean on the Trainer's own logging
        # cadence (calling self.log here would log once per gradient-accumulation micro-batch).
        # Values are this rank's local means.
        for k in ("text_loss", "pred_loss", "roi_loss"):
            v = outputs.get(k)
            if v is not None:
                self._comp_sums[k] = self._comp_sums.get(k, 0.0) + float(v.detach())
                self._comp_counts[k] = self._comp_counts.get(k, 0) + 1
        ids = inputs.get("input_ids")
        if ids is not None:
            # Under dynamic batching the size floats per batch: accumulate over the whole logging
            # window and report min/max at log time (sampling only the last batch would, in mixed
            # training, attribute an OOM to the wrong branch's batch).
            if not hasattr(self, "_bs_window"):
                self._bs_window = []
            self._bs_window.append(int(ids.shape[0]))
        # Per-batch sum of text tokens / sum of chronos patches (for memory attribution / cost-model
        # calibration; readable per batch with logging_steps=1).
        if not hasattr(self, "_cost_window"):
            self._cost_window = []
        am, tl = inputs.get("attention_mask"), inputs.get("true_lengths")
        self._cost_window.append((
            int(am.sum().item()) if am is not None else 0,
            int(torch.ceil(tl.float() / 16).sum().item()) if tl is not None else 0,
            int(ids.numel()) if ids is not None else 0,   # padded text tokens (bs x longest in batch)
        ))
        # GPU power (W) sampled per step: read torch.cuda.power_draw in compute_loss (the point of
        # full GPU load) (mW, mean over the last sampling period; a pure NVML host query, no stream
        # sync). Accumulate over the logging window and report mean/peak at log time -- more robust
        # than an instantaneous read in log() (which reflects only the last ~1s and tends to land in
        # the idle gap between steps), and also valid for logging_steps>1. Local to this rank (each
        # process is bound to its own GPU).
        if (torch.cuda.is_available() and hasattr(torch.cuda, "power_draw")
                and not getattr(self, "_power_unavailable", False)):
            if not hasattr(self, "_power_window"):
                self._power_window = []
            try:
                self._power_window.append(torch.cuda.power_draw() / 1000.0)
            except Exception as e:
                # On the first failure disable sampling and warn (instead of raising every
                # micro-batch). The most common cause is a missing pynvml (torch.cuda.power_draw
                # depends on it) -- swallowing it silently would make the power metric vanish
                # without a trace and be hard to debug.
                self._power_unavailable = True
                import warnings
                warnings.warn(
                    f"GPU power sampling disabled: torch.cuda.power_draw() failed ({type(e).__name__}: {e}); "
                    f"usually pynvml is missing, `pip install nvidia-ml-py` restores the gpu_power_w log."
                )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        # Training log entries (the ones carrying loss) get: (1) the mean component losses since the
        # last log (then the window is reset); (2) gate opening |tanh(gate)| mean/max -- if it stays
        # ~0 for long, the LLM->chronos feedback path has not opened.
        if "loss" in logs:
            if self._comp_counts:
                for k, c in self._comp_counts.items():
                    logs[k] = round(self._comp_sums[k] / c, 6)
                self._comp_sums.clear()
                self._comp_counts.clear()
            gates = self._gate_params()
            if gates:
                with torch.no_grad():
                    t = torch.stack([g.detach().float().reshape(()) for g in gates]).tanh().abs()
                logs["gate_tanh_mean"] = round(t.mean().item(), 6)
                logs["gate_tanh_max"] = round(t.max().item(), 6)
            # Peak memory (GiB, window peak since the last log) + batch-size range within the window:
            # for calibration / monitoring. reset_peak_memory_stats makes each logging window measure
            # its own peak -- the process-level monotonic peak freezes after the first spike and would
            # never show memory dropping after a budget reduction.
            if torch.cuda.is_available():
                logs["cuda_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
                torch.cuda.reset_peak_memory_stats()
                # GPU power (W) over this logging window: mean + peak (window peak, same philosophy as
                # cuda_mem_gb). Persistently low => data starvation / idle GPU.
                if getattr(self, "_power_window", None):
                    pw = self._power_window
                    logs["gpu_power_w"] = round(sum(pw) / len(pw), 1)
                    logs["gpu_power_max_w"] = round(max(pw), 1)
                    pw.clear()
            if getattr(self, "_bs_window", None):
                logs["batch_bs_min"] = min(self._bs_window)
                logs["batch_bs_max"] = max(self._bs_window)
                self._bs_window.clear()
            if getattr(self, "_cost_window", None):
                logs["batch_tok"] = max(t for t, _, _ in self._cost_window)
                logs["batch_patch"] = max(p for _, p, _ in self._cost_window)
                logs["batch_padded_tok"] = max(pt for _, _, pt in self._cost_window)
                self._cost_window.clear()
        return super().log(logs, start_time)

    def _run_epoch(self, model, epoch, *args, **kwargs):
        # In transformers 5.9, on the first epoch after a mid-run resume, skip_first_batches
        # (accelerate returns a plain DataLoader with the batch_sampler wrapped in SkipBatchSampler)
        # happens **before** the hasattr(set_epoch) check => that epoch's set_epoch is lost: the
        # sampler stays on epoch 0's shuffle, and the skip also skips the first K batches of the wrong
        # plan (samples duplicated/lost, completely silently). Set the epoch on the sampler explicitly
        # before the wrapping happens; on the normal path dataloader.set_epoch(epoch) then sets the
        # same value again -- idempotent and harmless.
        if getattr(self, "_dual_batch_sampler", None) is not None:
            self._dual_batch_sampler.set_epoch(int(epoch))
        return super()._run_epoch(model, epoch, *args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        concat = self.train_dataset  # ConcatBranchDataset
        n_u = len(self._understanding_dataset) if self._understanding_dataset is not None else 0
        n_f = len(self._forecast_dataset) if self._forecast_dataset is not None else 0
        num_replicas = max(1, self.args.world_size)
        rank = self.args.process_index

        # Exact costs (dynamic batching and length bucketing share one algorithm): LLM tokens =
        # cached text token count + soft token count from the model's real qformer.plan; chronos
        # side = patch count (P x C). Bucketing sort key = LLM tokens + patches (same "text + time
        # series" notion as the old cost_keys, but exact).
        pf = self._length_pool_factor
        u_costs = f_costs = None
        u_tok = f_tok = u_patch = f_patch = None
        if (pf or self._u_token_budget or self._f_token_budget) and (n_u or n_f):
            base = self.model
            base = base.get_base_model() if hasattr(base, "get_base_model") else base
            patch = base.input_patch_size

            def _budget_costs(ds, cap, allow_upsample):
                text = ds.token_lengths(cache_dir=self._token_cache_dir)
                pc = ds.history_patches(cap, patch=patch)
                # Soft-token / chronos-patch costs follow the model's real accounting (including
                # upsampling and the stats tokens), matching what _encode_history_to_soft_prompt
                # actually produces per sample. The forecast branch uses allow_upsample=False (no
                # upsampling) => its cost excludes upsampled patches, consistently.
                llm = [t + base.soft_token_count(p, c, allow_upsample) for t, (p, c) in zip(text, pc)]
                return llm, [base._effective_patches(p, allow_upsample) * c for p, c in pc]

            if n_u:
                u_tok, u_patch = _budget_costs(self._understanding_dataset,
                                               self._understanding_max_context, allow_upsample=True)
                if pf:
                    u_costs = [t + p for t, p in zip(u_tok, u_patch)]
            if n_f:
                f_tok, f_patch = _budget_costs(self._forecast_dataset,
                                               self._forecast_max_context, allow_upsample=False)
                if pf:
                    f_costs = [t + p for t, p in zip(f_tok, f_patch)]
        batch_sampler = DualBranchBatchSampler(
            n_understanding=n_u,
            n_forecast=n_f,
            understanding_bs=self._understanding_bs,
            forecast_bs=self._forecast_bs,
            num_replicas=num_replicas,
            rank=rank,
            seed=self._sampler_seed,
            drop_last=True,
            understanding_costs=u_costs,
            forecast_costs=f_costs,
            pool_factor=pf,
            understanding_token_costs=u_tok,
            forecast_token_costs=f_tok,
            understanding_patch_costs=u_patch,
            forecast_patch_costs=f_patch,
            # Budgets gated per branch: in single-branch training a leftover budget for the other
            # branch (scripts often pass both) would trip the sampler's "budget>0 requires costs"
            # check on the branch with n=0.
            understanding_token_budget=self._u_token_budget if n_u else 0,
            forecast_token_budget=self._f_token_budget if n_f else 0,
            patch_budget=self._patch_budget,
            max_dynamic_bs=self._max_dynamic_bs,
            bs_ladder=self._dynamic_bs_ladder,
            patch_token_weight=self._patch_token_weight,
            understanding_repeats=self._u_epoch_repeats,
            forecast_repeats=self._f_epoch_repeats,
        )
        self._dual_batch_sampler = batch_sampler
        collator = ChronosLLMCollator(
            self.processing_class,
            understanding_max_context=self._understanding_max_context,
            forecast_max_context=self._forecast_max_context,
        )
        return _EpochAwareDataLoader(
            concat,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_train_sampler(self, *args, **kwargs):
        return None  # we use a batch_sampler; no plain sampler needed

    def _save(self, output_dir=None, state_dict=None):
        # Reuse the default _save: for a PeftModel it stores the adapter (LoRA + modules_to_save) +
        # tokenizer + training_args.
        super()._save(output_dir, state_dict)
        # Add just one thing: the base ChronosLLMConfig (config.json) so from_pretrained can rebuild
        # the base from chronos_ckpt/llm_path (by default only adapter_config.json is produced, which
        # lacks these two paths).
        output_dir = output_dir or self.args.output_dir
        base = self.accelerator.unwrap_model(self.model)
        base = base.get_base_model() if hasattr(base, "get_base_model") else base
        base.config.save_pretrained(output_dir)
