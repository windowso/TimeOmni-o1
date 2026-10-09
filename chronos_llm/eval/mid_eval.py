"""Automatic evaluation during training at an epoch interval (``MidTrainEvalCallback``).

Mechanism: ``on_epoch_end`` unwraps PEFT -> ChronosLLM and runs the inference loops of eval/infer_* directly
on the model being trained (parameters are fully resident on the GPU -- ZeRO-2 shards only the optimizer
state; ZeRO-3 shards parameters and is not supported). The dist process group is already initialised by
training, so the strided sharding + rank-0 merge of ``run_*_infer`` apply automatically (all training ranks
evaluate together); rank 0 then computes the metric CSVs under ``{output_dir}/mid_eval/epoch_{E}/``.
``generate_*`` saves/restores eval()/train() mode and runs under ``@torch.no_grad``, so evaluation does not
disturb the training state.

The task follows the training branch automatically (decided where train.py assembles the callback): the
understanding branch must be given test jsonl files explicitly (the training jsonl is the train split); the
forecasting branch defaults to ``split=test`` of the training parquet. The forecasting branch evaluates
**both settings** by default (``forecast_teacher_forced=True``): generate = autoregressively generated
reasoning (``forecast_metrics.csv``), teacher_forced = the true reasoning/conclusion fed in and its hidden
states fed back into Chronos-2 (``forecast_metrics_tf.csv``) -- the gap between the two is the exposure
bias, tracked per epoch as it converges. The understanding branch has no teacher-forced evaluation (feeding
the ground-truth text would make the text metrics trivially perfect and meaningless).
"""
import os

from torch.utils.data import Subset
from transformers import TrainerCallback

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.eval import eval_forecast, eval_understanding
from chronos_llm.eval.dist_utils import barrier, dist_info
from chronos_llm.eval.infer_forecast import run_forecast_infer
from chronos_llm.eval.infer_understanding import run_understanding_infer


class MidTrainEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir, every_epochs=1,
                 understanding_jsonl=None, understanding_base_dir=None,
                 understanding_limit=200, understanding_bs=32, understanding_max_new_tokens=200,
                 forecast_parquet=None, forecast_limit=500, forecast_bs=16,
                 forecast_max_new_tokens=320, forecast_teacher_forced=True,
                 max_user_tokens=1500, max_tokens=4096,
                 sample_chunks=0, overview_chunk=True, sample_window=8192,
                 understanding_no_reasoning=False):
        self.tok = tokenizer
        self.output_dir = output_dir
        self.every = every_epochs
        self.u_files = list(understanding_jsonl or [])
        self.u_base_dir = understanding_base_dir
        self.u_limit = understanding_limit
        self.u_bs = understanding_bs
        self.u_new_tokens = understanding_max_new_tokens
        # Same source as the training-side --understanding_no_reasoning: under no-CoT training the inference
        # prefix must switch to "complete empty think, continue with the answer only", otherwise mid-eval
        # would make the model continue a reasoning segment it never learned to produce.
        self.u_no_reasoning = understanding_no_reasoning
        self.f_parquet = forecast_parquet
        self.f_limit = forecast_limit
        self.f_bs = forecast_bs
        self.f_new_tokens = forecast_max_new_tokens
        self.f_tf = forecast_teacher_forced
        self.max_user_tokens = max_user_tokens
        self.max_tokens = max_tokens
        self.sample_chunks = sample_chunks
        self.overview_chunk = overview_chunk
        self.sample_window = sample_window
        self._done_epochs = set()
        self._llm_only = False  # refreshed from the model config in on_epoch_end (llm_only => ts_as_text evaluation protocol)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch or 0))
        if self.every <= 0 or epoch <= 0 or epoch % self.every or epoch in self._done_epochs:
            return
        self._done_epochs.add(epoch)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        device = next(base.parameters()).device
        self._llm_only = bool(getattr(base.config, "llm_only", False))
        rank, _ = dist_info()
        out_root = os.path.join(self.output_dir, "mid_eval", f"epoch_{epoch}")
        if rank == 0:
            print(f"\n===== mid-train eval @ epoch {epoch} =====", flush=True)
        if self.u_files:
            infer_dir = os.path.join(out_root, "understanding")
            run_understanding_infer(base, self.tok, self.u_files, infer_dir,
                                    base_dir=self.u_base_dir, batch_size=self.u_bs,
                                    max_new_tokens=self.u_new_tokens, device=device,
                                    max_user_tokens=self.max_user_tokens,
                                    max_tokens=self.max_tokens, num_workers=0,
                                    limit=self.u_limit,
                                    sample_chunks=self.sample_chunks,
                                    overview_chunk=self.overview_chunk,
                                    sample_window=self.sample_window,
                                    no_reasoning=self.u_no_reasoning,
                                    ts_as_text=self._llm_only)
            if rank == 0:
                eval_understanding.evaluate_all_files(
                    infer_dir, os.path.join(out_root, "understanding_metrics.csv"))
        if self.f_parquet:
            # Two settings, generate (autoregressively generated reasoning) + teacher_forced (true
            # reasoning/conclusion fed in), each producing its own npz/csv (the latter with a _tf suffix); the
            # gap between the two metric sets is the per-epoch size of the exposure bias.
            self._eval_forecast(base, device, rank, out_root, teacher_forced=False)
            if self.f_tf and not self._llm_only:  # llm_only has no tf protocol (the numbers come from the generated text itself)
                self._eval_forecast(base, device, rank, out_root, teacher_forced=True)
        barrier()  # all ranks return to training together once the metrics are done, so rank 0 does not lag behind the next epoch's collectives
        if rank == 0:
            print(f"===== mid-train eval @ epoch {epoch} done -> {out_root} =====", flush=True)

    def _eval_forecast(self, base, device, rank, out_root, *, teacher_forced):
        """Run the forecasting evaluation for one setting (generate or teacher_forced), each with its own npz/csv.

        - generate (teacher_forced=False): inference-prefix rendering (inference_mode=True), reasoning is
          generated autoregressively -> ``forecast_preds.npz`` / ``forecast_metrics.csv``;
        - teacher_forced=True: training rendering (inference_mode=False, input_ids contain the true
          reasoning/conclusion), a single forward pass whose hidden states are fed back into Chronos-2 (no
          autoregression) -> ``forecast_preds_tf.npz`` / ``forecast_metrics_tf.csv``. Both datasets carry
          their own future/roi_mask and the metrics are computed independently (no cross-mode row alignment needed).
        """
        suffix = "_tf" if teacher_forced else ""
        # prompt_only (no-reasoning) checkpoints: evaluation must also use the plain-prompt-only ids (otherwise generate
        # would autoregressively produce arbitrary text and tf would feed true reasoning, both inconsistent
        # with the training regime). Read from the model config, same protocol as training.
        prompt_only = bool(getattr(base.config, "forecast_prompt_only", False))
        ds = ForecastParquetDataset(self.f_parquet, self.tok, split="test",
                                    inference_mode=not teacher_forced,
                                    max_user_tokens=self.max_user_tokens,
                                    max_tokens=self.max_tokens,
                                    forecast_prompt_only=prompt_only,
                                    ts_as_text=getattr(self, "_llm_only", False),
                                    emit_meta=True)  # tf uses the training rendering but still needs the real id/dataset_name
        if self.f_limit and self.f_limit < len(ds):
            ds = Subset(ds, range(self.f_limit))
        npz = os.path.join(out_root, f"forecast_preds{suffix}.npz")
        run_forecast_infer(base, self.tok, ds, npz, batch_size=self.f_bs,
                           max_new_tokens=self.f_new_tokens, device=device, num_workers=0,
                           teacher_forced=teacher_forced)
        if rank == 0:
            eval_forecast.main(["--pred", npz, "--output_csv",
                                os.path.join(out_root, f"forecast_metrics{suffix}.csv")])
