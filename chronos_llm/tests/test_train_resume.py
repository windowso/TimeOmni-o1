"""Training checkpoint + resume check (CPU, tiny Qwen2 + a little real data).

Phase 1 trains 2 steps and saves checkpoint-2; phase 2 rebuilds a base with the same weights + add_lora and resumes
from checkpoint-2 up to 4 steps. An on_step_end counter proves **conclusively** that phase 2 ran only 2 new optimizer
steps (it really resumed from step 2 instead of restarting from 0). It also checks that the checkpoint directory
contains adapter + optimizer + trainer_state (i.e. our custom _save does not break resumability).
"""
import gc
import json
import os
import tempfile

import torch
from transformers import TrainerCallback, TrainingArguments

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import ConcatBranchDataset
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.models.chronos_llm_model import add_lora
from chronos_llm.tests.test_pretrained_peft import _build_tiny_base
from chronos_llm.trainer import ChronosLLMTrainer

PARQUET = os.environ.get("FORECAST_PARQUET", "data/forecast/mmtr_forecast_corpus.parquet")
# Shipped MMTR understanding test set; its ori_path values are relative to the repository root, so run from the
# repo root (base_dir=None). Both are overridable through the environment.
JSONL = os.environ.get("UNDERSTANDING_JSONL", "data/understanding/opentslm/sleep_test.jsonl")
JSONL_BASE_DIR = os.environ.get("UNDERSTANDING_BASE_DIR")  # None => ori_path resolved as given


class StepCounter(TrainerCallback):
    def __init__(self):
        self.n = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.n += 1


def _datasets(tok):
    f_ds = ForecastParquetDataset(PARQUET, tok, split=None, max_user_tokens=256, max_tokens=768, max_rows=8)
    u_ds = UnderstandingJsonlDataset([JSONL], tok, base_dir=JSONL_BASE_DIR, max_user_tokens=256, max_tokens=768)
    u_ds.data = u_ds.data[:8]
    return u_ds, f_ds


def _make_trainer(model, tok, u_ds, f_ds, out, max_steps, ignore_data_skip=False, cbs=()):
    targs = TrainingArguments(
        output_dir=out, per_device_train_batch_size=2, max_steps=max_steps,
        save_steps=2, save_total_limit=5, logging_steps=1, use_cpu=True,
        ignore_data_skip=ignore_data_skip,
        remove_unused_columns=False, label_names=["labels"], report_to=[], learning_rate=1e-4,
    )
    return ChronosLLMTrainer(
        model=model, args=targs, train_dataset=ConcatBranchDataset(u_ds, f_ds),
        processing_class=tok, understanding_dataset=u_ds, forecast_dataset=f_ds,
        understanding_bs=2, forecast_bs=2,
        understanding_max_context=2048, forecast_max_context=512,
        sampler_seed=0, callbacks=list(cbs),
    )


def main():
    torch.manual_seed(0)
    base, tok = _build_tiny_base()
    clean_sd = {k: v.clone() for k, v in base.state_dict().items()}
    model1 = add_lora(base, r=8, alpha=16, dropout=0.0)
    u_ds, f_ds = _datasets(tok)

    with tempfile.TemporaryDirectory() as d:
        # Phase 1: train 2 steps -> checkpoint-2
        _make_trainer(model1, tok, u_ds, f_ds, d, max_steps=2).train()
        ckpt = os.path.join(d, "checkpoint-2")
        for f in ["adapter_model.safetensors", "optimizer.pt", "trainer_state.json"]:
            assert os.path.exists(os.path.join(ckpt, f)), f"checkpoint is missing {f}"
        st = json.load(open(os.path.join(ckpt, "trainer_state.json")))
        assert st["global_step"] == 2, st["global_step"]
        print("phase 1: checkpoint-2 complete (adapter+optimizer+trainer_state), global_step=2")

        # Release the phase-1 objects to lower the peak memory (clean_sd is kept for the phase-2 rebuild).
        del model1, base
        gc.collect()

        # Phase 2: rebuild a base with the same weights + add_lora, resume from checkpoint-2 up to 4 steps
        base2, tok2 = _build_tiny_base()
        base2.load_state_dict(clean_sd, strict=True)
        model2 = add_lora(base2, r=8, alpha=16, dropout=0.0)
        u2, f2 = _datasets(tok2)
        counter = StepCounter()
        t2 = _make_trainer(model2, tok2, u2, f2, d, max_steps=4, ignore_data_skip=True, cbs=[counter])
        t2.train(resume_from_checkpoint=ckpt)
        assert t2.state.global_step == 4, t2.state.global_step
        assert counter.n == 2, f"should have trained only 2 new steps (from step 2), got {counter.n} (a restart from 0 would give 4)"
        print(f"phase 2: resume succeeded, trained {counter.n} more steps -> global_step={t2.state.global_step}")

    print("TRAIN RESUME PASSED")


if __name__ == "__main__":
    main()
