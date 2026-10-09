"""Milestone-4 integration check (CPU, tiny Qwen2 + a little real data): run the HF Trainer for a few steps and verify
that DualBranchBatchSampler yields homogeneous batches and that the collator, compute_loss and optimizer step all work
end to end.
"""

import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, TrainingArguments

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import ConcatBranchDataset
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.models.chronos_llm_model import (
    ChronosLLM, ChronosLLMConfig, add_lora, _add_ts_special_tokens,
)
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn
from chronos_llm.trainer import ChronosLLMTrainer

CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")
PARQUET = os.environ.get("FORECAST_PARQUET", "data/forecast/mmtr_forecast_corpus.parquet")
# Shipped MMTR understanding test set; its ori_path values are relative to the repository root, so run from the
# repo root (base_dir=None). Both are overridable through the environment.
JSONL = os.environ.get("UNDERSTANDING_JSONL", "data/understanding/opentslm/sleep_test.jsonl")
JSONL_BASE_DIR = os.environ.get("UNDERSTANDING_BASE_DIR")  # None => ori_path resolved as given


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    _add_ts_special_tokens(tok)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                      max_position_embeddings=8192)
    llm = Qwen2ForCausalLM(cfg)
    llm.config.use_cache = False
    chronos = load_chronos2_with_cross_attn(CHRONOS); chronos.train()
    base = ChronosLLM(
        ChronosLLMConfig(chronos_ckpt=CHRONOS, llm_path=LLM, sw_queries_per_window=2,
                         sw_target_windows=8, sw_min_windows=2,
                         fb_num_query_tokens=8, qformer_num_heads=4),
        chronos, llm, tok,
    )
    model = add_lora(base, r=8, alpha=16, dropout=0.0)
    print("Trainable parameters:\n" + model.get_base_model().trainable_parameter_summary())

    f_ds = ForecastParquetDataset(PARQUET, tok, split=None, max_user_tokens=256, max_tokens=768, max_rows=8)
    u_ds = UnderstandingJsonlDataset([JSONL], tok, base_dir=JSONL_BASE_DIR, max_user_tokens=256, max_tokens=768)
    u_ds.data = u_ds.data[:8]
    concat = ConcatBranchDataset(u_ds, f_ds)

    targs = TrainingArguments(
        output_dir="/tmp/chronos_llm_test", per_device_train_batch_size=2,
        max_steps=4, logging_steps=1, save_steps=10_000, use_cpu=True,
        remove_unused_columns=False, label_names=["labels"], report_to=[],
        learning_rate=1e-4,
    )
    trainer = ChronosLLMTrainer(
        model=model, args=targs, train_dataset=concat, processing_class=tok,
        understanding_dataset=u_ds, forecast_dataset=f_ds,
        understanding_bs=2, forecast_bs=2,
        understanding_max_context=2048, forecast_max_context=512,
    )
    # Check that the dataloader yields homogeneous batches (both understanding and forecast)
    dl = trainer.get_train_dataloader()
    branches = [b["branch"] for b in dl]
    print("batch branches:", branches)
    assert "forecast" in branches and "understanding" in branches, "both kinds of homogeneous batch should appear"

    out = trainer.train()
    print("final train loss:", out.training_loss, "finite:", out.training_loss == out.training_loss)
    assert out.training_loss == out.training_loss  # not nan
    print("TRAINER INTEGRATION PASSED")


if __name__ == "__main__":
    main()
