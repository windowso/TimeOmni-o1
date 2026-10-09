"""Single-branch training (understanding only / forecasting only, the paths used by
train_understanding.sh / train_forecast.sh): ConcatBranchDataset / DualBranchBatchSampler /
trainer must be trainable end to end when the other side's dataset is None, every batch belongs to
the corresponding branch, and the loss is finite.
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
# Shipped MMTR understanding test set; its ori_path values are relative to the repository root,
# so run from the repo root (no base_dir needed).
JSONL = os.environ.get("UNDERSTANDING_JSONL", "data/understanding/opentslm/sleep_test.jsonl")


def _run_branch(model, tok, u_ds, f_ds, expect_branch, out_dir):
    concat = ConcatBranchDataset(u_ds, f_ds)
    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=2,
        max_steps=2, logging_steps=1, save_steps=10_000, use_cpu=True,
        remove_unused_columns=False, label_names=["labels"], report_to=[],
        learning_rate=1e-4,
    )
    trainer = ChronosLLMTrainer(
        model=model, args=targs, train_dataset=concat, processing_class=tok,
        understanding_dataset=u_ds, forecast_dataset=f_ds,
        understanding_bs=2, forecast_bs=2,
        understanding_max_context=2048, forecast_max_context=512,
    )
    branches = [b["branch"] for b in trainer.get_train_dataloader()]
    assert branches and all(b == expect_branch for b in branches), \
        f"with {expect_branch} data only, every batch should belong to that branch; got {branches}"
    out = trainer.train()
    assert out.training_loss == out.training_loss, f"{expect_branch} single-branch training loss is NaN"
    print(f"single branch {expect_branch}: {len(branches)} batches, loss={out.training_loss:.4f} OK")


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

    u_ds = UnderstandingJsonlDataset([JSONL], tok, max_user_tokens=256, max_tokens=768)
    u_ds.data = u_ds.data[:8]
    f_ds = ForecastParquetDataset(PARQUET, tok, split=None, max_user_tokens=256, max_tokens=768, max_rows=8)

    _run_branch(model, tok, u_ds, None, "understanding", "/tmp/chronos_llm_test_u_only")
    _run_branch(model, tok, None, f_ds, "forecast", "/tmp/chronos_llm_test_f_only")
    print("SINGLE BRANCH TRAINING PASSED")


if __name__ == "__main__":
    main()
