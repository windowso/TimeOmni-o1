"""CPU unit test for ``freeze_ts_modules`` (train only the LLM LoRA, freeze the perception path)
(tiny Qwen2 + real chronos-2).

Checks:
1. After freezing, no parameter of chronos2 / the two Q-formers has requires_grad; all LLM LoRA
   parameters remain trainable.
2. After a training step the LoRA weights change while the chronos/Q-former weights are identical
   value for value (genuinely not updated).
3. ``create_optimizer`` keeps only the llm group (the chronos/qformer/gate groups simply disappear,
   without errors).
"""

import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, TrainingArguments

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import ConcatBranchDataset
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.models.chronos_llm_model import (
    ChronosLLM, ChronosLLMConfig, add_lora, freeze_ts_modules, _add_ts_special_tokens,
)
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn
from chronos_llm.trainer import ChronosLLMTrainer

CHRONOS = os.environ.get("CHRONOS2_PATH", "checkpoints/chronos-2")
LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")
PARQUET = os.environ.get("FORECAST_PARQUET", "data/forecast/mmtr_forecast_corpus.parquet")
# Shipped MMTR understanding test file; its ori_path values are relative to the repository root, so
# run from the repo root (no base_dir needed).
JSONL = os.environ.get("UNDERSTANDING_JSONL", "data/understanding/st-bench/stbench_test.jsonl")

TS_KEYS = ("chronos", "history_qformer", "feedback_qformer")


def main():
    for p in (CHRONOS, LLM, PARQUET, JSONL):
        if not os.path.exists(p):
            print(f"SKIP: required resource not found: {p}")
            return
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

    n_lora_before = sum(1 for n, p in model.named_parameters()
                        if "lora_" in n and p.requires_grad)
    assert n_lora_before > 0, "there should be trainable LoRA parameters after add_lora"
    n_fr = freeze_ts_modules(model)
    assert n_fr > 0, "there should be frozen perception-path parameters"

    # 1. the frozen set is correct
    for n, p in model.named_parameters():
        if any(k in n for k in TS_KEYS):
            assert not p.requires_grad, f"perception-path parameter not frozen: {n}"
    n_lora_after = sum(1 for n, p in model.named_parameters()
                       if "lora_" in n and p.requires_grad)
    assert n_lora_after == n_lora_before, "trainable LoRA parameters must not be frozen"

    # 2/3. train two steps: LoRA updates, perception path identical value for value; optimizer
    # groups contain no frozen parameters (trainer setup copied from test_optimizer_groups --
    # ConcatBranchDataset + dual-branch arguments)
    u_ds = UnderstandingJsonlDataset([JSONL], tok, max_user_tokens=256, max_tokens=768)
    u_ds.data = u_ds.data[:4]
    f_ds = ForecastParquetDataset(PARQUET, tok, split=None, max_user_tokens=256,
                                  max_tokens=768, max_rows=4)
    concat = ConcatBranchDataset(u_ds, f_ds)
    snap_ts = {n: p.detach().clone() for n, p in model.named_parameters()
               if any(k in n for k in TS_KEYS)}
    snap_lora = {n: p.detach().clone() for n, p in model.named_parameters()
                 if "lora_" in n and p.requires_grad}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        args = TrainingArguments(output_dir=td, per_device_train_batch_size=2, max_steps=2,
                                 learning_rate=2e-5, logging_steps=1, save_steps=10_000,
                                 report_to=[], remove_unused_columns=False,
                                 label_names=["labels"], use_cpu=True)
        trainer = ChronosLLMTrainer(model=model, args=args, train_dataset=concat,
                                    processing_class=tok,
                                    understanding_dataset=u_ds, forecast_dataset=f_ds,
                                    understanding_bs=2, forecast_bs=2,
                                    understanding_max_context=2048, forecast_max_context=512,
                                    lr_lora=1e-3, lr_qformer=3e-4, lr_gate=1e-3)
        opt = trainer.create_optimizer()
        trained = {id(p) for g in opt.param_groups for p in g["params"]}
        for n, p in model.named_parameters():
            if any(k in n for k in TS_KEYS):
                assert id(p) not in trained, f"frozen parameter entered the optimizer: {n}"
        trainer.train()
    changed = sum((model.get_parameter(n) - v).abs().max().item() > 0
                  for n, v in snap_lora.items())
    assert changed > 0, "LoRA weights should have changed after training"
    for n, v in snap_ts.items():
        assert torch.equal(model.get_parameter(n), v), f"frozen parameter was updated: {n}"
    print(f"OK test_freeze_ts: {n_fr} tensors frozen, {n_lora_after} LoRA tensors trainable and {changed} updated")


if __name__ == "__main__":
    main()
