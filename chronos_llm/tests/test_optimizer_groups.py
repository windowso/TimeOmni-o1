"""CPU unit test for grouped learning rates + gate monitoring (tiny Qwen2 + real Chronos-2).

Checks:
1. ``create_optimizer`` puts every trainable parameter into exactly one group (none missing, none duplicated, no
   frozen ones): gate->lr_gate, Q-former->lr_qformer, LLM (LoRA)->lr_lora, the rest of chronos->learning_rate.
2. The training log carries ``gate_tanh_mean/max`` (cold-start gate=0 => 0.0) and the component losses are still logged.
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

LR_BASE, LR_LORA, LR_QF, LR_GATE = 2e-5, 1e-4, 3e-4, 1e-3


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

    f_ds = ForecastParquetDataset(PARQUET, tok, split=None, max_user_tokens=256, max_tokens=768, max_rows=4)
    u_ds = UnderstandingJsonlDataset([JSONL], tok, base_dir=JSONL_BASE_DIR, max_user_tokens=256, max_tokens=768)
    u_ds.data = u_ds.data[:4]
    concat = ConcatBranchDataset(u_ds, f_ds)

    targs = TrainingArguments(
        output_dir="/tmp/chronos_llm_optgroups", per_device_train_batch_size=2,
        max_steps=2, logging_steps=1, save_steps=10_000, use_cpu=True,
        remove_unused_columns=False, label_names=["labels"], report_to=[],
        learning_rate=LR_BASE, weight_decay=0.01,
    )
    trainer = ChronosLLMTrainer(
        model=model, args=targs, train_dataset=concat, processing_class=tok,
        understanding_dataset=u_ds, forecast_dataset=f_ds,
        understanding_bs=2, forecast_bs=2,
        understanding_max_context=2048, forecast_max_context=512,
        lr_lora=LR_LORA, lr_qformer=LR_QF, lr_gate=LR_GATE,
    )

    # ---- (1) grouping correctness
    opt = trainer.create_optimizer()
    id2lr = {}
    for grp in opt.param_groups:
        for p in grp["params"]:
            assert id(p) not in id2lr, "the same parameter appears in several groups"
            id2lr[id(p)] = grp["lr"]
    named_trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(id2lr) == len(named_trainable), (
        f"optimizer covers {len(id2lr)} parameters != trainable {len(named_trainable)}"
    )

    def lr_of(substr_filter):
        sel = {n: id2lr[id(p)] for n, p in named_trainable.items() if substr_filter(n)}
        assert sel, "the probe matched no parameter"
        return sel

    for n, lr in lr_of(lambda n: n.endswith("cross_attn.gate")).items():
        assert lr == LR_GATE, (n, lr)
    for n, lr in lr_of(lambda n: "history_qformer" in n or "feedback_qformer" in n).items():
        assert lr == LR_QF, (n, lr)
    # The cross-attn projections/LN newly injected into chronos are initialised from scratch and belong to the qformer group (not chronos's 2e-5)
    for n, lr in lr_of(lambda n: ".cross_attn." in "." + n and not n.endswith(".gate")).items():
        assert lr == LR_QF, (n, lr)
    for n, lr in lr_of(lambda n: ".llm." in "." + n).items():
        assert lr == LR_LORA, (n, lr)
    for n, lr in lr_of(lambda n: ".chronos." in "." + n and ".cross_attn." not in "." + n).items():
        assert lr == LR_BASE, (n, lr)
    # The gate is exempt from weight decay (targs.weight_decay=0.01): otherwise decoupled decay keeps pulling the zero-initialised gate back to 0
    id2wd = {id(p): grp["weight_decay"] for grp in opt.param_groups for p in grp["params"]}
    for g in trainer._gate_params():
        assert id2wd[id(g)] == 0.0, "the gate must not be pulled back to 0 by weight decay"
    # Frozen parameters are not in the optimizer
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert id(p) not in id2lr, f"frozen parameter entered the optimizer: {n}"
    n_gates = len(trainer._gate_params())
    print(f"grouped learning rates OK: gate({n_gates})={LR_GATE} qformer={LR_QF} lora={LR_LORA} chronos={LR_BASE}, "
          f"all {len(id2lr)} trainable parameters covered")

    # ---- (2) train 2 steps; the log carries the gate opening (cold start = 0) and the component losses
    trainer.train()
    train_logs = [l for l in trainer.state.log_history if "loss" in l and "train_runtime" not in l]
    assert train_logs, "no training logs"
    for l in train_logs:
        assert "gate_tanh_mean" in l and "gate_tanh_max" in l, l
    assert train_logs[0]["gate_tanh_mean"] == 0.0, train_logs[0]  # gate zero-initialised
    assert any("text_loss" in l for l in train_logs)
    print("gate monitoring OK:", {k: v for k, v in train_logs[-1].items()
                                  if k.startswith("gate_") or k.endswith("_loss")})
    print("OPTIMIZER GROUPS PASSED")


if __name__ == "__main__":
    main()
