"""End-to-end check: after real training through ChronosLLMTrainer, the LoRA and chronos/qformer parameters are
actually updated by the optimizer (the frozen base LLM is not), the production save path
(trainer.save_model -> _save) writes adapter + config.json, and reloading with ChronosLLM.from_pretrained gives
forward values that match the trained model element-wise.

It closes the links the existing tests do not connect:
1. "It really trains" -- compare parameter snapshots before/after training: the trainable groups (lora_/chronos
   body/cross_attn/the two qformers) change by a non-zero amount; the frozen base LLM weights stay exactly unchanged.
2. "It saves correctly" -- go through the trainer's _save (HF default adapter save + our extra ChronosLLMConfig
   config.json), assert that the saved adapter_model.safetensors really contains lora_* and
   chronos.*/history_qformer.*/feedback_qformer.* keys, and that config.json carries chronos_ckpt/llm_path.
3. "It reloads end to end" -- from_pretrained(merge=False) rebuilds the base (from_config is mocked to load the clean
   pre-training weights) + stacks the trained adapter; understanding/forecast forward values match the trained model
   element-wise.

CPU + tiny Qwen2 (stand-in for the 9B) + real Chronos-2 + a little real data.
"""
import gc
import json
import os
import tempfile
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from transformers import TrainingArguments

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import ConcatBranchDataset
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.models.chronos_llm_model import ChronosLLM, add_lora
from chronos_llm.tests.test_pretrained_peft import _build_tiny_base, _toy_batch
from chronos_llm.trainer import ChronosLLMTrainer

PARQUET = os.environ.get("FORECAST_PARQUET", "data/forecast/mmtr_forecast_corpus.parquet")
# Shipped MMTR understanding test set; its ori_path values are relative to the repository root, so run from the
# repo root (base_dir=None). Both are overridable through the environment.
JSONL = os.environ.get("UNDERSTANDING_JSONL", "data/understanding/opentslm/sleep_test.jsonl")
JSONL_BASE_DIR = os.environ.get("UNDERSTANDING_BASE_DIR")  # None => ori_path resolved as given


def _datasets(tok):
    f_ds = ForecastParquetDataset(PARQUET, tok, split=None, max_user_tokens=256, max_tokens=768, max_rows=8)
    u_ds = UnderstandingJsonlDataset([JSONL], tok, base_dir=JSONL_BASE_DIR, max_user_tokens=256, max_tokens=768)
    u_ds.data = u_ds.data[:8]
    return u_ds, f_ds


def _pick(named, pred):
    for k, p in named.items():
        if pred(k, p):
            return k
    raise AssertionError("no parameter matches this probe condition")


def main():
    torch.manual_seed(0)
    base, tok = _build_tiny_base()
    # Snapshot the clean base weights before training (before add_lora, so keys are not yet rewritten by PEFT): the
    # frozen base LLM + the initial chronos/qformer. Fed back through the mocked from_config on reload -- its
    # chronos/qformer get overwritten by the trained adapter, while the frozen base LLM matches the trained model,
    # so the forward can be compared element-wise.
    clean_sd = {k: v.clone() for k, v in base.state_dict().items()}

    model = add_lora(base, r=8, alpha=16, dropout=0.0)
    # Cold-start gate zero-init -> tanh(0)=0 closes the feedback path; warm it to 0.3 so the feedback path through
    # feedback_qformer/cross_attn is really used in training and its updates are observable (same connectivity
    # approach as test_add_lora).
    gbase = model.get_base_model()
    for blk in gbase.chronos.encoder.block:
        blk.cross_attn.gate.data.fill_(0.3)

    named = dict(model.named_parameters())
    probes = {
        "lora":             _pick(named, lambda k, p: "lora_" in k and p.requires_grad),
        "chronos_body":     _pick(named, lambda k, p: ".chronos." in k and "cross_attn" not in k and p.requires_grad),
        "chronos_xattn":    _pick(named, lambda k, p: ".chronos." in k and "cross_attn" in k and "gate" not in k and p.requires_grad),
        "history_qformer":  _pick(named, lambda k, p: "history_qformer" in k and p.requires_grad),
        "feedback_qformer": _pick(named, lambda k, p: "feedback_qformer" in k and p.requires_grad),
        "frozen_llm":       _pick(named, lambda k, p: ".llm." in k and "lora_" not in k and "modules_to_save" not in k and not p.requires_grad),
    }
    before = {name: named[key].detach().clone() for name, key in probes.items()}
    print("probe parameters:")
    for name, key in probes.items():
        print(f"  {name:18s} <- {key}")

    u_ds, f_ds = _datasets(tok)
    with tempfile.TemporaryDirectory() as d:
        out_dir = os.path.join(d, "final")
        targs = TrainingArguments(
            output_dir=os.path.join(d, "run"), per_device_train_batch_size=2,
            max_steps=4, logging_steps=1, save_steps=10_000, use_cpu=True,
            remove_unused_columns=False, label_names=["labels"], report_to=[],
            learning_rate=5e-4, seed=0,
        )
        trainer = ChronosLLMTrainer(
            model=model, args=targs, train_dataset=ConcatBranchDataset(u_ds, f_ds),
            processing_class=tok, understanding_dataset=u_ds, forecast_dataset=f_ds,
            understanding_bs=2, forecast_bs=2,
            understanding_max_context=2048, forecast_max_context=512, sampler_seed=0,
        )
        # Confirm both branches appear within these 4 steps (guarantees a forecast step exists -> the feedback path is trained).
        branches = [b["branch"] for b in trainer.get_train_dataloader()]
        print("batch branches:", branches)
        assert "forecast" in branches and "understanding" in branches

        out = trainer.train()
        assert out.training_loss == out.training_loss, "train loss is NaN"
        print("final train loss:", out.training_loss)

        # ---- (1) parameters really trained: trainable change, frozen do not ----
        named_after = dict(model.named_parameters())
        print("parameter change before/after training (max|delta|):")
        deltas = {}
        for name, key in probes.items():
            delta = (named_after[key].detach() - before[name]).abs().max().item()
            deltas[name] = delta
            print(f"  {name:18s} {delta:.3e}")
        for name in ("lora", "chronos_body", "chronos_xattn", "history_qformer", "feedback_qformer"):
            assert deltas[name] > 1e-9, f"{name} unchanged after training (delta={deltas[name]:.2e}); not updated by the optimizer"
        assert deltas["frozen_llm"] == 0.0, f"the frozen base LLM was modified (delta={deltas['frozen_llm']:.2e})"
        print("(1) trainable (lora/chronos/both qformers) all updated, frozen base LLM unchanged [OK]")

        # ---- post-training forward reference (one fixed batch each for understanding + forecast, reused for the reload comparison) ----
        model.eval()
        eval_u = _toy_batch(model.get_base_model(), tok, "understanding")
        eval_f = _toy_batch(model.get_base_model(), tok, "forecast")
        with torch.no_grad():
            ref_u = model(eval_u)["loss"].item()
            ref_f = model(eval_f)["loss"].item()

        # ---- (2) production save path trainer.save_model -> _save ----
        trainer.save_model(out_dir)
        for f in ("adapter_model.safetensors", "adapter_config.json", "config.json"):
            assert os.path.exists(os.path.join(out_dir, f)), f"save dir is missing {f}"
        saved_cfg = json.load(open(os.path.join(out_dir, "config.json")))
        assert saved_cfg.get("model_type") == "chronos_llm"
        assert saved_cfg.get("chronos_ckpt") and saved_cfg.get("llm_path"), \
            "config.json does not carry chronos_ckpt/llm_path (from_pretrained needs them to rebuild the base)"
        adapter_keys = list(load_file(os.path.join(out_dir, "adapter_model.safetensors")).keys())
        assert any("lora_" in k for k in adapter_keys), "adapter contains no LoRA weights"
        for sub in ("chronos", "history_qformer", "feedback_qformer"):
            assert any(sub in k for k in adapter_keys), f"adapter does not contain {sub} (modules_to_save not saved)"
        print(f"(2) trainer._save wrote adapter ({len(adapter_keys)} keys incl. lora+chronos+both qformers)"
              f" + config.json (with chronos_ckpt/llm_path) [OK]")

        # Release the training objects to lower the peak memory during reload.
        del trainer, model, base, gbase
        gc.collect()

        # ---- (3) forward after from_pretrained reload matches the trained model element-wise ----
        def fake_from_config(config):
            b2, _ = _build_tiny_base()
            b2.load_state_dict(clean_sd, strict=True)  # load the clean pre-training weights; chronos/qformer are then overwritten by the adapter
            return b2

        with patch.object(ChronosLLM, "from_config", staticmethod(fake_from_config)):
            loaded = ChronosLLM.from_pretrained(out_dir, merge=False)  # PeftModel, keeps the trained adapter
        loaded.eval()
        with torch.no_grad():
            got_u = loaded(eval_u)["loss"].item()
            got_f = loaded(eval_f)["loss"].item()
        assert abs(ref_u - got_u) < 1e-4, f"understanding reload mismatch: {ref_u} vs {got_u}"
        assert abs(ref_f - got_f) < 1e-4, f"forecast reload mismatch: {ref_f} vs {got_f}"
        print(f"(3) from_pretrained reload forward matches: understanding {ref_u:.5f}~{got_u:.5f}, "
              f"forecast {ref_f:.5f}~{got_f:.5f} [OK]")

    print("TRAIN+SAVE PARAMS PASSED")


if __name__ == "__main__":
    main()
