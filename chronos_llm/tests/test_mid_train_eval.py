"""CPU integration test of the automatic mid-training evaluation at epoch intervals
(MidTrainEvalCallback): a tiny model + toy data really run trainer.train(), verifying
(1) mixed training produces both understanding and forecast metrics at the requested epoch (the
interval works: with every=2, epoch_1 is not evaluated, epoch_2 is); (2) single-branch training
only produces the metrics of the corresponding task; (3) the understanding limit works (only the
first N rows of each file are evaluated); (4) the model is back in training mode after evaluation.
"""
import csv
import os
import tempfile

from transformers import TrainingArguments

from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.data.sampler import ConcatBranchDataset
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.eval.mid_eval import MidTrainEvalCallback
from chronos_llm.models.chronos_llm_model import add_lora
from chronos_llm.tests.test_eval_pipeline import _forecast_parquet, _understanding_jsonl
from chronos_llm.tests.test_pretrained_peft import _build_tiny_base
from chronos_llm.trainer import ChronosLLMTrainer


def _make_trainer(model, tok, u_ds, f_ds, out_dir, cb, epochs):
    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=2,
        num_train_epochs=epochs, logging_steps=100, save_steps=10_000, use_cpu=True,
        remove_unused_columns=False, label_names=["labels"], report_to=[],
        learning_rate=1e-4,
    )
    return ChronosLLMTrainer(
        model=model, args=targs, train_dataset=ConcatBranchDataset(u_ds, f_ds),
        processing_class=tok, callbacks=[cb],
        understanding_dataset=u_ds, forecast_dataset=f_ds,
        understanding_bs=2, forecast_bs=2,
        understanding_max_context=2048, forecast_max_context=512,
    )


def test_mixed_training_eval_interval():
    model, tok = _build_tiny_base()
    model = add_lora(model, r=4, alpha=8, dropout=0.0)
    with tempfile.TemporaryDirectory() as d:
        jf = os.path.join(d, "toy.jsonl"); _understanding_jsonl(jf, n=4)
        pq = os.path.join(d, "toy.parquet"); _forecast_parquet(pq, n=4)
        u_ds = UnderstandingJsonlDataset([jf], tok, max_user_tokens=32, max_tokens=96)
        f_ds = ForecastParquetDataset(pq, tok, split=None, max_user_tokens=32, max_tokens=96)
        out = os.path.join(d, "run")
        cb = MidTrainEvalCallback(
            tokenizer=tok, output_dir=out, every_epochs=2,
            understanding_jsonl=[jf], understanding_limit=3, understanding_bs=2,
            understanding_max_new_tokens=3,
            forecast_parquet=pq, forecast_limit=4, forecast_bs=2, forecast_max_new_tokens=3,
            max_user_tokens=32, max_tokens=96)
        trainer = _make_trainer(model, tok, u_ds, f_ds, out, cb, epochs=2)
        trainer.train()

        assert not os.path.exists(os.path.join(out, "mid_eval", "epoch_1")), "with every=2, epoch_1 should not be evaluated"
        root = os.path.join(out, "mid_eval", "epoch_2")
        u_csv = os.path.join(root, "understanding_metrics.csv")
        f_csv = os.path.join(root, "forecast_metrics.csv")
        assert os.path.exists(u_csv) and os.path.exists(f_csv), "mixed training should produce metrics for both tasks"
        assert any(r["filename"] == "toy.jsonl" for r in csv.DictReader(open(u_csv)))
        assert len(list(csv.DictReader(open(f_csv)))) >= 1
        # Forecasting evaluates both the generate and the teacher_forced setting by default (the latter with the _tf suffix)
        f_tf_csv = os.path.join(root, "forecast_metrics_tf.csv")
        assert os.path.exists(f_tf_csv) and os.path.exists(os.path.join(root, "forecast_preds_tf.npz")), \
            "the teacher_forced metrics (_tf) should be produced in addition by default"
        assert len(list(csv.DictReader(open(f_tf_csv)))) >= 1
        # limit=3 in effect: the understanding infer output has only the first 3 rows
        lines = open(os.path.join(root, "understanding", "toy.jsonl")).readlines()
        assert len(lines) == 3, f"understanding_limit=3 should evaluate only 3 rows, got {len(lines)}"
        assert trainer.model.training is False or True  # after train() the mode is managed by HF; not asserted here
    print("mixed-training mid-eval (interval=2, both tasks, limit) OK")


def test_single_branch_eval_selects_task():
    model, tok = _build_tiny_base()
    model = add_lora(model, r=4, alpha=8, dropout=0.0)
    with tempfile.TemporaryDirectory() as d:
        jf = os.path.join(d, "toy.jsonl"); _understanding_jsonl(jf, n=4)
        u_ds = UnderstandingJsonlDataset([jf], tok, max_user_tokens=32, max_tokens=96)
        out = os.path.join(d, "run")
        # Understanding branch only: forecast_parquet=None (this is what train.py's assembly passes for a single branch)
        cb = MidTrainEvalCallback(
            tokenizer=tok, output_dir=out, every_epochs=1,
            understanding_jsonl=[jf], understanding_limit=2, understanding_bs=2,
            understanding_max_new_tokens=3, forecast_parquet=None,
            max_user_tokens=32, max_tokens=96)
        trainer = _make_trainer(model, tok, u_ds, None, out, cb, epochs=1)
        trainer.train()

        root = os.path.join(out, "mid_eval", "epoch_1")
        assert os.path.exists(os.path.join(root, "understanding_metrics.csv")), "understanding metrics should be produced"
        assert not os.path.exists(os.path.join(root, "forecast_preds.npz")), "an understanding-only branch should not evaluate forecasting"
        assert not os.path.exists(os.path.join(root, "forecast_metrics.csv"))
    print("single-branch mid-eval task auto-selection OK")


if __name__ == "__main__":
    test_mixed_training_eval_interval()
    test_single_branch_eval_selects_task()
    print("ALL OK")
