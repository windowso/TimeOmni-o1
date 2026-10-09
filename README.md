# TimeOmni-o1

Code for the paper *TimeOmni-o1: Bidirectionally Coupling Time Series Foundation Models and LLMs
for Multimodal Reasoning and Forecasting* (Ziyang Zhang, Shenyi Li, Wen Wu, Chao Zhang).
Project page: https://windowso.github.io/TimeOmni-o1-web/

**TimeOmni-o1** couples a pretrained time series foundation model (TSFM; Chronos-2) and an LLM
(Qwen3.5-9B) bidirectionally, so that one model architecture reads and generates both series and
text, reasons in text, and draws on the TSFM's pretrained forecasting ability:

* **series to LLM**: the TSFM encodes the multivariate series; a sliding-window Q-Former compresses
  its patch representations into a bounded soft prompt that the LLM reads next to the text. The LLM
  writes a reasoning trace followed by the answer (**understanding**) or by a forecast instruction
  (**forecasting**).
* **LLM to TSFM** (forecasting): a feedback Q-Former compresses the LLM's last-layer hidden states,
  and gated cross-attention injects them into every TSFM block; the TSFM then outputs the
  21-quantile forecast. The quantile loss back-propagates to the LLM through this pathway.

**MMTR** (Multimodal Time series Reasoning) is the dataset the model is trained on: a multimodal
time series reasoning dataset with annotated reasoning traces that covers the understanding and the
forecasting task across 12 domains. It is not included in this repository yet.

This repository contains the model, training, inference, evaluation (including the trace-quality
judges and the additional benchmarks), the baselines, the architecture ablations, the other choices
of the TSFM and the LLM, and unit tests. The MMTR dataset and the model weights are not included yet.

---

## Repository layout

```
chronos_llm/                 TimeOmni-o1
  models/                    ChronosLLM (PreTrainedModel), sliding-window / feedback Q-Formers, Chronos-2 with gated cross-attention
  data/                      jsonl / parquet datasets, chat template + supervision masking, multichannel collator, dynamic batching sampler
  train.py, trainer.py       HF-Trainer-based training (DeepSpeed ZeRO-2, grouped learning rates, scheduled sampling)
  rl/                        GRPO for the understanding task and its reward
  eval/                      inference (torchrun, multi-node), metrics, LLM-judge trace evaluation, baselines
  scripts/                   entry points: train_*.sh, eval_*.sh (see "Training" / "Inference and evaluation")
  scripts/utils/             per-domain aggregation, reasoning-ablation inputs, judge controls, benchmark converters, baseline launchers
  configs/                   DeepSpeed config and data manifests
  tests/                     CPU unit tests (real Chronos-2 + tiny LLM stand-in); run_all_cpu.sh
  train_headcls.py           TSFM component with per-subtask classification heads (understanding)
docs/                        EVALUATION.md (metrics and protocols)
src/chronos/                 unmodified upstream Chronos-2 code (Apache-2.0)
src/timesfm3/                unmodified upstream TimesFM-3.0 code (Apache-2.0), used when the TSFM is replaced
```

## Setup

```bash
conda create -n timeomni python=3.12 && conda activate timeomni
pip install -r requirements.txt
pip install -e .                      # installs `chronos` (upstream) and `chronos_llm`

# components (any local path works; the scripts read CHRONOS2_PATH / LLM_PATH)
hf download amazon/chronos-2 --local-dir checkpoints/chronos-2
hf download Qwen/Qwen3.5-9B --local-dir checkpoints/Qwen3.5-9B
# only for the other choices of the TSFM and the LLM (TIMESFM3_PATH / LLM_PATH)
hf download google/timesfm-3.0-pytorch --local-dir checkpoints/TimesFM3.0
hf download Qwen/Qwen3.5-4B --local-dir checkpoints/Qwen3.5-4B
```

Notes on the environment: `transformers>=5`; Chronos-2 is loaded through
`chronos_llm.models.cross_attn_chronos.load_chronos2_with_cross_attn` (direct construction +
`load_state_dict`), which also wraps every TSFM block with gated cross-attention.
Qwen3.5's hybrid linear attention needs `flash-linear-attention`; its kernels are specialized per
batch size, which is why the training scripts quantize dynamic batch sizes to a ladder.

## Data

The MMTR dataset is not included in this repository yet. The scripts read the understanding task as
jsonl files listed in `chronos_llm/configs/understanding_train_mmtr.txt` /
`understanding_test_mmtr.txt` and the forecasting task as one parquet file (`FORECAST_PARQUET`).
The CiK and ST-Bench T4 inputs are rebuilt from the official releases with
`build_cik_eval_parquet.py` and `build_stbench_forecast_parquet.py`.

## Model

`chronos_llm/models/chronos_llm_model.py` defines `ChronosLLMConfig` / `ChronosLLM`:

| component | file | notes |
|---|---|---|
| TSFM with feedback | `chronos_llm/models/cross_attn_chronos.py` | Chronos-2 (d=768, 21 quantiles) with a gated cross-attention module `tanh(g) * CrossAttn(h, Z)` added to every TSFM block. The gate starts at 1.0 and the cross-attention output projection at zero, so at initialization the TSFM computes the same function as the pretrained model. |
| sliding-window Q-Former | `chronos_llm/models/qformer.py::SlidingWindowQFormer` | non-overlapping windows over the patch sequence, `k=8` learned queries per window plus 16 global queries; the window count grows log-linearly with the number of patches x channels. Internal width 768 with an output projection to the LLM width. |
| statistics token | `--history_stats_token` | per-window location / scale / range statistics passed to the LLM as one token (instance normalization removes them from the TSFM input). |
| feedback Q-Former | `chronos_llm/models/qformer.py::QFormer` | `M=16` queries over the LLM's last-layer hidden states -> `(M, 768)` cross-states for the TSFM. |
| LLM adaptation | `add_lora()` | LoRA on the LLM only (r=8, alpha=32, dropout 0.1); the TSFM, both Q-Formers and the gates are fully trained. |
| chat template / masking | `chronos_llm/data/chat_utils.py` | the inference prefix is token-identical to the training context. Understanding: `<think>trace</think>` + answer; forecasting: `<think>trace</think>` + forecast instruction. |
| losses | `chronos_llm/models/chronos_llm_model.py::forward` | text cross-entropy + quantile (pinball) loss over the horizon, reweighted inside the event region of interest (ROI). |

Multivariate inputs: the collator folds all channels of all samples into `(sum C, L)` rows with
group ids; the TSFM attends across channels within a sample only; each sample yields one soft
prompt. Long histories are encoded chunk-wise.

## Training

All entry points are in `chronos_llm/scripts/`; every hyperparameter is an environment variable
with the paper's value as default (see the header of each script). Multi-GPU / multi-node via
`torchrun` + DeepSpeed ZeRO-2; four learning-rate groups (TSFM 1e-5, LoRA 1e-4, Q-Formers 1e-4,
gates 1e-3, no weight decay on the gates).

Training has two stages: supervised fine-tuning, then GRPO on the understanding task.

```bash
# Stage 1 - supervised fine-tuning on the understanding task of MMTR (8 GPUs, 10 epochs)
UNDERSTANDING_LIST=chronos_llm/configs/understanding_train_mmtr.txt \
EVAL_U_MAX_NEW_TOKENS=2048 bash chronos_llm/scripts/train_understanding.sh
#   no-reasoning model: NO_REASONING=1 ...      fine-tuned LLM component: LLM_ONLY=1 PROC_PER_NODE=2 U_TOKEN_BUDGET=80000 ...

# Stage 2 - GRPO on the LLM LoRA (exact-match answer reward, G=8, T=1.0, 8 GPUs);
# questions are sampled by domain, with a higher rate for the domains with fewer training samples
python chronos_llm/scripts/utils/make_grpo_domain_pools_v10.py
INIT_CKPT=outputs/understanding_run/<stage1_run>/checkpoint-<N> bash chronos_llm/scripts/train_grpo_understanding.sh

# Forecasting fine-tuning on the forecasting task of MMTR (2 GPUs, 30 epochs)
GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 SS_END_RATIO=0.7 PROC_PER_NODE=2 bash chronos_llm/scripts/train_forecast.sh
#   no-reasoning model: FORECAST_PROMPT_ONLY=1   fine-tuned TSFM component: CHRONOS_ONLY=1   fine-tuned LLM component: LLM_ONLY=1
```

Scheduled sampling (`SS_START_RATIO -> SS_END_RATIO`, 1.0 -> 0.7) decays the teacher-forcing
probability of the text fed back to the TSFM during forecasting training, so that late in training
part of the batches feed the model's own generated text back. Checkpoints are PEFT adapters
(LoRA + TSFM + Q-Formers) plus `config.json`; `ChronosLLM.from_pretrained(path, merge=False)`
reloads them.

The TSFM component on the understanding task is trained with per-subtask classification heads:
`bash chronos_llm/scripts/train_headcls.sh`.

### Architecture ablations

The architecture ablations are switches of the same training scripts:

| pathway | variant | switch |
|---|---|---|
| series to LLM | sliding-window Q-Former -> mean pooling | `HISTORY_COMPRESSOR=pool` |
| series to LLM | no global queries | `SW_GLOBAL_QUERIES=0` |
| series to LLM | soft prompt from TSFM block 6 | `HISTORY_ENCODE_LAYER=6` |
| LLM to TSFM | feedback Q-Former -> mean pooling | `FEEDBACK_COMPRESSOR=pool` |
| LLM to TSFM | feedback from LLM layer 24 | `FEEDBACK_LLM_LAYER=24` |
| LLM to TSFM | inject into the last / first 6 TSFM blocks | `CROSS_ATTN_LAYERS=last6` / `first6` |

### Other choices of the TSFM and the LLM

The same scripts replace one component at a time; the Q-Formers, the gated cross-attention, the
losses and the recipe stay as above.

* **TSFM -> TimesFM-3.0**: `TSFM_BACKBONE=timesfm3` (weights from `TIMESFM3_PATH`).
  `chronos_llm/models/timesfm_backbone.py` wraps TimesFM-3.0 into the Chronos-2 interface (patch
  representations for the sliding-window Q-Former, quantile head for the forecast) and
  `chronos_llm/models/cross_attn_timesfm.py` adds the same gated cross-attention to every TimesFM
  block, again starting from the identity.
* **LLM -> Qwen3.5-4B**: `LLM_PATH=checkpoints/Qwen3.5-4B`.

```bash
# forecasting: the coupled configuration and the TSFM alone, zero-shot and fine-tuned
TSFM_BACKBONE=timesfm3 GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 SS_END_RATIO=0.7 PROC_PER_NODE=2 \
  bash chronos_llm/scripts/train_forecast.sh                                      # TimesFM-3.0 + Qwen3.5-9B
TSFM_BACKBONE=timesfm3 CHRONOS_ONLY=1 PROC_PER_NODE=2 bash chronos_llm/scripts/train_forecast.sh   # TimesFM-3.0 alone
python chronos_llm/eval/baseline_forecast_zeroshot.py --tsfm_backbone timesfm3 \
  --output outputs/eval/zeroshot_timesfm3/preds.npz --output_csv outputs/eval/zeroshot_timesfm3/metrics.csv
LLM_PATH=checkpoints/Qwen3.5-4B GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 SS_END_RATIO=0.7 PROC_PER_NODE=2 \
  bash chronos_llm/scripts/train_forecast.sh                                      # Chronos-2 + Qwen3.5-4B

# understanding: the two stages above with the switch set on stage 1 (stage 2 reads the
# components from the stage-1 checkpoint)
TSFM_BACKBONE=timesfm3 UNDERSTANDING_LIST=chronos_llm/configs/understanding_train_mmtr.txt \
  EVAL_U_MAX_NEW_TOKENS=2048 bash chronos_llm/scripts/train_understanding.sh
```

TimesFM-3.0 outputs nine quantile levels and Chronos-2 outputs 21. Before scoring, the nine
quantiles are interpolated linearly onto the 21 levels (the four outermost levels are
extrapolated): `python chronos_llm/scripts/utils/interp_quantiles.py IN.npz OUT.npz`.

## Inference and evaluation

```bash
# understanding: sharded generation + scoring (native per-subtask protocols, per-domain table)
bash chronos_llm/scripts/eval_understanding.sh <checkpoint_dir>
# forecasting: generation + CRPS over the horizon and the ROI, per-domain table
bash chronos_llm/scripts/eval_forecast.sh <checkpoint_dir>
```

`docs/EVALUATION.md` describes the metrics and protocols. Further evaluation scripts:

| what | script |
|---|---|
| per-domain tables (understanding / forecasting) | `scripts/utils/summarize_agg6_by_domain.py`, `scripts/utils/paper_forecast_by_domain.py` |
| success-rate weighting of the forecasting scores | `scripts/utils/paper_forecast_success_rate.py` |
| quality of the generated reasoning, understanding (LLM judge: consistency / supportiveness) | `chronos_llm/eval/explanation_judge.py`; mismatched reference: `scripts/utils/shuffled_explanation_judge_control.py` |
| quality of the generated reasoning, forecasting (consistency / supportiveness) | `scripts/utils/split_text_judge_eval.py`, `chronos_llm/eval/forecast_visual_faithfulness.py`; mismatched reference: `scripts/utils/shuffled_visual_faithfulness_control.py` |
| counterfactual magnitude edit of the forecast instruction | `chronos_llm/eval/causal_faithfulness.py` |
| reasoning ablation (shuffled events / no-info at test time; no-reasoning prompts) | `scripts/utils/make_prompt_content_probes.py`, `make_shuffled_event_plain_prompt.py`, `build_plain_prompt.py` |

### Additional benchmarks

* **SciTS** (understanding, trained on its training split): `UNDERSTANDING_LIST=chronos_llm/configs/scits_train.txt
  EVAL_LIST=chronos_llm/configs/scits_test.txt EVAL_BASE_DIR=<SciTS release dir> SAMPLE_CHUNKS=8 EVAL_U_MAX_NEW_TOKENS=200
  bash chronos_llm/scripts/train_understanding.sh`; aggregation with `scripts/utils/paper_understanding_domains.py`.
* **CiK** (forecasting, evaluated directly, official RCRPS): `build_cik_eval_parquet.py` -> `eval_cik.sh` -> `eval_cik_rcrps.py`
  (the RCRPS step imports the official benchmark code).
* **ST-Bench T4** (forecasting, trained on its training split, official MAE): `build_stbench_forecast_parquet.py` ->
  `train_forecast.sh` with `FORECAST_BS=32 F_TOKEN_BUDGET=0`.

### Baselines

`chronos_llm/eval/baseline_*.py` (+ launchers in `scripts/utils/`) run the compared models under
the same test protocols: ChatTime-1-7B, Chat-TS-8B, Time-MQA, TimeOmni-VL-15B, TimeOmni-1-4B,
TimeReasoner, DoubleCast, TimeOmni and UniTS (both fine-tuned on MMTR), the TSFM component
zero-shot (`baseline_forecast_zeroshot.py`), and the LLM component with the series serialized as
text (`LLM_ONLY=1` / `LLM_ONLY_ZEROSHOT=1`). Each baseline script expects the third-party code /
weights under `third_party/` and `checkpoints/`; see its docstring.

## Tests

```bash
bash chronos_llm/tests/run_all_cpu.sh            # CPU tests: real Chronos-2 / TimesFM-3.0 + tiny LLM stand-in
bash chronos_llm/tests/run_train_save_check.sh   # train -> save -> reload equivalence
bash chronos_llm/scripts/smoke_test.sh           # GPU: six steps with the real 9B LLM
```

The tests that build training batches read a forecasting parquet and an understanding jsonl from
`FORECAST_PARQUET` / `UNDERSTANDING_JSONL`. The tests cover the gate / zero-output-projection
identity, batched-vs-single generation with left padding, inference-prefix == training-context
alignment, multichannel folding, bucketed encoding, dynamic batching, resume, metrics against
hand-computed values, the TimesFM-3.0 adapter and its cross-attention identity, and the evaluation
pipeline end to end.

## License

Code in `src/chronos/` is the upstream Chronos-2 release (Apache-2.0, see `LICENSE` and
`src/chronos/NOTICE`) and code in `src/timesfm3/` is the upstream TimesFM-3.0 release (Apache-2.0,
see `src/timesfm3/LICENSE`); the rest of the code is released under the same Apache-2.0 license.
No model weights are shipped; the TimesFM-3.0 weights come with their own license on the Hugging Face hub.
