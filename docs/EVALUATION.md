# Evaluation protocols

## Understanding

Inference: `chronos_llm/eval/infer_understanding.py` (via `scripts/eval_understanding.sh`) writes one
jsonl per test file with `generated_text`, `ground_truth`, `task`, `scene`. Decoding is greedy with
`max_new_tokens=2048`; the inference prefix opens a `<think>` segment except for sources whose test
protocol is direct answering (ST-Bench).

Each subtask is scored by its source's **native protocol** (`scripts/utils/summarize_agg6_eval.py`):

| source | metric |
|---|---|
| OpenTSLM ECG-QA / HAR / Sleep | the official OpenTSLM evaluation (`eval/eval_opentslm_answer.py`: answer extraction after `Answer:`, label canonicalization, accuracy / F1) |
| ST-Bench, HiTSR, TSRBench | exact-match accuracy of the extracted answer (letter boundaries are enforced for single-letter multiple-choice answers) |
| TelecomTS | exact match against the reference answer |
| RATs40K | support-weighted F1 over the anomaly types |

Subtask scores aggregate into the 12 domains by sample count
(`scripts/utils/summarize_agg6_by_domain.py`; `domain_of` assigns every sample to one domain), and
**Avg.** is the mean of the 12 domain scores.

Baselines are scored by the same scripts on their generated text
(`scripts/utils/score_agg6_baseline.sh`, `split_agg6_for_baselines.py`,
`merge_agg6_baseline_parts.py`). Models with a single-channel interface read the first channel of a
multivariate sample.

## Forecasting

Inference: `chronos_llm/eval/infer_forecast.py` (via `scripts/eval_forecast.sh`) on the test split
of the parquet. The model writes its own reasoning trace and forecast instruction, and the TSFM
produces the forecast conditioned on them. Outputs: an `npz` with the 21 quantile forecasts, the
target, the validity and ROI masks, and a jsonl with the generated text.

Metric (`chronos_llm/eval/metrics.py`, per sample, then averaged):

* **CRPS** = normalized weighted quantile loss over the 21 quantile levels,
  `sum_q 2|(y - q_hat)(1{y <= q_hat} - alpha)| / (K * sum |y|)`; samples with an all-zero target
  are excluded.
* computed over the **full horizon** and over the **event ROI**.
* A model that outputs one point trajectory has that trajectory copied to all 21 levels, which
  reduces its CRPS to a weighted absolute error. Forecasts on another quantile grid are interpolated
  onto the 21 levels first (`scripts/utils/interp_quantiles.py`).

Reporting: per-domain scores pool the samples of a domain (`scripts/utils/paper_forecast_by_domain.py`,
ten subsets -> five domains: solar, load, traffic, finance, climate), and **Avg.** is the
equal-weight mean of the five domains. A model that decodes its forecast as text can fail to produce
a parseable array; its score in domain *d* is weighted by the success rate `sr_d`, the share of the
samples of *d* with a finite forecast over the full horizon
(`scripts/utils/paper_forecast_success_rate.py`).

## Quality of the generated reasoning

The LLM judge is GPT-5.4, called through an OpenAI-compatible endpoint (`OPENAI_API_KEY` /
`--api_base_url`). Each trace is scored on two metrics, once against its own sample (matched) and
once against a partner sample from a derangement of the judged set (mismatched, the chance
reference).

* **Understanding** (`eval/explanation_judge.py`): *consistency* compares the generated reasoning
  with the annotated trace of the sample; *supportiveness* checks whether the reasoning supports the
  model's own answer (1-5). Samples are stratified by domain, and supportiveness is also split by
  whether the model's answer is correct. Mismatched reference:
  `scripts/utils/shuffled_explanation_judge_control.py`.
* **Forecasting**: *consistency* compares the generated trace and forecast instruction with the
  annotated ones (`scripts/utils/split_text_judge_eval.py`, prompts in `eval/text_alignment_eval.py`);
  *supportiveness* checks whether the plotted forecast matches what the generated text says
  (`eval/forecast_visual_faithfulness.py`). Mismatched reference:
  `scripts/utils/shuffled_visual_faithfulness_control.py`.
* **Counterfactual magnitude edit** (forecasting, `eval/causal_faithfulness.py`): five statements of
  relative magnitude are substituted in turn into the annotated forecast instruction of each test
  sample, the model is run forward on each variant, and the script reports the per-sample Spearman
  correlation between the stated level and the resulting forecast, and the agreement in direction.

## Additional benchmarks

* **SciTS**: trained on the official training split and scored with the suite's protocol (F1 for
  understanding tasks, accuracy for multiple choice, domain = mean over its tasks;
  `scripts/utils/paper_understanding_domains.py`, `paper_scits_baselines.py`).
* **CiK**: evaluated directly with the model trained on MMTR; the official RCRPS implementation is
  imported from the benchmark code and fed the 21 quantiles (`scripts/utils/eval_cik_rcrps.py`).
* **ST-Bench T4**: trained on the ST-Bench T4 training rows and scored with the official MAE on its
  test rows (`scripts/utils/build_stbench_forecast_parquet.py`).
