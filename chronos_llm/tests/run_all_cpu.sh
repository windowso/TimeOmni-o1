#!/bin/bash
# Run the CPU test suite sequentially (one process per test so memory is released between tests).
# Every test uses the real Chronos-2 weights plus a tiny randomly initialised Qwen2 stand-in for the
# 9B LLM; set CHRONOS2_PATH / LLM_PATH if the defaults (checkpoints/...) do not apply.
#   bash chronos_llm/tests/run_all_cpu.sh                      # whole suite
#   bash chronos_llm/tests/run_all_cpu.sh test_eval_metrics    # selected tests (no .py)
# pipefail is essential: without it the pipeline's exit code is tail's (always 0).
set -e -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/src:${PYTHONPATH:-}"

FILTER="Loading weights|UserWarning|warnings.warn|fast path|fla-org|Dao-AILab|Consider using tensor.detach|it/s\]|s/it\]"
DEFAULT_TESTS="test_sliding_window_ladder test_qformer_hidden_dim test_cross_attn_identity test_composite_wiring \
test_timesfm_cross_attn test_timesfm_backbone test_timesfm_running_stats_patch test_timesfm_wiring \
test_pretrained_peft test_trainer_integration test_train_resume test_single_branch \
test_multichannel_qformer test_multichannel_collator test_multichannel_encode \
test_multichannel_forecast test_multichannel_datasets test_generate_left_padding \
test_inference_prefix_alignment test_eval_metrics test_eval_understanding_metrics test_eval_pipeline test_dist_eval_merge \
test_mid_train_eval test_data_hygiene test_optimizer_groups test_length_bucketing test_encode_row_bucketing \
test_encode_checkpointing test_window_sampling test_chunked_ce test_dynamic_batch test_epoch_repeats \
test_short_window_enhance test_forecast_ablations test_rl_reward test_grpo_rollout test_grpo_understanding \
test_chronos_random_init test_text_alignment_eval test_structured_conclusion \
test_forecast_visual_faithfulness test_freeze_llm_continue test_freeze_ts test_understanding_reasoning \
test_eval_opentslm_answer \
test_eval_limit_sampling test_summarize_agg6_eval test_summarize_by_domain \
test_explanation_judge test_shuffled_explanation_control test_headcls test_llm_only \
test_zero_out_proj_floor test_fb_layer_grad_ckpt test_arch_ablations"
TESTS="${*:-$DEFAULT_TESTS}"
for t in $TESTS; do
  echo "===== $t ====="
  python chronos_llm/tests/$t.py 2>&1 | { grep -vE "$FILTER" || true; } | tail -8
done
echo "===== ALL CPU TESTS DONE ====="
