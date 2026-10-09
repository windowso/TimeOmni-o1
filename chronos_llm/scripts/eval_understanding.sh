#!/bin/bash
# Understanding evaluation, two stages: (1) torchrun inference — every rank generates for its
# shard of samples and rank 0 merges the per-file jsonl outputs; (2) CPU scoring.
#
#   bash chronos_llm/scripts/eval_understanding.sh <MODEL_PATH> [TEST_JSONL ...]
#
# Without explicit jsonl files the MMTR test manifest is used. Environment overrides:
#   BASE_DIR (directory that relative `ori_path` values are relative to; default `.` = repository root,
#            which is what the shipped MMTR files use; SciTS files need data/raw/scits/Release_v1), OUT_DIR, BATCH_SIZE, MAX_NEW_TOKENS, NUM_WORKERS,
#   SAMPLE_CHUNKS / OVERVIEW_CHUNK (must match training: 15/1 for MMTR, 8/1 for SciTS),
#   NO_REASONING=1        the checkpoint was trained without reasoning (prefix closes <think> immediately)
#   FORCE_REASONING=1     always open a <think> segment even for sources whose test set has no trace
#   TF_REASONING=1        diagnostic: teacher-force the ground-truth trace, generate only the answer
#   LLM_ONLY_ZEROSHOT=1   MODEL_PATH is a bare LLM directory (series serialized as text, zero-shot)
#   ANSWER_PREFIX="Answer:"  diagnostic: forced answer-prefix decoding — the prefix continues past the
#                         closed think with the answer opening, so the model cannot write a rationale.
#                         Must be combined with NO_REASONING=1; write to a separate OUT_DIR.
# Scoring: eval_understanding.py writes a per-file metrics CSV; the per-subtask native
# protocols and discipline aggregation are produced by scripts/utils/summarize_agg6_eval.py and
# summarize_agg6_by_domain.py (see docs/EVALUATION.md).
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_PATH=${1:?usage: eval_understanding.sh <MODEL_PATH> [TEST_JSONL ...]}
TEST_FILES=("${@:2}")
if [ ${#TEST_FILES[@]} -eq 0 ]; then
  read_manifest TEST_FILES "${EVAL_LIST:-chronos_llm/configs/understanding_test_mmtr.txt}"
fi
BASE_DIR=${BASE_DIR:-.}
OUT_DIR=${OUT_DIR:-outputs/eval/understanding}
INFER_DIR="$OUT_DIR/infer"
mkdir -p "$INFER_DIR"
# The scorer scans the whole INFER_DIR; remove stale outputs from earlier runs first (rank 0 only).
if [ "$NODE_RANK" = "0" ]; then rm -f "$INFER_DIR"/*.jsonl "$INFER_DIR"/*.jsonl.rank* 2>/dev/null || true; fi

FLAGS=()
[ "${NO_REASONING:-0}" = "1" ]      && FLAGS+=(--no_reasoning)
[ "${TF_REASONING:-0}" = "1" ]      && FLAGS+=(--teacher_forced_reasoning)
[ "${FORCE_REASONING:-0}" = "1" ]   && FLAGS+=(--force_reasoning)
[ "${CF_REASONING:-0}" = "1" ]      && FLAGS+=(--counterfactual_reasoning)
[ "${LLM_ONLY_ZEROSHOT:-0}" = "1" ] && FLAGS+=(--llm_only_zeroshot)
[ -n "${ANSWER_PREFIX:-}" ]         && FLAGS+=(--answer_prefix "$ANSWER_PREFIX")

torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK --nproc_per_node=$PROC_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))} \
  chronos_llm/eval/infer_understanding.py \
  --model_path "$MODEL_PATH" \
  --data_file_list "${TEST_FILES[@]}" \
  --output_folder "$INFER_DIR" \
  --base_dir "$BASE_DIR" \
  --batch_size ${BATCH_SIZE:-128} --max_new_tokens ${MAX_NEW_TOKENS:-2048} \
  --num_workers ${NUM_WORKERS:-16} --limit ${LIMIT:-0} \
  --encode_sample_chunks ${SAMPLE_CHUNKS:-15} --encode_overview_chunk ${OVERVIEW_CHUNK:-1} \
  "${FLAGS[@]}"

if [ "$NODE_RANK" = "0" ]; then
  python chronos_llm/eval/eval_understanding.py "$INFER_DIR" --output "$OUT_DIR/understanding_metrics.csv"
  python chronos_llm/scripts/utils/summarize_agg6_eval.py "$INFER_DIR" > "$OUT_DIR/native_protocol_summary.md" || true
  python chronos_llm/scripts/utils/summarize_agg6_by_domain.py "run=$INFER_DIR" > "$OUT_DIR/by_discipline.md" || true
fi
