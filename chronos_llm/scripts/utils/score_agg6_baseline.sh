#!/bin/bash
# One-shot scoring of external baselines on the MMTR understanding pool ("agg6"), pure CPU:
#   merge parts -> summarize_agg6_eval.py (native protocol of each subtask)
#               -> summarize_agg6_by_domain.py (aggregation into the 12 domains)
# Usage: bash score_agg6_baseline.sh chattime [chatts_2503 ...]   # no args = all baselines
set -e
REPO=.
cd "$REPO"
UTILS=chronos_llm/scripts/utils

BLS=${@:-chattime chatts_2503 timemqa timeomni1 timeomnivl}
for BL in $BLS; do
  D=outputs/eval/baseline_$BL/understanding_agg6
  [ -d "$D" ] || { echo "[skip] $BL: $D does not exist"; continue; }
  echo "===== $BL"
  python $UTILS/merge_agg6_baseline_parts.py "$D"
  python $UTILS/summarize_agg6_eval.py "$D" --out "$D/agg6_native_protocol.md" || true
  python $UTILS/summarize_agg6_by_domain.py "$BL=$D" --out "$D/agg6_by_domain.md" || true
done
