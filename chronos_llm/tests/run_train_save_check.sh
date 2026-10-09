#!/bin/bash
# Train -> save -> reload value-by-value check (trainable parameters change, the frozen base LLM
# does not, and from_pretrained reproduces the trained weights). CPU, tiny LLM stand-in.
set -e -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/src:${PYTHONPATH:-}"
FILTER="Loading weights|UserWarning|warnings.warn|fast path|fla-org|Dao-AILab|Consider using tensor.detach|it/s\]|s/it\]"
for t in test_pretrained_peft test_train_save_params; do
  echo "===== $t ====="
  python chronos_llm/tests/$t.py 2>&1 | { grep -vE "$FILTER" || true; } | tail -30
done
echo "===== TRAIN+SAVE CHECK DONE ====="
