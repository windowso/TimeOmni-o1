#!/bin/bash
# Shared preamble for the training / evaluation entry points. Sourced, not executed.
#
#   REPO            repository root (derived from this file's location)
#   CHRONOS2_PATH   Chronos-2 weights (default: checkpoints/chronos-2, e.g. downloaded from
#                   the Hugging Face hub with `hf download amazon/chronos-2 --local-dir checkpoints/chronos-2`)
#   LLM_PATH        Qwen3.5-9B weights (default: checkpoints/Qwen3.5-9B)
#   TSFM_BACKBONE   backbone ablation: chronos2 (default, the released recipe) | timesfm3
#   TIMESFM3_PATH   TimesFM-3.0 weights, used when TSFM_BACKBONE=timesfm3 (default: checkpoints/TimesFM3.0)
#
# Multi-node launches read NODE_COUNT / NODE_RANK / MASTER_ADDR / MASTER_PORT / PROC_PER_NODE
# from the environment (single node, all visible GPUs by default).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Dynamic batching changes tensor shapes from step to step; expandable segments reduce fragmentation.
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Backbone ablation: swapping the TSFM (TSFM_BACKBONE=timesfm3) or the LLM (LLM_PATH=...) only
# changes the weights loaded here -- everything downstream is identical. Both go into the run name
# (BACKBONE_TAG), so runs with different backbones cannot end up in the same output directory.
TSFM_BACKBONE=${TSFM_BACKBONE:-chronos2}
if [ "$TSFM_BACKBONE" = "timesfm3" ]; then
  CHRONOS=${TIMESFM3_PATH:-checkpoints/TimesFM3.0}
else
  CHRONOS=${CHRONOS2_PATH:-checkpoints/chronos-2}
fi
LLM=${LLM_PATH:-checkpoints/Qwen3.5-9B}
BACKBONE_TAG=""
[ "$TSFM_BACKBONE" != "chronos2" ] && BACKBONE_TAG="_${TSFM_BACKBONE}"
_llm_base=$(basename "$LLM")
[ "$_llm_base" != "Qwen3.5-9B" ] && BACKBONE_TAG="${BACKBONE_TAG}_$(echo "$_llm_base" | tr 'A-Z.' 'a-z-')"

NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
PROC_PER_NODE=${PROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$PROC_PER_NODE" -ge 1 ] || PROC_PER_NODE=1

# Read a manifest of jsonl paths (one per line, `#` comments allowed) into the named array.
read_manifest() {  # read_manifest <array_name> <manifest_file>
  local -n _out=$1
  mapfile -t _out < <(grep -vE '^[[:space:]]*(#|$)' "$2")
  [ ${#_out[@]} -gt 0 ] || { echo "ERROR: manifest $2 is missing or empty" >&2; exit 1; }
}

# Archive a copy of the launching script next to the run outputs (rank 0 only).
archive_script() {  # archive_script <output_dir> <script_path>
  if [ "$NODE_RANK" = "0" ]; then
    cp "$2" "$1/$(basename "$2" .sh)_$(date +%Y%m%d-%H%M%S).sh" 2>/dev/null || true
  fi
}
