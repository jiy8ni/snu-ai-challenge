#!/usr/bin/env bash
set -euo pipefail

# Source this file before installing packages, training, or inference:
#   source cloud/runpod_env.sh
#
# The goal is to keep heavyweight caches away from the small container/root
# filesystem and under the persistent RunPod workspace instead.

export SNUAI_BASE_DIR="${SNUAI_BASE_DIR:-/workspace/snuai}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SNUAI_BASE_DIR/cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export TMPDIR="${TMPDIR:-$SNUAI_BASE_DIR/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

mkdir -p \
  "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" \
  "$TMPDIR" "$SNUAI_BASE_DIR/outputs" "$SNUAI_BASE_DIR/models" "$SNUAI_BASE_DIR/reports"

echo "RunPod cache base: $SNUAI_BASE_DIR"
