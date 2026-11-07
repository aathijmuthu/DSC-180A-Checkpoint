#!/usr/bin/env bash
# Fallback for this Mac when DSMLP is not connected. Run under caffeinate -i.
set -uo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dsc180-local-mpl"
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p logs
failed=0
for task in sweep train100 train1000; do
  echo "Starting local CIFAR $task at $(date)"
  if "$PYTHON" -m checkpoint.cifar "$task" --device mps 2>&1 | tee "logs/local_cifar_${task}.log"; then
    echo "Completed local CIFAR $task at $(date)"
  else
    echo "Local CIFAR $task failed; see logs/local_cifar_${task}.log"
    failed=1
  fi
  "$PYTHON" -m checkpoint.summarize
done
exit "$failed"
