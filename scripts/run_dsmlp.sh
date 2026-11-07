#!/usr/bin/env bash
# Run INSIDE a GPU-enabled DSMLP session, from the repository root.
set -uo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dsc180-mpl-${USER:-student}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHON="${PYTHON:-python}"
mkdir -p logs artifacts
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU: launch this inside your DSMLP GPU session"; print(torch.cuda.get_device_name(0))' || exit 1
"$PYTHON" -m pip freeze > artifacts/environment.txt
failed=0
run() {
  local name="$1"
  shift
  echo "Starting $name at $(date)"
  if "$PYTHON" "$@" 2>&1 | tee "logs/$name.log"; then
    echo "Completed $name at $(date)"
  else
    echo "FAILED: $name. See logs/$name.log; continuing other deliverables."
    failed=1
  fi
}
# Exact required sample counts. Training limits can be changed explicitly if necessary.
case "${1:-all}" in
  cifar)
    run cifar_figures -m checkpoint.cifar figures
    run cifar_sweep -m checkpoint.cifar sweep
    run cifar_train100 -m checkpoint.cifar train100 --updates "${CIFAR_UPDATES:-20000}"
    run cifar_train1000 -m checkpoint.cifar train1000 --updates "${CIFAR_UPDATES:-20000}"
    ;;
  mnist)
    run mnist_lab -m checkpoint.mnist_lab
    run mnist -m checkpoint.mnist --updates "${MNIST_UPDATES:-10000}"
    ;;
  all)
    run cifar_figures -m checkpoint.cifar figures
    run cifar_sweep -m checkpoint.cifar sweep
    run mnist_lab -m checkpoint.mnist_lab
    run mnist -m checkpoint.mnist --updates "${MNIST_UPDATES:-10000}"
    run cifar_train100 -m checkpoint.cifar train100 --updates "${CIFAR_UPDATES:-20000}"
    run cifar_train1000 -m checkpoint.cifar train1000 --updates "${CIFAR_UPDATES:-20000}"
    ;;
  *) echo 'Usage: bash scripts/run_dsmlp.sh [all|cifar|mnist]'; exit 2 ;;
esac
"$PYTHON" -m checkpoint.summarize
exit "$failed"
