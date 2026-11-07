#!/usr/bin/env bash
# Run inside a DSMLP batch container allocated two GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dsc180-mpl-${USER:-student}"
export PYTHONUNBUFFERED=1
mkdir -p logs artifacts
python -m venv --system-site-packages .venv-dsmlp
export PYTHON="$PWD/.venv-dsmlp/bin/python"
"$PYTHON" -m pip install -r requirements.txt 2>&1 | tee logs/dsmlp_setup.log
"$PYTHON" -m pip freeze > artifacts/environment-dsmlp.txt
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print("CUDA GPUs:", torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
"$PYTHON" -m unittest discover -s tests -v
if ! "$PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 2'; then
  exec bash scripts/run_dsmlp.sh cifar
fi
run() {
  local name="$1"
  shift
  echo "Starting $name at $(date)"
  "$PYTHON" "$@" 2>&1 | tee "logs/${name}.log"
}
set +e
(
  export CUDA_VISIBLE_DEVICES=0
  run dsmlp_sweep -m checkpoint.cifar sweep
) &
sweep_pid=$!
(
  export CUDA_VISIBLE_DEVICES=1
  failed=0
  run dsmlp_train100 -m checkpoint.cifar train100 || failed=1
  run dsmlp_train1000 -m checkpoint.cifar train1000 || failed=1
  exit "$failed"
) &
training_pid=$!
wait "$sweep_pid"
sweep_status=$?
wait "$training_pid"
training_status=$?
"$PYTHON" -m checkpoint.summarize
if command -v pdflatex >/dev/null; then
  (cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex)
fi
if (( sweep_status || training_status )); then
  echo "At least one task failed; inspect logs/dsmlp_*.log. Completed work is saved."
  exit 1
fi
echo "All required CIFAR experiments finished at $(date)"
