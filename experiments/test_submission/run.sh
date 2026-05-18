#!/usr/bin/env bash
set -e

# --- environment fixes for HTCondor docker jobs ---
export USER="${USER:-condor}"
export LOGNAME="${LOGNAME:-$USER}"
export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torch/inductor}"
export TORCHDYNAMO_CACHE_DIR="${TORCHDYNAMO_CACHE_DIR:-$XDG_CACHE_HOME/torch/dynamo}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCHDYNAMO_CACHE_DIR" "$TMPDIR"
export LC_ALL=C
export GRB_LICENSE_FILE=/home/jain/gurobi.lic

echo "Job started: $(date)"
echo "Running on: $(hostname)"

cd /home/jain/jani_env
python learning.py \
    --algo mask_ppo \
    --jani_model /home/jain/jani_env/benchmarks_generator/benchmarks/two_way_line_non_det_with_park/two_way_line_15_10/model.jani \
    --total_timesteps 1000 \
    --n_eval_episodes 10 \
    --eval_freq 500 \
    --n_steps 500 \
    --disable_wandb \
    --device cpu \
    --seed 42

echo "Job finished: $(date)"
