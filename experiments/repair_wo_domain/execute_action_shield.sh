#!/usr/bin/env bash
set -euo pipefail

export USER=atml_team041
export LOGNAME=atml_team041
export HOME=/home/atml_team041

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${USER}
export TRITON_CACHE_DIR=/tmp/triton_${USER}
export XDG_CACHE_HOME=/tmp/cache_${USER}

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

BASE=/home/atml_team041/condor_tutorial/jani_env
cd "$BASE"
export PYTHONPATH="${BASE}:${PYTHONPATH:-}"

mode="${1:?Expected mode: train, evaluate, or merge-evaluation}"
shift

case "$mode" in
  train)
    python3 -u experiments/repair_wo_domain/train_action_fault_classifiers.py "$@"
    ;;
  evaluate)
    python3 -u experiments/repair_wo_domain/evaluate_action_fault_shield.py evaluate "$@"
    ;;
  merge-evaluation)
    python3 -u experiments/repair_wo_domain/evaluate_action_fault_shield.py merge "$@"
    ;;
  *)
    echo "Unknown action-shield mode: $mode" >&2
    exit 2
    ;;
esac
