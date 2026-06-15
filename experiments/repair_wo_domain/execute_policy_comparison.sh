#!/bin/bash
set -euo pipefail

BASE=/home/atml_team041/condor_tutorial/jani_env

cd ${BASE}

export PYTHONPATH=${BASE}:${BASE}/benchmarks_generator/benchmarks_library:${BASE}/benchmarks_generator/python_library:${PYTHONPATH:-}

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export XDG_CACHE_HOME=/tmp/cache_atml_team041
export HOME=${HOME:-/tmp}

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

python3 -u experiments/repair_wo_domain/compare_policies.py "$@"
