#!/usr/bin/env bash
set -euo pipefail

export USER=atml_team041
export LOGNAME=atml_team041
export HOME=/home/atml_team041

export GRB_LICENSE_FILE=/home/atml_team041/condor_tutorial/gurobi.lic

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${USER}
export TRITON_CACHE_DIR=/tmp/triton_${USER}
export XDG_CACHE_HOME=/tmp/cache_${USER}

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

cd /home/atml_team041/condor_tutorial/jani_env

python3 pipeline.py "$@"