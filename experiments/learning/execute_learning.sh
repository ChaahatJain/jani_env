#!/bin/bash

BASE=/home/atml_team041/condor_tutorial/jani_env

cd ${BASE}

export PYTHONPATH=${BASE}/benchmarks_generator/benchmarks_library:${BASE}/benchmarks_generator/python_library:${PYTHONPATH}

# PyTorch/inductor tries to resolve the username from UID, which fails in Docker
# when the UID has no /etc/passwd entry. Set cache dirs explicitly to avoid this.
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export HOME=${HOME:-/tmp}

python3 learning.py "$@"
