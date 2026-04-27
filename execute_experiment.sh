#!/usr/bin/env bash
echo "===== CONTAINER VERSION ====="
cat /IMAGE_VERSION || echo "No version file found!"
echo "============================="
echo "=== identity/debug ==="
echo "uid=$(id -u) gid=$(id -g)"
echo "whoami=$(whoami 2>/dev/null || true)"
echo "USER=${USER-} LOGNAME=${LOGNAME-} HOME=${HOME-}"
echo "_CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR-}"
echo "======================"

# Ensure getpass.getuser() won't query /etc/passwd
export USER="${USER:-condor}"
export LOGNAME="${LOGNAME:-$USER}"

# Ensure HOME exists and is writable
export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"

# Put caches somewhere writable
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torch/inductor}"
export TORCHDYNAMO_CACHE_DIR="${TORCHDYNAMO_CACHE_DIR:-$XDG_CACHE_HOME/torch/dynamo}"

export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCHDYNAMO_CACHE_DIR" "$TMPDIR"

export LC_ALL=C
export GRB_LICENSE_FILE=/nethome/julee/2026-policy-repair/gurobi.lic

cd /nethome/julee/2026-policy-repair/jani_env
python learning.py "$@"
