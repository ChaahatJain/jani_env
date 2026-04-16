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
export GRB_LICENSE_FILE=/home/neuronet_team119/gurobi.lic

echo "=== Gurobi License Check ==="
if [[ ! -f "$GRB_LICENSE_FILE" ]]; then
    echo "[ERROR] License file not found: $GRB_LICENSE_FILE"
    exit 1
fi
echo "[OK]    License file found: $GRB_LICENSE_FILE"
python - <<'EOF'
import sys
try:
    import gurobipy as gp
    with gp.Env() as env:
        print(f"[OK]    Gurobi loaded successfully (version {gp.gurobi.version()})")
except Exception as e:
    print(f"[ERROR] Gurobi license validation failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF
echo "============================"

cd /home/neuronet_team119/jani_env

# Parse --algo flag to dispatch to the right training script
ALGO="pipeline"
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo)
            ALGO="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$ALGO" in
    pipeline)
        python pipeline.py "${REMAINING_ARGS[@]}"
        ;;
    safe_dqn)
        python -m safe_dqn.safe_dqn "${REMAINING_ARGS[@]}"
        ;;
    ppo_lag)
        python -m ppo_lag.ppo_lag "${REMAINING_ARGS[@]}"
        ;;
    *)
        echo "[ERROR] Unknown algorithm: $ALGO"
        echo "Supported: pipeline, safe_dqn, ppo_lag"
        exit 1
        ;;
esac
