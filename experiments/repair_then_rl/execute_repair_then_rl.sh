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
export GRB_LICENSE_FILE=/home/jain/gurobi.lic

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

cd /home/jain/jani_env

# ── Parse arguments ──
# We split arguments into two groups separated by "---"
# Phase 1 (repair) args come before "---"
# Phase 2 (RL) args come after "---"
REPAIR_ARGS=()
RL_ARGS=()
PHASE="repair"
RL_ALGO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        ---)
            PHASE="rl"
            shift
            ;;
        --rl_algo)
            RL_ALGO="$2"
            shift 2
            ;;
        *)
            if [[ "$PHASE" == "repair" ]]; then
                REPAIR_ARGS+=("$1")
            else
                RL_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$RL_ALGO" ]]; then
    echo "[ERROR] --rl_algo is required (mask_ppo, safe_dqn, ppo_lag)"
    exit 1
fi

echo ""
echo "========================================="
echo "  PHASE 1: Repair random policy"
echo "========================================="
echo "Repair args: ${REPAIR_ARGS[*]}"
echo ""

python pipeline.py "${REPAIR_ARGS[@]}"
REPAIR_EXIT=$?

if [[ $REPAIR_EXIT -ne 0 ]]; then
    # If exit code is 137 (OOM killed) or any other error, check if we have a checkpoint
    echo "[WARN] Repair exited with code $REPAIR_EXIT (may be OOM). Checking for checkpoints..."
fi

# Find the repaired policy: prefer final_policy.pth, fall back to latest iteration checkpoint
# The repair output_dir is extracted from REPAIR_ARGS
REPAIR_OUTPUT_DIR=""
for i in "${!REPAIR_ARGS[@]}"; do
    if [[ "${REPAIR_ARGS[$i]}" == "--output_dir" ]]; then
        REPAIR_OUTPUT_DIR="${REPAIR_ARGS[$((i+1))]}"
        break
    fi
done

# Also need --repair_method to find the right subdir
REPAIR_METHOD="milp"
for i in "${!REPAIR_ARGS[@]}"; do
    if [[ "${REPAIR_ARGS[$i]}" == "--repair_method" ]]; then
        REPAIR_METHOD="${REPAIR_ARGS[$((i+1))]}"
        break
    fi
done

CHECKPOINTS_DIR="${REPAIR_OUTPUT_DIR}/repair_checkpoints/${REPAIR_METHOD}"
REPAIRED_POLICY="${CHECKPOINTS_DIR}/final_policy.pth"

if [[ ! -f "$REPAIRED_POLICY" ]]; then
    # No final policy; find the latest iteration checkpoint
    REPAIRED_POLICY=$(ls -t "${CHECKPOINTS_DIR}"/policy_iter_*.pth 2>/dev/null | head -1)
fi

if [[ -z "$REPAIRED_POLICY" || ! -f "$REPAIRED_POLICY" ]]; then
    echo "[ERROR] No repaired policy checkpoint found in ${CHECKPOINTS_DIR}"
    exit 1
fi

# Reuse states nearest failures/cycles as RL restart states. Explicit RL
# arguments take precedence; otherwise train on a 50/50 mix of repaired-problem
# states and the model's original initial-state distribution.
FAULTY_STATES_FILE="${REPAIR_OUTPUT_DIR}/faulty_states.json"
HAS_FAULTY_STATES_PATH=false
HAS_FAULTY_RESET_PROB=false
for arg in "${RL_ARGS[@]}"; do
    if [[ "$arg" == "--faulty_states_path" ]]; then
        HAS_FAULTY_STATES_PATH=true
    elif [[ "$arg" == "--faulty_state_reset_prob" ]]; then
        HAS_FAULTY_RESET_PROB=true
    fi
done

if [[ "$HAS_FAULTY_STATES_PATH" == false && -f "$FAULTY_STATES_FILE" ]]; then
    RL_ARGS+=("--faulty_states_path" "$FAULTY_STATES_FILE")
    HAS_FAULTY_STATES_PATH=true
fi
if [[ "$HAS_FAULTY_STATES_PATH" == true && "$HAS_FAULTY_RESET_PROB" == false ]]; then
    RL_ARGS+=("--faulty_state_reset_prob" "${FAULTY_STATE_RESET_PROB:-0.5}")
fi

echo ""
echo "========================================="
echo "  PHASE 2: RL training on repaired policy"
echo "========================================="
echo "RL algorithm:    $RL_ALGO"
echo "Repaired policy: $REPAIRED_POLICY"
if [[ "$HAS_FAULTY_STATES_PATH" == true ]]; then
    echo "Faulty restarts: enabled"
else
    echo "Faulty restarts: disabled (no failed/cyclic states collected)"
fi
echo "RL args:         ${RL_ARGS[*]}"
echo ""

case "$RL_ALGO" in
    mask_ppo)
        python -m mask_ppo.train "${RL_ARGS[@]}" --load_policy_path "$REPAIRED_POLICY"
        ;;
    safe_dqn)
        python -m safe_dqn.safe_dqn "${RL_ARGS[@]}" --load_policy_path "$REPAIRED_POLICY"
        ;;
    ppo_lag)
        python -m ppo_lag.ppo_lag "${RL_ARGS[@]}" --load_policy_path "$REPAIRED_POLICY"
        ;;
    *)
        echo "[ERROR] Unknown RL algorithm: $RL_ALGO"
        echo "Supported: mask_ppo, safe_dqn, ppo_lag"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "  Done: repair-then-RL pipeline complete"
echo "========================================="
