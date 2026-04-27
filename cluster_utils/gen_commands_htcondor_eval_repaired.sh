#!/bin/bash

# Generate commands for HTCondor evaluation of repaired policies
# Bash translation of gen_commands_htcondor_eval_repaired.py

set -e

# Helper function to normalize paths (remove trailing slashes and double slashes)
normalize_path() {
    echo "$1" | sed 's:/*$::' | sed 's://*:/:g' | sed 's:/\./:/:g' | sed 's:^\./::'
}

# Default values
LOG_DIR="logs"
MODEL_SAVE_DIR="models"
SEED=42
MAX_STEPS=256
MAX_TRAINING_IDX=80000
GOAL_REWARD=1.0
FAILURE_REWARD=-1.0
UNSAFE_REWARD=-0.01
DEVICE="cpu"
USE_ORACLE=false
DISABLE_ORACLE_CACHE=false
NO_MEMORY_REDUCED_MODE=false
OUTPUT_FILE=""
ROOT_DIR=""
CONDOR_PREFIX=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --root_dir)
            ROOT_DIR="$2"
            shift 2
            ;;
        --condor_prefix)
            CONDOR_PREFIX="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --model_save_dir)
            MODEL_SAVE_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max_training_idx)
            MAX_TRAINING_IDX="$2"
            shift 2
            ;;
        --goal_reward)
            GOAL_REWARD="$2"
            shift 2
            ;;
        --failure_reward)
            FAILURE_REWARD="$2"
            shift 2
            ;;
        --unsafe_reward)
            UNSAFE_REWARD="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --use_oracle)
            USE_ORACLE=true
            shift
            ;;
        --disable_oracle_cache)
            DISABLE_ORACLE_CACHE=true
            shift
            ;;
        --no_memory_reduced_mode)
            NO_MEMORY_REDUCED_MODE=true
            shift
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --root_dir <dir> --condor_prefix <prefix> --output_file <file> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --root_dir              Root directory containing domain subdirectories"
            echo "  --condor_prefix         Prefix path for HTCondor environment"
            echo "  --output_file           File to save generated commands"
            echo ""
            echo "Optional arguments:"
            echo "  --log_dir               Base directory for logs (default: logs)"
            echo "  --model_save_dir        Base directory for saved models (default: models)"
            echo "  --seed                  Random seed for reproducibility (default: 42)"
            echo "  --max_steps             Maximum steps per episode (default: 256)"
            echo "  --max_training_idx      Maximum training index for evaluation (default: 80000)"
            echo "  --goal_reward           Reward for reaching the goal (default: 1.0)"
            echo "  --failure_reward        Reward for failure (default: -1.0)"
            echo "  --unsafe_reward         Reward for unsafe states (default: -0.01)"
            echo "  --device                Device to run evaluation on (default: cpu)"
            echo "  --use_oracle            Use oracle for evaluation"
            echo "  --disable_oracle_cache  Disable oracle cache"
            echo "  --no_memory_reduced_mode  Disable memory reduced mode"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$ROOT_DIR" ]]; then
    echo "Error: --root_dir is required"
    exit 1
fi

if [[ -z "$CONDOR_PREFIX" ]]; then
    echo "Error: --condor_prefix is required"
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: --output_file is required"
    exit 1
fi

# Normalize input paths (strip trailing slashes and double slashes)
ROOT_DIR=$(normalize_path "$ROOT_DIR")
CONDOR_PREFIX=$(normalize_path "$CONDOR_PREFIX")
LOG_DIR=$(normalize_path "$LOG_DIR")
MODEL_SAVE_DIR=$(normalize_path "$MODEL_SAVE_DIR")

# Function to generate command for a single benchmark
gen_command_for_benchmark() {
    local benchmark_dir="$1"
    local domain_name="$2"

    local jani_model="${benchmark_dir}/model.jani"
    local property_file="${benchmark_dir}/pa_model_random_starts_100000.jani"

    # Check if model file exists
    if [[ ! -f "$jani_model" ]]; then
        echo "Error: Model file $jani_model does not exist in benchmark directory $benchmark_dir" >&2
        return 1
    fi

    # Check if property file exists
    if [[ ! -f "$property_file" ]]; then
        echo "Error: Property file $property_file does not exist in benchmark directory $benchmark_dir" >&2
        return 1
    fi

    local benchmark_name
    benchmark_name=$(basename "$benchmark_dir")

    # Create log directories
    local domain_log_dir
    domain_log_dir=$(normalize_path "${CONDOR_PREFIX}/${LOG_DIR}/${domain_name}")
    local benchmark_log_dir
    benchmark_log_dir=$(normalize_path "${domain_log_dir}/${benchmark_name}")
    mkdir -p "$benchmark_log_dir"

    # Get repair policy save directory (read from existing model structure)
    local domain_model_save_dir
    domain_model_save_dir=$(normalize_path "${CONDOR_PREFIX}/${MODEL_SAVE_DIR}/${domain_name}")
    local benchmark_model_save_dir
    benchmark_model_save_dir=$(normalize_path "${domain_model_save_dir}/${benchmark_name}")
    local repair_policy_save_dir
    repair_policy_save_dir=$(normalize_path "${benchmark_model_save_dir}/repair_policies")

    # Build command (prepend CONDOR_PREFIX to model/property paths)
    local jani_model_path
    jani_model_path=$(normalize_path "${CONDOR_PREFIX}/${jani_model}")
    local property_file_path
    property_file_path=$(normalize_path "${CONDOR_PREFIX}/${property_file}")

    local cmd="eval_repaired_policy.py"
    cmd+=" --jani_model ${jani_model_path}"
    cmd+=" --jani_property ${property_file_path}"
    cmd+=" --start_states ${property_file_path}"
    cmd+=" --repair_save_dir ${repair_policy_save_dir}"
    cmd+=" --log_dir ${benchmark_log_dir}"
    cmd+=" --seed ${SEED}"
    cmd+=" --max_steps ${MAX_STEPS}"
    cmd+=" --max_training_idx ${MAX_TRAINING_IDX}"
    cmd+=" --goal_reward ${GOAL_REWARD}"
    cmd+=" --failure_reward ${FAILURE_REWARD}"
    cmd+=" --unsafe_reward ${UNSAFE_REWARD}"
    cmd+=" --device ${DEVICE}"

    # Add optional boolean flags
    if [[ "$USE_ORACLE" == true ]]; then
        cmd+=" --use_oracle"
    fi
    if [[ "$DISABLE_ORACLE_CACHE" == true ]]; then
        cmd+=" --disable_oracle_cache"
    fi
    if [[ "$NO_MEMORY_REDUCED_MODE" == true ]]; then
        cmd+=" --no_memory_reduced_mode"
    fi

    echo "$cmd"
}

# Main logic: generate commands for all benchmarks
generate_all_commands() {
    local commands=()

    # Iterate over domain directories
    for domain_dir in "$ROOT_DIR"/*/; do
        if [[ ! -d "$domain_dir" ]]; then
            continue
        fi

        local domain_name
        domain_name=$(basename "$domain_dir")

        # Iterate over benchmark directories within each domain
        for benchmark_dir in "$domain_dir"/*/; do
            if [[ ! -d "$benchmark_dir" ]]; then
                continue
            fi

            # Remove trailing slash
            benchmark_dir="${benchmark_dir%/}"

            local cmd
            cmd=$(gen_command_for_benchmark "$benchmark_dir" "$domain_name")
            if [[ $? -eq 0 && -n "$cmd" ]]; then
                commands+=("$cmd")
            fi
        done
    done

    # Write commands to output file
    local num_commands=${#commands[@]}
    for ((i=0; i<num_commands; i++)); do
        if [[ $i -eq $((num_commands - 1)) ]]; then
            # Last line without newline (matching Python behavior)
            printf "%s" "${commands[$i]}" >> "$OUTPUT_FILE"
        else
            echo "${commands[$i]}" >> "$OUTPUT_FILE"
        fi
    done

    echo "Generated $num_commands commands to $OUTPUT_FILE"
}

# Clear output file if it exists
> "$OUTPUT_FILE"

# Run main function
generate_all_commands
