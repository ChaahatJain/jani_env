#!/bin/bash

# Generate commands for HTCondor pipeline
# Bash translation of gen_commands_htcondor_pipeline.py

set -e

# Helper function to normalize paths (remove trailing slashes and double slashes)
normalize_path() {
    echo "$1" | sed 's:/*$::' | sed 's://*:/:g' | sed 's:/\./:/:g' | sed 's:^\./::'
}

# Default values
LOG_DIR="logs"
MODEL_SAVE_DIR="models"
MAX_ITERATIONS=50
NUM_TRACES_PER_ITER=200
TOTAL_TIMESTEPS=1000000
SEED=42
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
        --max_iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --num_traces_per_iter)
            NUM_TRACES_PER_ITER="$2"
            shift 2
            ;;
        --total_timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --root_dir <dir> --condor_prefix <prefix> --output_file <file> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --root_dir          Root directory containing domain subdirectories"
            echo "  --condor_prefix     Prefix path for HTCondor environment"
            echo "  --output_file       File to save generated commands"
            echo ""
            echo "Optional arguments:"
            echo "  --log_dir           Base directory for logs (default: logs)"
            echo "  --model_save_dir    Base directory for saving models (default: models)"
            echo "  --max_iterations    Maximum iterations for the pipeline (default: 50)"
            echo "  --num_traces_per_iter  Number of traces to sample per iteration (default: 200)"
            echo "  --total_timesteps   Total timesteps for training policies (default: 1000000)"
            echo "  --seed              Random seed for reproducibility (default: 42)"
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

    # Create model save directories
    local domain_model_save_dir
    domain_model_save_dir=$(normalize_path "${CONDOR_PREFIX}/${MODEL_SAVE_DIR}/${domain_name}")
    local benchmark_model_save_dir
    benchmark_model_save_dir=$(normalize_path "${domain_model_save_dir}/${benchmark_name}")
    local rl_policy_save_dir
    rl_policy_save_dir=$(normalize_path "${benchmark_model_save_dir}/rl_policies")
    local repair_policy_save_dir
    repair_policy_save_dir=$(normalize_path "${benchmark_model_save_dir}/repair_policies")
    mkdir -p "$rl_policy_save_dir"
    mkdir -p "$repair_policy_save_dir"

    # Build command (prepend CONDOR_PREFIX to model/property paths)
    local jani_model_path
    jani_model_path=$(normalize_path "${CONDOR_PREFIX}/${jani_model}")
    local property_file_path
    property_file_path=$(normalize_path "${CONDOR_PREFIX}/${property_file}")

    local cmd="pipeline_enum.py"
    cmd+=" --jani_model ${jani_model_path}"
    cmd+=" --jani_property ${property_file_path}"
    cmd+=" --start_states ${property_file_path}"
    cmd+=" --disable_wandb"
    cmd+=" --max_iterations ${MAX_ITERATIONS}"
    cmd+=" --num_traces_per_iter ${NUM_TRACES_PER_ITER}"
    cmd+=" --model_save_dir ${rl_policy_save_dir}"
    cmd+=" --repair_save_dir ${repair_policy_save_dir}"
    cmd+=" --log_dir ${benchmark_log_dir}"
    cmd+=" --seed ${SEED}"
    cmd+=" --total_timesteps ${TOTAL_TIMESTEPS}"

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
