#!/bin/bash

# Script to generate training commands for IQL policies
# Translated from gen_commands_htcondor_iql.py

set -e

# Helper function to normalize paths (remove trailing slashes and double slashes)
normalize_path() {
    echo "$1" | sed 's:/*$::' | sed 's://*:/:g'
}

# Default values
LOG_DIRECTORY="./logs"
TOTAL_TIMESTEPS=512000
BATCH_SIZE=64
STEPS_PER_EPOCH=1000
EXPECTILE=0.7
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --condor_dir_prefix)
            CONDOR_DIR_PREFIX="$2"
            shift 2
            ;;
        --log_directory)
            LOG_DIRECTORY="$2"
            shift 2
            ;;
        --total_timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --steps_per_epoch)
            STEPS_PER_EPOCH="$2"
            shift 2
            ;;
        --expectile)
            EXPECTILE="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$ROOT" ]]; then
    echo "Error: --root is required"
    exit 1
fi

if [[ -z "$CONDOR_DIR_PREFIX" ]]; then
    echo "Error: --condor_dir_prefix is required"
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: --output_file is required"
    exit 1
fi

# Normalize input paths (strip trailing slashes)
ROOT=$(normalize_path "$ROOT")
CONDOR_DIR_PREFIX=$(normalize_path "$CONDOR_DIR_PREFIX")
LOG_DIRECTORY=$(normalize_path "$LOG_DIRECTORY")

# Clear output file
> "$OUTPUT_FILE"

# Function to process a single model file within a variant directory
process_model_file() {
    local model_file=$(normalize_path "$1")
    local variant_dir=$(normalize_path "$2")
    local domain_dir=$(normalize_path "$3")

    local variant_name=$(basename "$variant_dir")
    local domain_name=$(basename "$domain_dir")
    local jani_name=$(basename "$model_file" .jani)

    local model_save_dir
    local property_dir

    if [[ "$variant_name" == "models" ]]; then
        property_dir="${domain_dir}/additional_properties"
        model_save_dir="${CONDOR_DIR_PREFIX}/${domain_dir}/iql_policies/${jani_name}"
    else
        property_dir="${domain_dir}/additional_properties/${variant_name}"
        model_save_dir="${CONDOR_DIR_PREFIX}/${domain_dir}/iql_policies/${variant_name}/${jani_name}"
    fi

    # Locate training property file
    local training_property_dir="${property_dir}/random_starts_20000/${jani_name}"
    if [[ ! -d "$training_property_dir" ]]; then
        echo "Error: Training property directory $training_property_dir does not exist." >&2
        exit 1
    fi

    local training_property_files=("$training_property_dir"/*)
    if [[ ${#training_property_files[@]} -ne 1 ]]; then
        echo "Error: Expected one property file in $training_property_dir, found ${#training_property_files[@]}" >&2
        exit 1
    fi
    local training_property_file="${training_property_files[0]}"

    # Build the command line
    local cmd="-m offline_rl.iql"
    cmd+=" --model_path ${CONDOR_DIR_PREFIX}/${model_file}"
    cmd+=" --property_path ${CONDOR_DIR_PREFIX}/${training_property_file}"
    cmd+=" --start_states ${CONDOR_DIR_PREFIX}/${training_property_file}"
    cmd+=" --goal_reward 0.0"
    cmd+=" --failure_reward -1.0"
    cmd+=" --unsafe_reward -0.01"
    cmd+=" --batch_size ${BATCH_SIZE}"
    cmd+=" --total_timesteps ${TOTAL_TIMESTEPS}"
    cmd+=" --steps_per_epoch ${STEPS_PER_EPOCH}"
    cmd+=" --expectile ${EXPECTILE}"
    cmd+=" --model_save_dir ${model_save_dir}"
    cmd+=" --seed ${SEED}"

    echo "$cmd"
}

# Function to process a variant directory (contains .jani model files)
process_variant_dir() {
    local variant_dir=$(normalize_path "$1")
    local domain_dir=$(normalize_path "$2")

    for model_file in "$variant_dir"/*.jani; do
        if [[ -f "$model_file" ]]; then
            process_model_file "$model_file" "$variant_dir" "$domain_dir"
        fi
    done
}

# Function to process a domain directory
process_domain_dir() {
    local domain_dir=$(normalize_path "$1")
    local model_dir="${domain_dir}/models"

    if [[ ! -d "$model_dir" ]]; then
        return
    fi

    # Check if model_dir contains subdirectories (variants) or files directly
    local has_subdirs=false
    local has_files=false

    for entry in "$model_dir"/*; do
        if [[ -d "$entry" ]]; then
            has_subdirs=true
        elif [[ -f "$entry" ]]; then
            has_files=true
        fi
    done

    if [[ "$has_subdirs" == true && "$has_files" == true ]]; then
        echo "Error: Model directory $model_dir contains a mix of files and directories." >&2
        exit 1
    fi

    if [[ "$has_subdirs" == true ]]; then
        # Process each variant subdirectory
        for variant_dir in "$model_dir"/*/; do
            if [[ -d "$variant_dir" ]]; then
                process_variant_dir "$variant_dir" "$domain_dir"
            fi
        done
    elif [[ "$has_files" == true ]]; then
        # Process model_dir directly as the variant directory
        process_variant_dir "$model_dir" "$domain_dir"
    fi
}

# Main: iterate through all domain directories
FIRST_LINE=true
for domain_dir in "$ROOT"/*/; do
    if [[ -d "$domain_dir" ]]; then
        while IFS= read -r line; do
            if [[ "$FIRST_LINE" == true ]]; then
                echo -n "$line" >> "$OUTPUT_FILE"
                FIRST_LINE=false
            else
                echo "" >> "$OUTPUT_FILE"
                echo -n "$line" >> "$OUTPUT_FILE"
            fi
        done < <(process_domain_dir "$domain_dir")
    fi
done

echo "Generated commands written to $OUTPUT_FILE"
