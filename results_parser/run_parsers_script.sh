#!/bin/bash

set -euo pipefail

BASE_DIR="../cluster/pipeline/local_repair_logs/home/jain/jani_env/artifacts/pipeline"
PLOTS_DIR="./plots"

mkdir -p "$PLOTS_DIR"

for dir in "$BASE_DIR"/*/*/repair_logs; do
    experiment=$(basename "$(dirname "$dir")")       # e.g. one_way_line_70_35
    category=$(basename "$(dirname "$(dirname "$dir")")")  # e.g. one_way_line_det
    name="${category}__${experiment}"
    echo "Processing: $name"
    python3 repair_pipeline.py "$dir" --title "Repair pipeline comparison - ${name}" --output "${PLOTS_DIR}/${name}.png"
done

echo "Done. Plots saved to $PLOTS_DIR"