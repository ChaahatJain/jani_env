#!/bin/sh

prefix=""
sub_prefix=""
#
num_blocks=${1}
table_limit=${2}
#
flags=""
flags=${flags}" --failing-prob 0.5"
flags=${flags}" --cost-per-item 50"
flags=${flags}" --continuous-cost false"
flags=${flags}" --non-det-cost false"
flags=${flags}" --accumulate-cost -1"  # 1000
flags=${flags}" --cost-terminal true"
flags=${flags}" --terminal-at-unsafe 0"
flags=${flags}" --use-hand-empty-flag"
flags=${flags}" --use-clear-flags"
flags=${flags}" --use-height"
# flags=${flags}" --use-time"
flags=${flags}" --use-table-counter"
# flags=${flags}"" # failing_probability item_cost_bound
# For property.
# For sample-in-model (learning).
# flags=${flags}" --sample-in-model"
# flags=${flags}" --use-goal-potential"
# For verification (default).
flags=${flags}" --safety-in-model"
# flags=${flags}" --hand-empty-at-start"
# flags=${flags}" --ordered-index-start"
flags=${flags}" --zero-cost-start"
#
model_file=../../benchmarks//blocksworld_${prefix}/blocksworld_${num_blocks}_${table_limit}/model.jani
python3 generator.py --generation 0 --num-blocks ${num_blocks} --table-limit ${table_limit} ${flags} --out ${model_file}

#!/bin/sh

property_type="random_starts_100000"
flags=""
flags=${flags}" --property-inline"
# flags=${flags}" --reuse-random-state ./cached_random_states" # _old
flags=${flags}" --use-goal-potential"
# flags=${flags}" --ground-start-potential"
# flags=${flags}" --ground-terminal-potential"

target_dir=../../benchmarks/blocksworld_${prefix}/blocksworld_${num_blocks}_${table_limit}/
python3 generator.py --model-file ${model_file} --generation 1 --property-type ${property_type} --splits 0 ${flags} --out ${target_dir}  # --networks ${nn_dir}