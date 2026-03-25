#!/bin/sh

prefix="swap_unsafe"
#
description=${1}
failing_prob=0.5 # ${2}
#
flags="--safety-in-model"
flags=${flags}

file_name=$(basename $description .json)
model_file=../../benchmarks/beluga_${prefix}/${file_name}/model.jani
python3 generator.py --generation 0 --description ${description} --terminal-at-unsafe 0 --failing-prob ${failing_prob} ${flags} --out ${model_file}


property_type="random_starts_20000"
#
flags=""
flags=${flags}"--property-inline"
# flags=${flags}" --reuse-random-state ./cached_random_states"
flags=${flags}" --use-goal-potential"
# flags=${flags}" --ground-start-potential"
# flags=${flags}" --ground-terminal-potential"

target_dir="../../benchmarks/beluga_${prefix}/${file_name}/"
python3 generator.py --model-file ${model_file} --generation 1 --property-type ${property_type} --splits 0 ${flags} --out ${target_dir}