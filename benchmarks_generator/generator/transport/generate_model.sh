#!/bin/sh

prefix=""
#
description=${1}
dropping_prob=0 # ${2}
tank_capacity=40
#
flags="--safety-in-model"
flags=${flags}

# For property.
# For sample-in-model (learning).
flags=${flags}""

file_name=$(basename $description .json)
model_file=../../benchmarks//transport_${prefix}/${file_name}/model.jani
python3 generator.py --generation 0 --description ${description} --dropping-prob ${dropping_prob} --tank-capacity ${tank_capacity} ${flags} --out ${model_file}

property_type="random_starts_100000"
#
prefix=""
#
flags=""
flags=${flags}"--property-inline"
# flags=${flags}" --reuse-random-state ./cached_random_states"
flags=${flags}" --use-goal-potential"
# flags=${flags}" --ground-start-potential"
# flags=${flags}" --ground-terminal-potential"


target_dir="../../benchmarks//transport_${prefix}/${file_name}/"
python3 generator.py --model-file ${model_file} --generation 1 --property-type ${property_type} --splits 0 ${flags} --out ${target_dir}  # --networks ${nn_dir}

