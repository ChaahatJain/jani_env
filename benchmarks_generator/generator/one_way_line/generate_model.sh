#!/bin/sh

prefix=${1}
#
description=${2}
dropping_prob=${3} # 0 means no dropping, 0.1 means some dropping
slipping_prob=0
icy_prob=${4} # 0 mea
tank_capacity=0
parking=${5} # Parking
fail_dec_on_ice=${6} # Can decelerate be failed on ice? For safe region, this should only happen when parking is enabled
#
flags="--safety-in-model"

# For property.
# For sample-in-model (learning).
flags=${flags}

file_name=$(basename $description .json)
model_file=../../benchmarks//one_way_line_${prefix}/${file_name}/model.jani
python3 generator.py --generation 0 --description ${description} --dropping-prob ${dropping_prob} --slipping-prob ${slipping_prob} --icy-prob ${icy_prob} --tank-capacity ${tank_capacity} --add-parking ${parking} --fail-dec-on-ice ${fail_dec_on_ice} ${flags} --out ${model_file}

echo "Here ${model_file}"
property_type="random_starts_20000"
#
flags=""
flags=${flags}"--property-inline"
# flags=${flags}" --reuse-random-state ./cached_random_states"
flags=${flags}" --use-goal-potential"
# flags=${flags}" --ground-start-potential"
# flags=${flags}" --ground-terminal-potential"

target_dir="../../benchmarks/one_way_line_${prefix}/${file_name}/"
python3 generator.py --model-file ${model_file} --generation 1 --property-type ${property_type} --splits 0 ${flags} --out ${target_dir}  # --networks ${nn_dir}
