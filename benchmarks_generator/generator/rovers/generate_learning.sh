#!/bin/sh

property_type="random_starts_1000"
#
prefix=""
models="../../benchmarks/rovers/models/${prefix}"
property_prefix="${property_type}"
nn_prefix=""  # no_filtering
#
flags=""
flags=${flags}" --property-inline"
flags=${flags}" --reuse-random-states ./cached_random_states"
# flags=${flags}" --use-goal-potential"
# flags=${flags}" --ground-start-potential"
# flags=${flags}" --ground-terminal-potential"

for file in ${models}/*.jani
do
    file_name=$(basename $file .jani)
    # nn_dir="../../benchmarks/rovers/networks/${nn_prefix}/${file_name}/"
    target_dir="../../benchmarks/rovers/additional_properties/learning/${prefix}/${property_prefix}/${nn_prefix}/${file_name}/"
    python3 generator.py --model-file ${file} --generation 1 --property-type ${property_type} --splits 0 ${flags} --out ${target_dir}  # --networks ${nn_dir}
done
