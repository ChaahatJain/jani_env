#!/bin/sh

property_type="compact_starts"
#
prefix=""
models="../../benchmarks/rovers/models/${prefix}"
property_prefix="${property_type}"
nn_prefix=""  # no_filtering
#
flags="--splits 0 --property-inline"
# flags=${flags}" --rover-at-lander"

for file in ${models}/*.jani
do
    file_name=$(basename $file .jani)
    # nn_dir="../../benchmarks/rovers/networks/${nn_prefix}/${file_name}/"
    target_dir="../../benchmarks/rovers/additional_properties/safety_verification/${prefix}/${property_prefix}/${nn_prefix}/${file_name}/"
    python3 generator.py --model-file ${file} --generation 1 ${flags} --property-type ${property_type} --out ${target_dir}  # --networks ${nn_dir}
done
