#!/bin/sh

property_type="compact_starts_no_predicates"
# property_type="compact_starts_truck_pos_predicates_first"
#
prefix=""
models="../../benchmarks//flight_diversion/models/${prefix}"
property_prefix="${property_type}"
nn_prefix=""  # no_filtering
#
flags=""

for file in ${models}/*.jani
do
    echo "File: $file"
    file_name=$(basename $file .jani)
    # nn_dir="../../benchmarks/transport/networks/${nn_prefix}/${file_name}/"
    target_dir="../../benchmarks//flight_diversion/additional_properties/safety_verification/${prefix}/${property_prefix}/${nn_prefix}/${file_name}/"
    echo "Target Dir: $target_dir"
    python3 generator.py --model-file ${file} --generation 1 --property-type ${property_type} ${flags} --out ${target_dir}  # --networks ${nn_dir}
done
