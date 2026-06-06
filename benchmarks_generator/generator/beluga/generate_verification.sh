#!/bin/sh

property_type="compact_starts_no_predicates"
# property_type="compact_starts_truck_pos_predicates_first"
#
prefix="swap_unsafe"
models="../../benchmarks/beluga/models/${prefix}"
property_prefix="${property_type}"
nn_prefix=""  # no_filtering
#
flags=""
flags=${flags}" --splits 0 --property-inline"
flags=${flags}" --single-safe-start-value"
# flags=${flags}" --fix-truck-start"
# flags=${flags}" --fix-package-start"
# flags=${flags}" --zero-load-start"

for file in ${models}/beluga*.jani
do
    file_name=$(basename $file .jani)
    # nn_dir="../../benchmarks/sum_transport/networks/${nn_prefix}/${file_name}/"
    target_dir="../../benchmarks/beluga/additional_properties/safety_verification/${prefix}/${property_prefix}/${nn_prefix}/${file_name}/"
    python3 generator.py --model-file ${file} --generation 1 ${flags} --property-type ${property_type} --out ${target_dir}  # --networks ${nn_dir}
done
