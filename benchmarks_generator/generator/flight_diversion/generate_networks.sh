#!/bin/sh

prefix=""
models="../../benchmarks//flight_diversion/models/${prefix}"
nn_prefix=""
#
flags="--oob-crash false" # "--applicability-filtering"

for file in ${models}/*.jani
do
    file_name=$(basename $file .jani)
    for layer_size in 16 32 64
    do
        nn_dir="../../benchmarks//flight_diversion/networks/${prefix}/${file_name}/"
        python3 generator.py --model-file ${file} --generation 2 --hidden-layers ${layer_size} ${layer_size} ${flags} --out ${nn_dir}/${file_name}
    done
done
