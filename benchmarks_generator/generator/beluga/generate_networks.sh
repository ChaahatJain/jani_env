#!/bin/sh

prefix=${1}
models="../../benchmarks//beluga/models/${prefix}"
nn_prefix=""
#
flags="" # "--applicability-filtering"

for file in ${models}/beluga*.jani
do
    file_name=$(basename $file .jani)
    for layer_size in 16 32 64 128 256
    do
        nn_dir="../../benchmarks//beluga/networks/${prefix}/${file_name}/"
        python3 new_generator.py --model-file ${file} --generation 2 --hidden-layers $layer_size $layer_size ${flags} --out ${nn_dir}/${file_name}
    done

    python3 new_generator.py --model-file ${file} --generation 2 --hidden-layers 256 128 64 32 16 ${flags} --out ${nn_dir}/${file_name}

done
