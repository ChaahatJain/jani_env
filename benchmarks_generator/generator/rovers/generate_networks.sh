#!/bin/sh

prefix=""
models="../../benchmarks/rovers/models/${prefix}"
nn_prefix="ite_policy_based"
#
flags="" # "--applicability-filtering"

for file in ${models}/*.jani
do
    file_name=$(basename $file .jani)
    for layer_size in 16 32 64
    do
        nn_dir="../../benchmarks/rovers/networks/${nn_prefix}/${file_name}/"
        python3 generator.py --model-file ${file} --generation 2 --hidden-layers $layer_size $layer_size ${flags} --out ${nn_dir}/${file_name}
    done
done
