#!/bin/sh

prefix=""
#
description=${1}
#
fail_prob_charge=0
fail_prob_sample=${fail_prob_charge}
fail_prob_image=${fail_prob_charge}
flags=""
flags=${flags}" --enable-battery-overload"
flags=${flags}" --enable-oom-moves"
# flags=${flags}" --use-policy"
# flags=${flags}" --use-multi-loc-policy"

file_name=$(basename $description .json)

python3 generator.py --generation 0 --fail-prob-charge ${fail_prob_charge} --fail-prob-sample ${fail_prob_sample} --fail-prob-image ${fail_prob_image} --description ${description} ${flags} --out ../../benchmarks/rovers/models/${prefix}/${file_name}.jani

