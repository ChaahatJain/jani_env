#!/bin/sh

prefix="swap_unsafe"
#
description=${1}
swaps=${2}
# failing_prob=0.5 # ${2}
#
flags="--safety-in-model"
flags=${flags}

file_name=$(basename $description .json)

# python3 generator.py --generation 0 --description ${description} --terminal-at-unsafe 0 --failing-prob ${failing_prob} ${flags} --out ../../benchmarks//beluga/models/${prefix}/${file_name}.jani

python3 new_generator.py --generation 0 --description ${description} --swaps ${swaps} --terminal-at-unsafe 0 ${flags} --out ../../benchmarks//beluga/models/${prefix}/${file_name}.jani

