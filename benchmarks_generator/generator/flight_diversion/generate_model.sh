#!/bin/sh

prefix=""
#
description=${1}
oob_crash=0
#
flags=""
flags=${flags}

file_name=$(basename $description .json)

# For property.
# For sample-in-model (learning).
echo --out "../../benchmarks//flight_diversion/models/${prefix}/${file_name}.jani"
python3 generator.py --generation 0 --description ${description} ${flags} --out "../../benchmarks//flight_diversion/models/${prefix}/${file_name}.jani"
