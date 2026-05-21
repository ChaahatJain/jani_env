#!/bin/bash

# Arguments: $1 = num_locations, $2 = num_packages, $3 = num_icy_locations, $4 = max_speed
LOCATIONS=$1
PACKAGES=$2
ICY=$3
SPEED=$4

BASE_DIR=/home/atml_team041/condor_tutorial/jani_env/benchmarks_generator
ONE_WAY_LINE_DIR=${BASE_DIR}/generator/one_way_line

export PYTHONPATH=${BASE_DIR}/benchmarks_library:${BASE_DIR}/python_library:${PYTHONPATH}

cd ${ONE_WAY_LINE_DIR}

python3 instance_generator.py -l ${LOCATIONS} -p ${PACKAGES} -i ${ICY} -s ${SPEED}
