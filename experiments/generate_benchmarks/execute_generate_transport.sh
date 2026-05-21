#!/bin/bash

# Arguments: $1 = num_locations, $2 = num_packages
LOCATIONS=$1
PACKAGES=$2

BASE_DIR=/home/atml_team041/condor_tutorial/jani_env/benchmarks_generator
TRANSPORT_DIR=${BASE_DIR}/generator/transport

export PYTHONPATH=${BASE_DIR}/benchmarks_library:${BASE_DIR}/python_library:${PYTHONPATH}

cd ${TRANSPORT_DIR}

python3 instance_generator.py -l ${LOCATIONS} -p ${PACKAGES}
