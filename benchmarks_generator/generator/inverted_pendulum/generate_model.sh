#!/bin/sh

prefix=""
length=1
mass=1
half_poll_length=0.5
gravity=9.8607
terminal_unsafe=0
flags="--safety-in-model"

property_type="random_starts_20000"
flags_state=""
flags_state=${flags_state}"--property-inline"
flags_state=${flags_state}" --use-goal-potential"

# List of timestep values to iterate over
for timestep in 0.01 0.05 0.08 0.09 0.1 0.2 0.5
do
  # Format the timestep for filenames (replace dot with underscore if needed)
  model_file=../../benchmarks//inverted_pendulum_${prefix}/inverted_pendulum_${timestep}/model.jani

  echo "Generating model for timestep $timestep -> $out_path"

  python3 generator.py \
    --generation 0 \
    --timestep $timestep \
    --mass $mass \
    --length $length \
    --gravity $gravity \
    --terminal-at-unsafe $terminal_unsafe ${flags}\
    --out $model_file

  
  target_dir=../../benchmarks/inverted_pendulum_${prefix}/inverted_pendulum_${timestep}/
  python3 generator.py --model-file ${model_file} --generation 1 --property-type ${property_type} --splits 0 ${flags_state} --out ${target_dir}

done