#!/bin/sh

prefix=""
#
gravity=9.8067
bounce=0.9
push_lower=7
push_upper=9
complex=0
terminal_unsafe=0
#
flags="--safety-in-model"
flags=${flags}

property_type="random_starts_20000"
flags_state=""
flags_state=${flags_state}"--property-inline"
flags_state=${flags_state}" --use-goal-potential"

# For property.
# For sample-in-model (learning).
# List of timestep values to iterate over
for timestep in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7 0.9
do
  # Format the timestep for filenames (replace dot with underscore if needed)
  model_file="../../benchmarks/bouncing_ball_${prefix}/bouncing_ball_${timestep}/model.jani"

  python3 generator.py --generation 0 --timestep ${timestep} --gravity ${gravity} --bounce ${bounce} --push_lower ${push_lower} --push_upper ${push_upper} --complex ${complex} --terminal-at-unsafe ${terminal_unsafe} ${flags} \
    --out $model_file

  target_dir=../../benchmarks//bouncing_ball_${prefix}/bouncing_ball_${timestep}/
  python3 generator.py --model-file ${model_file} --generation 1 --property-type ${property_type} --splits 0 ${flags_state} --out ${target_dir}

done