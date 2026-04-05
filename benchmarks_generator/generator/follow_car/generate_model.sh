#!/bin/sh

prefix=""
#
velocity=28.0
terminal_unsafe=0
#
flags="--safety-in-model"
flags=${flags}

property_type="random_starts_100000"
flags_state=""
flags_state=${flags_state}"--property-inline"
flags_state=${flags_state}" --use-goal-potential"

# For property.
# For sample-in-model (learning).
for timestep in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  # Format the timestep for filenames (replace dot with underscore if needed)
  model_file="../../benchmarks/follow_car_${prefix}/follow_car_${timestep}/model.jani"

  echo "Generating model for timestep $timestep -> $model_file"

  python3 generator.py --generation 0 --timestep $timestep --velocity $velocity --terminal-at-unsafe $terminal_unsafe ${flags} \
    --out $model_file

  
  target_dir=../../benchmarks/follow_car_${prefix}/follow_car_${timestep}/
  python3 generator.py --model-file ${model_file} --generation 1 --property-type ${property_type} --splits 0 ${flags_state} --out ${target_dir}
done