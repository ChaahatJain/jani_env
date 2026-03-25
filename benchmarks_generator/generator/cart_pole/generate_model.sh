#!/bin/sh

prefix=""
#
timestep=0.003 # ${2}
cart_mass=1
half_poll_length=0.5
gravity=9.8607
terminal_unsafe=0
#
flags=""
flags=${flags}

model_file=../../benchmarks/cart_pole_/cart_pole_${timestep}/model.jani
# For property.
# For sample-in-model (learning).
python3 generator.py --generation 0 --timestep ${timestep} --cart_mass ${cart_mass} --poll_mass 0.1 --half_poll_length ${half_poll_length} --gravity $gravity --terminal-at-unsafe $terminal_unsafe --out 

#!/bin/sh

property_type="random_starts_20000"
flags=""
flags=${flags}"--property-inline"
# flags=${flags}" --reuse-random-state ./cached_random_states"
flags=${flags}" --use-goal-potential"
# flags=${flags}" --ground-start-potential"
# flags=${flags}" --ground-terminal-potential"

target_dir=../../benchmarks/cart_pole_/cart_pole_${timestep}/
python3 generator.py --model-file ${model_file} --generation 1 --property-type ${property_type} --splits 0 ${flags} --out ${target_dir}
