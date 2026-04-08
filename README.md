## Benchmark generation
Use the generate_all_instances.py inside benchmarks_generator/generator to generate instances for one way and two way line. Can be extended to all instances using the commented line in there. 

## Commands I am using for testing stuff on my side repair:

python pipeline.py --jani_model benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/model.jani --jani_property benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/model.jani --initial_policy artifacts/pipeline/two_way_line_det/two_way_line_80_40/bootstrap/models/final_actor.pth --start_states benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 10 --output_dir artifacts/pipeline/two_way_line_det/two_way_line_80_40/ --device cpu --accumulate_faults --repair_method milp

python pipeline.py --jani_model benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/model.jani --jani_property benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/model.jani --initial_policy artifacts/pipeline/one_way_line_det/one_way_line_80_40/bootstrap/models/final_actor.pth --start_states benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 10 --output_dir artifacts/pipeline/one_way_line_det/one_way_line_80_40/ --device cpu --accumulate_faults --repair_method milp

python pipeline.py --jani_model benchmarks_generator/benchmarks/two_way_line_det/two_way_line_20_10/model.jani --jani_property benchmarks_generator/benchmarks/two_way_line_det/two_way_line_20_10/model.jani --initial_policy artifacts/pipeline/two_way_line_det/two_way_line_20_10/bootstrap/models/final_actor.pth --start_states benchmarks_generator/benchmarks/two_way_line_det/two_way_line_20_10/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 5 --output_dir artifacts/pipeline/two_way_line_det/two_way_line_20_10/ --device cpu --accumulate_faults --repair_method milp

## TODOS:
TODO Songtuan: For each intermediate checkpoint (from RL), we want Total timesteps (in terms of start states and trajectories), Total Time to reach here. Get this in a usable log file format.

TODO Chaahat: Implement targeted MILP repair. This actually becomes a MIQP.

TODO Hasanat: Priority number 1 is running experiments. Please get me experiments for one way and two way line by end of week if possible. Remember to rebuild the environment.

Nice to have:
 Have a look at JiUng's model editing codebase. Try to create a policy updater taking inspiration from there. 
 Another policy updater to implement is: https://arxiv.org/pdf/2012.01872. 
 Also, adapt the supervised learning updater to use a similar input if possible.

## Cluster experiments:  
python pipeline.py --jani_model <model> --jani_property <model> --initial_policy <train it if not available> --start_states <random starts file> --objective "" --failure_property "" --max_steps 1100 --traces_per_iteration 1000 --max_iterations <ideally should be until convergence. Or do 1000 max iterations> --output_dir <something reasonable. Remember that we will have several variants running from the intermediate policies we get from RL as well> --device cpu --accumulate_faults --repair_method <milp/spec_repair/supervised>

We should also try running a naive variant where we just continue RL for 1 million timesteps and then evaluate the policy using our evaluate_policy function.