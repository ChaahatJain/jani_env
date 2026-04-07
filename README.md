## Benchmark generation
Use the generate_all_instances.py inside benchmarks_generator/generator to generate instances for one way and two way line. Can be extended to all instances using the commented line in there. 

## Commands I am using for repair:

python pipeline.py --jani_model benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/model.jani --jani_property benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/model.jani --initial_policy artifacts/pipeline/two_way_line_det/two_way_line_80_40/bootstrap/models/final_actor.pth --start_states benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 10 --output_dir artifacts/pipeline/two_way_line_det/two_way_line_80_40/ --device cpu --accumulate_faults --repair_method milp

python pipeline.py --jani_model benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/model.jani --jani_property benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/model.jani --initial_policy artifacts/pipeline/one_way_line_det/one_way_line_80_40/bootstrap/models/final_actor.pth --start_states benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 10 --output_dir artifacts/pipeline/one_way_line_det/one_way_line_80_40/ --device cpu --accumulate_faults --repair_method milp


## TODOS:
TODO Songtuan: For each intermediate checkpoint (from RL), we want to evaluate %Goal, %Fail, Total timesteps (in terms of start states and trajectories), Total Time to reach here. Get this in a usable log file format.

TODO Chaahat: Separate repair into using a fixed set of start states. A disjoint set of start states will be used for evaluation over repaired policies. Setup evaluation over intermediate checkpoints. Implement targeted MILP repair.

TODO Hasanat: Have a look at JiUng's model editing codebase. Try to create a policy updater taking inspiration from there. Another policy updater to implement is: https://arxiv.org/pdf/2012.01872. Please try to keep the updater format consistent with the MILP and SpecRepair one. Also, adapt the supervised learning updater to use a similar input if possible.