## Benchmark generation
Use the generate_all_instances.py inside benchmarks_generator/generator to generate instances for one way and two way line. Can be extended to all instances using the commented line in there. 

## Commands I am using for testing stuff on my side repair:

python pipeline.py --jani_model benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/model.jani --jani_property benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/model.jani --initial_policy artifacts/pipeline/two_way_line_det/two_way_line_80_40/rl_training/models/final_actor.pth --start_states benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 10 --output_dir artifacts/pipeline/two_way_line_det/two_way_line_80_40/ --device cpu --accumulate_faults --repair_method milp

python pipeline.py --jani_model benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/model.jani --jani_property benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/model.jani --initial_policy artifacts/pipeline/one_way_line_det/one_way_line_80_40/rl_training/models/final_actor.pth --start_states benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 10 --output_dir artifacts/pipeline/one_way_line_det/one_way_line_80_40/ --device cpu --accumulate_faults --repair_method milp

python pipeline.py --jani_model benchmarks_generator/benchmarks/two_way_line_det/two_way_line_20_10/model.jani --jani_property benchmarks_generator/benchmarks/two_way_line_det/two_way_line_20_10/model.jani --initial_policy artifacts/pipeline/two_way_line_det/two_way_line_20_10/rl_training/models/actor_iter_0.pth --start_states benchmarks_generator/benchmarks/two_way_line_det/two_way_line_20_10/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 5 --output_dir artifacts/pipeline/two_way_line_det/two_way_line_20_10/ --device cpu --accumulate_faults --repair_method milp --bootstrap_timesteps 100000

## Experiments we want to run:
1. RL as a warm start and then running Repair:: Already experiments running by Hasanat
2. Repair as a warm start and then running RL: TODO by Hasanat
3. RL and Repair starting from a random policy (given the same number of start states): TODO Songtuan
4. Repair and RL interleaved with each other: TODO Chaahat

TODO Chaahat: RL with intermediate steps of MILP repair. 

TODO Hasanat:  Start running experiments (one way line and two way line at minimum). Would be great to get control benchmarks + transport as well.

Nice to have:
 Another policy updater to implement is: https://arxiv.org/pdf/2012.01872. 
 Adapt the supervised learning updater to use a similar input if possible.
 Implement targeted MILP repair. This actually becomes a MIQP.
 Polish plots once results are finalized.
 Other control benchmarks domains? 

## Cluster experiments:  
python pipeline.py --jani_model <model> --jani_property <model> --initial_policy <train it if not available> --start_states <random starts file> --objective "" --failure_property "" --max_steps 1100 --traces_per_iteration 1000 --max_iterations <ideally should be until convergence. Or do 1000 max iterations> --output_dir <something reasonable. Remember that we will have several variants running from the intermediate policies we get from RL as well> --device cpu --accumulate_faults --repair_method <milp/spec_repair/supervised>

We should also try running a naive variant where we just continue RL for 1 million timesteps and then evaluate the policy using our evaluate_policy function.



## Commands I am using on my side for running RL

python -m learning --algo mask_ppo --jani_model benchmarks_generator/benchmarks/one_way_line_det/one_way_line_15_10/model.jani --jani_property benchmarks_generator/benchmarks/one_way_line_det/one_way_line_15_10/model.jani --start_states benchmarks_generator/benchmarks/one_way_line_det/one_way_line_15_10/pa_model_random_starts_100000.jani --goal_reward 1.0 --failure_reward -1.0 --max_steps 256 --total_timesteps 30000 --n_eval_episodes 50 --eval_freq 10 --experiment_name one_way_line_15_10_det --model_save_dir /jani_env/models/ppo/one_way_line_15_10 --disable_wandb --verbose 1 --device cpu --seed 50 --perf_file logs/ppo.csv --save_all_checkpoints

Remember to change the save directories and perf_files for these methods!!

The algo we support are: mask_ppo, ppo_lag and safe_dqn. 
Note that safe_dqn requires far larger parameters than ppo. I show two examples below.

PPO Lagrangian hyperparams: --total_timesteps 100_000 

SafeDQN hyperparams: --total_timesteps 1_000_000 (even this is not sufficient to get meaningful enough results)!