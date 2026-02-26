Sample commands to start exploring the codebase:

Training a neural network model given a planning problem with the goal and unsafety properties
python -m rl.train --model_file examples/transport/model.jani --goal_file examples/transport/goal.jani --start_file examples/transport/start.jani --safe_file examples/transport/safe.jani --experiment_name testing_stuff --disable-wandb --total_timesteps 20000

Simulating a trained neural policy to obtain faults 
python -m rl.simulation --model_file examples/transport/model.jani --goal_file examples/transport/goal.jani --start_file examples/transport/start.jani --safe_file examples/transport/safe.jani --policy_file models/testing_stuff/best_model.zip --n_steps 2048
# NOTE: The policy must be a zip file due to random stable baselines 3 magic
