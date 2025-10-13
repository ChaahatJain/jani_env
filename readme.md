# Examples

Minimal working example:

1. Train model via:

    python -m rl.train --model_file examples/transport/model.jani --goal_file examples/transport/goal.jani --start_file examples/transport/start.jani --safe_file examples/transport/safe.jani --experiment_name testing_stuff --disable-wandb --total_timesteps 20000

2. Find faults via:

    python -m rl.finetuning --model_file examples/transport/model.jani --goal_file examples/transport/goal.jani --start_file examples/transport/start.jani --safe_file examples/transport/safe.jani --policy_file models/testing_stuff/best_model.zip --n_steps 2048

