import argparse

from pathlib import Path


def get_configs_for_benchmark(variant_dir: str, domain_dir: str, shared_args: dict[str, any]) -> list[dict[str, any]]:
    """Get file arguments for benchmark environment."""
    variant_dir = Path(variant_dir)
    domain_dir = Path(domain_dir)
    variant_name = variant_dir.name
    domain_name = domain_dir.name

    condor_dir_prefix = Path(shared_args.get("condor_dir_prefix", ""))

    list_configs = []
    for model_file in variant_dir.iterdir():
        # Ignore directories storing policy files
        if not model_file.is_file():
            continue
        jani_name = model_file.name.replace(".jani", "")
        if variant_name == "models":
            property_dir = domain_dir / "additional_properties"
            policy_dir = domain_dir / "ppo_policies"
            log_dir = condor_dir_prefix / Path(shared_args.get("log_directory", "./logs")) / domain_name / jani_name
        else:
            property_dir = domain_dir / "additional_properties" / variant_name
            policy_dir = domain_dir / "ppo_policies" / variant_name
            log_dir = condor_dir_prefix / Path(shared_args.get("log_directory", "./logs")) / domain_name / variant_name / jani_name

        model_save_dir = log_dir / "policies"
        policy_path = policy_dir / shared_args.get("policy_filename")
        # if not policy_path.exists():
        #     policy_path = policy_dir / "best_model.pth"
        assert policy_path.exists(), f"Policy file {policy_path} does not exist."
        policy_path = condor_dir_prefix / policy_path

        # Locate the training property file
        training_property_dir = property_dir / "random_starts_20000" / jani_name
        assert training_property_dir.exists(), f"Training property directory {training_property_dir} does not exist."
        all_training_properties_files = list(training_property_dir.iterdir())
        assert len(all_training_properties_files) == 1, f"Expected one property file in {training_property_dir}, found {len(all_training_properties_files)}"
        training_property_file = all_training_properties_files[0]

        # Locate the evaluation property file
        eval_property_dir = property_dir / "random_starts_5000" / jani_name
        assert eval_property_dir.exists(), f"Evaluation property directory {eval_property_dir} does not exist."
        all_eval_properties_files = list(eval_property_dir.iterdir())
        assert len(all_eval_properties_files) == 1, f"Expected one property file in {eval_property_dir}, found {len(all_eval_properties_files)}"
        eval_property_file = all_eval_properties_files[0]

        # Whether to use multi-processors
        use_multiprocessors = shared_args.get("use_multiprocessors", False)

        # set seed
        seed = 42
        # Set up arguments
        args = {
            "jani_model": condor_dir_prefix / model_file,
            "jani_property": condor_dir_prefix / training_property_file,
            "start_states": condor_dir_prefix / training_property_file,
            "objective": "",
            "failure_property": "",
            "eval_start_states": condor_dir_prefix / eval_property_file,
            "policy_path": policy_path,
            "goal_reward": 1.0,
            "failure_reward": -1.0,
            "unsafe_reward": -0.01,
            "num_init_states": 20000,
            "num_iterations": shared_args.get("num_iterations", 20),
            "num_samples": shared_args.get("num_samples", 256),
            "batch_size": shared_args.get("batch_size", 64),
            "num_workers": shared_args.get("num_workers", 4) if use_multiprocessors else False,
            "max_steps": shared_args.get("max_steps", 256),
            "disable_oracle_cache": shared_args.get("disable_oracle_cache", False),
            "reduced_memory_mode": shared_args.get("reduced_memory_mode", False),
            "use_strict_rule": shared_args.get("use_strict_rule", False),
            "use_multiprocessors": use_multiprocessors,
            "empty_buffer": shared_args.get("empty_buffer", False),
            "steps_per_iteration": shared_args.get("steps_per_iteration", 5),
            "wandb_project": f"{jani_name}_clean",
            "experiment_name": f"dagger_{variant_name}" if variant_name != "models" else "dagger",
            "log_directory": log_dir,
            "model_save_dir": model_save_dir,
            "disable_wandb": shared_args.get("disable_wandb", False),
            "seed": shared_args.get("seed", 42),
        }
        # Create file arguments
        file_args = {
            "jani_model": args.get("jani_model", str(model_file)),
            "jani_property": args.get("jani_property", str(training_property_file)),
            "start_states": args.get("start_states", str(training_property_file)),
            "objective": args.get("objective", ""),
            "failure_property": args.get("failure_property", ""),
            "seed": args.get("seed", 42),
            "goal_reward": args.get("goal_reward", 1.0),
            "failure_reward": args.get("failure_reward", -1.0),
            "unsafe_reward": args.get("unsafe_reward", -0.01),
            "use_oracle": True,
            "disable_oracle_cache": args.get("disable_oracle_cache", False),
            "reduced_memory_mode": args.get("reduced_memory_mode", False),
            "max_steps": args.get("max_steps", 256)
        }
        list_configs.append((args, file_args))
    return list_configs


def get_configs_for_domain(domain_dir: str, shared_args: dict[str, any]) -> list[dict[str, any]]:
    """Get file arguments for all models in a domain directory."""
    domain_dir = Path(domain_dir)
    model_dir = domain_dir / "models"
    list_configs = []
    if all(e.is_dir() for e in model_dir.iterdir()):
        for variant_dir in model_dir.iterdir():
            variant_configs = get_configs_for_benchmark(str(variant_dir), str(domain_dir), shared_args)
            list_configs.extend(variant_configs)
    elif all(e.is_file() for e in model_dir.iterdir()):
        variant_configs = get_configs_for_benchmark(str(model_dir), str(domain_dir), shared_args)
        list_configs.extend(variant_configs)
    else:
        raise ValueError(f"Model directory {model_dir} contains a mix of files and directories.")
    return list_configs


def get_all_configs(root_dir: str, shared_args: dict[str, any]) -> list[dict[str, any]]:
    """Get file arguments for all domains in the root directory."""
    root_dir = Path(root_dir)
    list_configs = []
    for domain_dir in root_dir.iterdir():
        if domain_dir.is_dir():
            domain_configs = get_configs_for_domain(str(domain_dir), shared_args)
            list_configs.extend(domain_configs)
    return list_configs


def main():
    parser = argparse.ArgumentParser(description="Benchmark Trainer with Ray")
    parser.add_argument(
        "--root", 
        type=str, required=True, help="Path to the benchmark root directory")
    parser.add_argument(
        "--log_directory", 
        type=str, default="./logs", help="Directory to save logs")
    parser.add_argument(
        "--condor_dir_prefix", 
        required=True, type=str, default="", help="Prefix path for HTCondor environment")
    parser.add_argument(
        "--policy_filename", 
        type=str, default="best_model.pth", help="Filename of the policy to load")
    parser.add_argument(
        "--num_workers", 
        type=int, default=4, help="Number of rollout workers per trainer")
    parser.add_argument("--empty_buffer", 
        action="store_true", help="Whether to empty the replay buffer at the beginning of each iteration")
    parser.add_argument(
        "--num_iterations", 
        type=int, default=50, help="Number of training iterations per benchmark")
    parser.add_argument(
        "--steps_per_iteration", 
        type=int, default=5, help="Number of training steps per DAgger iteration")
    parser.add_argument(
        "--num_samples", 
        type=int, default=256, help="Number of samples to generate per benchmark")
    parser.add_argument(
        "--batch_size",
        type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--use_strict_rule", 
        action="store_true", help="Use strict rules for trajectory collection")
    parser.add_argument(
        "--use_multiprocessors", 
        action="store_true", help="Whether to use multi-processors")
    parser.add_argument(
        "--disable_oracle_cache", 
        action="store_true", help="Disable caching in the oracle")
    parser.add_argument(
        "--reduced_memory_mode", 
        action="store_true", help="Whether to use reduced memory mode in the oracle")
    parser.add_argument(
        "--max_steps", 
        type=int, default=256, help="Maximum steps per episode")
    parser.add_argument(
        "--disable_wandb", 
        action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument(
        "--device", 
        type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument(
        "--seed", 
        type=int, default=42, help="Random seed for training")
    parser.add_argument(
        "--output_file", 
        type=Path, required=True, help="File to save generated configurations")
    args = parser.parse_args()

    shared_args = {
        "log_directory": args.log_directory,
        "disable_wandb": args.disable_wandb,
        "disable_oracle_cache": args.disable_oracle_cache,
        "reduced_memory_mode": args.reduced_memory_mode,
        "num_workers": args.num_workers,
        "num_iterations": args.num_iterations,
        "steps_per_iteration": args.steps_per_iteration,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "use_strict_rule": args.use_strict_rule,
        "empty_buffer": args.empty_buffer,
        "policy_filename": args.policy_filename,
        "condor_dir_prefix": args.condor_dir_prefix,
        "use_multiprocessors": args.use_multiprocessors,
        "seed": args.seed
    }

    all_configs = get_all_configs(args.root, shared_args)

    lines = []
    for benchmark_args, _ in all_configs:

        python_prefix = [
            "-m",
            "dagger.train",
        ]

        line = python_prefix.copy()
        for k, v in benchmark_args.items():
            arg_key = "--" + k
            if type(v) == bool:
                if v:
                    line.append(arg_key)
            else:
                if v:
                    line.append(arg_key)
                    line.append(str(v))
        line = " ".join(line)

        lines.append(line)

    with args.output_file.open("w") as f:
        for idx, line in enumerate(lines):
            if idx == len(lines) - 1:
                f.write(line)
            else:
                f.write(line + "\n")    


if __name__ == "__main__":
    main()