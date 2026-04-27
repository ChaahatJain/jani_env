import argparse

from pathlib import Path


def gen_command_for_benchmark(benchmark_dir: Path | str, domain_name: str, shared_args: dict[str, any]):

    condor_prefix = shared_args.get("condor_dir_prefix", "")

    benchmark_path = Path(benchmark_dir)
    jani_model = benchmark_path / "model.jani"
    assert jani_model.exists(), f"Model file {jani_model} does not exist in benchmark directory {benchmark_dir}"
    property_file = benchmark_path / "pa_model_random_starts_100000.jani"
    assert property_file.exists(), f"Property file {property_file} does not exist in benchmark directory {benchmark_dir}"

    log_dir = shared_args.get("log_dir", "logs")
    domain_log_dir = condor_prefix / Path(log_dir) / domain_name
    benchmark_log_dir = domain_log_dir / benchmark_path.name
    benchmark_log_dir.mkdir(parents=True, exist_ok=True)

    model_save_dir = shared_args.get("model_save_dir", "models")
    domain_model_save_dir = condor_prefix / Path(model_save_dir) / domain_name
    benchmark_model_save_dir = domain_model_save_dir / benchmark_path.name
    repair_policy_save_dir = benchmark_model_save_dir / "repair_policies"

    cmd_args = {
        "jani_model": str(condor_prefix / jani_model),
        "jani_property": str(condor_prefix / property_file),
        "start_states": str(condor_prefix / property_file),
        "repair_save_dir": str(repair_policy_save_dir),
        "log_dir": str(benchmark_log_dir),
        "seed": shared_args.get("seed", 42),
        "max_steps": shared_args.get("max_steps", 256),
        "max_training_idx": shared_args.get("max_training_idx", 80000),
        "goal_reward": shared_args.get("goal_reward", 1.0),
        "failure_reward": shared_args.get("failure_reward", -1.0),
        "unsafe_reward": shared_args.get("unsafe_reward", -0.01),
        "device": shared_args.get("device", "cpu"),
    }

    # Add optional boolean flags
    if shared_args.get("use_oracle", False):
        cmd_args["use_oracle"] = True
    if shared_args.get("disable_oracle_cache", False):
        cmd_args["disable_oracle_cache"] = True
    if shared_args.get("no_memory_reduced_mode", False):
        cmd_args["no_memory_reduced_mode"] = True

    return cmd_args


def gen_commands(root_dir: Path | str, shared_args: dict[str, any]):
    root_dir = Path(root_dir)
    list_cmds = []
    for domain_dir in root_dir.iterdir():
        domain_name = domain_dir.name
        for benchmark_dir in domain_dir.iterdir():
            cmd_args = gen_command_for_benchmark(benchmark_dir, domain_name, shared_args)

            python_prefix = ["eval_repaired_policy.py"]
            line = python_prefix.copy()
            for k, v in cmd_args.items():
                arg_key = "--" + k
                if type(v) == bool:
                    if v:
                        line.append(arg_key)
                else:
                    if v:
                        line.append(arg_key)
                        line.append(str(v))
            line = " ".join(line)

            list_cmds.append(line)

    return list_cmds

def main():
    parser = argparse.ArgumentParser(description="Generate commands for HTCondor evaluation of repaired policies.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing domain subdirectories.")
    parser.add_argument("--condor_prefix", required=True, type=str, default="", help="Prefix path for HTCondor environment")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base directory for logs.")
    parser.add_argument("--model_save_dir", type=str, default="models", help="Base directory for saved models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_steps", type=int, default=256, help="Maximum steps per episode.")
    parser.add_argument("--max_training_idx", type=int, default=80000, help="Maximum training index for evaluation.")
    parser.add_argument("--goal_reward", type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument("--failure_reward", type=float, default=-1.0, help="Reward for failure.")
    parser.add_argument("--unsafe_reward", type=float, default=-0.01, help="Reward for unsafe states.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on.")
    parser.add_argument("--use_oracle", action="store_true", help="Use oracle for evaluation.")
    parser.add_argument("--disable_oracle_cache", action="store_true", help="Disable oracle cache.")
    parser.add_argument("--no_memory_reduced_mode", action="store_true", help="Disable memory reduced mode.")
    parser.add_argument("--output_file", type=Path, required=True, help="File to save generated commands.")

    args = parser.parse_args()

    shared_args = {
        "condor_dir_prefix": args.condor_prefix,
        "log_dir": args.log_dir,
        "model_save_dir": args.model_save_dir,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_training_idx": args.max_training_idx,
        "goal_reward": args.goal_reward,
        "failure_reward": args.failure_reward,
        "unsafe_reward": args.unsafe_reward,
        "device": args.device,
        "use_oracle": args.use_oracle,
        "disable_oracle_cache": args.disable_oracle_cache,
        "no_memory_reduced_mode": args.no_memory_reduced_mode,
    }

    commands = gen_commands(args.root_dir, shared_args)
    with open(args.output_file, "w") as f:
        for idx, line in enumerate(commands):
            if idx == len(commands) - 1:
                f.write(line)
            else:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
