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
    rl_policy_save_dir = benchmark_model_save_dir / "rl_policies"
    rl_policy_save_dir.mkdir(parents=True, exist_ok=True)
    repair_policy_save_dir = benchmark_model_save_dir / "repair_policies"
    repair_policy_save_dir.mkdir(parents=True, exist_ok=True)

    cmd_args = {
        "jani_model": str(condor_prefix / jani_model),
        "jani_property": str(condor_prefix / property_file),
        "start_states": str(condor_prefix / property_file),
        "disable_wandb": True,
        "max_iterations": shared_args.get("max_iterations", 10),
        "num_traces_per_iter": shared_args.get("num_traces_per_iter", 10),
        "model_save_dir": rl_policy_save_dir,
        "repair_save_dir": repair_policy_save_dir,
        "log_dir": benchmark_log_dir,
        "seed": shared_args.get("seed", 42),
        "total_timesteps": shared_args.get("total_timesteps", 512000),
    }

    return cmd_args


def gen_commands(root_dir: Path | str, shared_args: dict[str, any]):
    root_dir = Path(root_dir)
    list_cmds = []
    for domain_dir in root_dir.iterdir():
        domain_name = domain_dir.name
        for benchmark_dir in domain_dir.iterdir():
            cmd_args = gen_command_for_benchmark(benchmark_dir, domain_name, shared_args)

            python_prefix = ["pipeline_enum.py"]
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
    parser = argparse.ArgumentParser(description="Generate commands for HTCondor pipeline.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing domain subdirectories.")
    parser.add_argument("--condor_prefix", required=True, type=str, default="", help="Prefix path for HTCondor environment")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base directory for logs.")
    parser.add_argument("--model_save_dir", type=str, default="models", help="Base directory for saving models.")
    parser.add_argument("--max_iterations", type=int, default=50, help="Maximum iterations for the pipeline.")
    parser.add_argument("--num_traces_per_iter", type=int, default=200, help="Number of traces to sample per iteration.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timesteps for training policies.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_file", type=Path, required=True, help="File to save generated commands.")

    args = parser.parse_args()

    shared_args = {
        "condor_dir_prefix": args.condor_prefix,
        "log_dir": args.log_dir,
        "model_save_dir": args.model_save_dir,
        "max_iterations": args.max_iterations,
        "num_traces_per_iter": args.num_traces_per_iter,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
    }

    commands = gen_commands(args.root_dir, shared_args)
    with open(args.output_file, "w") as f:
        for idx, line in enumerate(commands):
            if idx == len(commands) - 1:
                f.write(line)
            else:
                f.write(line + "\n")    