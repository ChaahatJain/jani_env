#! /usr/bin/env python3

import os
import os.path
import sys
from pathlib import Path

GRB_LICENSE_FILE = Path(os.environ.get("GRB_LICENSE_FILE", "/home/neuronet_team119/gurobi.lic"))



BENCHMARKS_DIR = "/home/neuronet_team119/jani_env/benchmarks_generator/benchmarks"
ARTIFACTS_DIR  = "/home/neuronet_team119/jani_env/artifacts/pipeline"
EXP_DIR        = os.path.dirname(os.path.realpath(__file__))

time_limit    = 60 * 60 * 6   # 6 hours
memory_limit  = "128G"

output_dir      = EXP_DIR
submit_filename = "pipeline_experiments.sub"

REPAIR_METHODS       = [ "milp"]
ALGOS                = [ "mask_ppo","safe_dqn", "ppo_lag"]
MAX_ITERATIONS       = 1000
TRACES_PER_ITERATION = 10000
BOOTSTRAP_TIMESTEPS  = 500_000


def get_prefix_problem_combinations(base_dir=BENCHMARKS_DIR):
    combinations = []
    base_path = Path(base_dir)
    for model_path in base_path.glob("*/*/model.jani"):
        if model_path.is_file():
            category = model_path.parent.parent.name   # e.g. two_way_line_det
            instance = model_path.parent.name          # e.g. two_way_line_20_10
            combinations.append((category, instance, model_path))
    return combinations


def build_job(category: str, instance: str, model_path: Path, algo: str, repair_method: str) -> str:
    jani_model     = str(model_path)
    jani_property  = str(model_path)
    start_states   = str(model_path.parent / "pa_model_random_starts_100000.jani")

    if algo == "mask_ppo":
        initial_policy = str(Path(ARTIFACTS_DIR) / category / instance / "rl_training" / "models" / "final_actor.pth")
        output_dir_job = str(Path(ARTIFACTS_DIR) / category / instance)

        args = (
            f"--algo pipeline "
            f"--jani_model {jani_model} "
            f"--jani_property {jani_property} "
            f"--initial_policy {initial_policy} "
            f"--start_states {start_states} "
            f"--max_steps 1100 "
            f"--traces_per_iteration {TRACES_PER_ITERATION} "
            f"--max_iterations {MAX_ITERATIONS} "
            f"--output_dir {output_dir_job} "
            f"--device cpu "
            f"--accumulate_faults "
            f"--repair_method {repair_method} "
            f"--bootstrap_timesteps {BOOTSTRAP_TIMESTEPS} "
            f"--reduced_memory_mode"
        )
    else:
        output_dir_job = str(Path(ARTIFACTS_DIR) / category / instance / algo / repair_method)

        args = (
            f"--algo {algo} "
            f"--jani_model {jani_model} "
            f"--jani_property {jani_property} "
            f"--start_states {start_states} "
            f"--max_steps 1100 "
            f"--total_timesteps {BOOTSTRAP_TIMESTEPS} "
            f"--model_save_dir {output_dir_job}/models "
            f"--device cpu "
            f"--use_oracle "
            f"--enable_repair "
            f"--repair_algo {repair_method} "
            f"--repair_freq 100 "
            f"--repair_episodes {TRACES_PER_ITERATION} "
            f"--disable_eval "
            f"--disable_wandb "
            f"--seed 42 "
            f"--perf_file {output_dir_job}/perf.csv "
            f"--repair_log_file {output_dir_job}/repair_log.csv"
        )
        if algo == "safe_dqn":
            args += f" --log_dir {output_dir_job}/logs"
    return args


def get_initial_policy(model_path: Path) -> str:
    """
    Resolve the initial policy path. Returns the path string;
    pipeline.py will train from scratch if the file doesn't exist
    (assumes pipeline.py handles --initial_policy being absent or missing).
    """
    problem = model_path.stem
    # Convention: RL training artifacts live under ARTIFACTS_DIR/<benchmark_subdir>/rl_training/models/
    benchmark_subdir = model_path.parent.parent.name  # e.g. two_way_line_det/two_way_line_20_10
    policy_path = Path(ARTIFACTS_DIR) / benchmark_subdir / "rl_training" / "models" / "final_actor.pth"
    return str(policy_path)


def create_submit_file(jobs: list, output_path: Path, exe_path: str):
    submit_content = f"""universe = docker
docker_image = chaahatjain/jani_in_python:latest
executable = {exe_path}
getenv = HOME
+WantGPUHomeMounted = true
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

requirements = UidDomain == "cs.uni-saarland.de"

request_GPUs = 0
request_CPUs = 1
request_memory = {memory_limit}

output = test_logs/$(ClusterId).$(ProcId).out
error = test_logs/$(ClusterId).$(ProcId).err
log = test_logs/$(ClusterId).$(ProcId).log
max_idle = 100
arguments = $(args)
queue args from (
"""
    for job in jobs:
        submit_content += f"  {job}\n"
    submit_content += ")\n"

    with open(output_path, "w") as f:
        f.write(submit_content)

    print(f"Created submit file: {output_path}")
    print(f"Total jobs: {len(jobs)}")


def main():

    logs_path = Path(output_dir)
    logs_path.mkdir(exist_ok=True)

    exe_path = f"{EXP_DIR}/execute_pipeline.sh"

    benchmarks = get_prefix_problem_combinations()
    jobs = []

    for category, instance, model_path in benchmarks:
        for algo in ALGOS:
            for repair_method in REPAIR_METHODS:
                job = build_job(category, instance, model_path, algo, repair_method)
                jobs.append(job)

    submit_file_path = logs_path / submit_filename
    create_submit_file(jobs, submit_file_path, exe_path)

    print(f"\nTo submit: condor_submit {submit_file_path}")


if __name__ == "__main__":
    main()
