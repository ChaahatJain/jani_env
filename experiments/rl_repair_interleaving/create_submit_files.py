#! /usr/bin/env python3

import os.path
from pathlib import Path

BENCHMARKS_DIR = "/home/jain/jani_env/benchmarks_generator/benchmarks"
ARTIFACTS_DIR  = "/home/jain/jani_env/artifacts/rl_repair_interleaving"
EXP_DIR        = os.path.dirname(os.path.realpath(__file__))

time_limit    = 60 * 60 * 6   # 6 hours
memory_limit  = "100G"

output_dir      = EXP_DIR
submit_filename = "rl_repair_interleaving_experiments.sub"

RL_ALGORITHMS       = ["mask_ppo", "ppo_lag"]
MAX_STEPS_PER_TRACE = 1500
TOTAL_TIMESTEPS     = 10_000_000
NUM_EVAL_EPISODES   = 100
EVALUATION_FREQUENCY = 1000
REPAIR_FREQUENCY     = 500
NUM_REPAIR_EPISODES  = 50


def get_prefix_problem_combinations(base_dir=BENCHMARKS_DIR):
    combinations = []
    base_path = Path(base_dir)
    for model_path in base_path.glob("*/*/model.jani"):
        if model_path.is_file():
            category = model_path.parent.parent.name
            instance = model_path.parent.name
            combinations.append((category, instance, model_path))
    return combinations


def build_job(category: str, instance: str, model_path: Path, algo: str, repair: bool) -> tuple[str, str]:
    jani_model    = str(model_path)
    jani_property = str(model_path)
    start_states  = str(model_path.parent / "pa_model_random_starts_100000.jani")

    repair_suffix    = "with_repair" if repair else "no_repair"
    job_name         = f"{algo}_{category}_{instance}_{repair_suffix}"
    output_dir_job   = str(Path(ARTIFACTS_DIR) / category / instance / algo / repair_suffix)
    performance_logs = str(Path(ARTIFACTS_DIR) / "logs" / f"{job_name}.csv")

    args = (
        f"--algo {algo} "
        f"--jani_model {jani_model} "
        f"--jani_property {jani_property} "
        f"--start_states {start_states} "
        f"--goal_reward 1.0 "
        f"--failure_reward -1.0 "
        f"--max_steps {MAX_STEPS_PER_TRACE} "
        f"--total_timesteps {TOTAL_TIMESTEPS} "
        f"--n_eval_episodes {NUM_EVAL_EPISODES} "
        f"--eval_freq {EVALUATION_FREQUENCY} "
        f"--n_steps 1000 "
        f"--model_save_dir {output_dir_job} "
        "--disable_wandb "
        "--verbose 1 --device cpu --seed 50 --save_all_checkpoints "
        f"--perf_file {performance_logs} "
    )

    if repair:
        repair_logs = str(Path(ARTIFACTS_DIR) / "logs" / f"{job_name}_repair_logs.csv")
        args += (
            f"--enable_repair "
            f"--repair_freq {REPAIR_FREQUENCY} "
            f"--repair_episodes {NUM_REPAIR_EPISODES} "
            f"--repair_algo milp "
            f"--repair_log_file {repair_logs}"
        )

    return args, job_name


def create_submit_file(jobs: list, output_path: Path, exe_path: str):
    mapping_path = output_path.parent / "job_id_mapping.txt"
    with open(mapping_path, "w") as f:
        f.write("ProcId\tjob_name\n")
        for idx, (args, job_name) in enumerate(jobs):
            f.write(f"{idx}\t{job_name}\n")
    print(f"Created job mapping: {mapping_path}")

    submit_content = f"""universe = docker
docker_image = chaahatjain/jani_in_python:latest
executable = {exe_path}
getenv = HOME
+WantGPUHomeMounted = true
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT

requirements = UidDomain == "cs.uni-saarland.de"

request_GPUs = 0
request_CPUs = 1
request_memory = {memory_limit}
periodic_remove = (time() - JobStartDate) > {time_limit} 

output = logs/$(ClusterId).$(ProcId).out
error  = logs/$(ClusterId).$(ProcId).err
log    = logs/$(ClusterId).$(ProcId).log
max_idle = 100
arguments = $(args)
queue args from (
"""
    for args, job_name in jobs:
        submit_content += f"  {args}\n"
    submit_content += ")\n"

    with open(output_path, "w") as f:
        f.write(submit_content)

    print(f"Created submit file: {output_path}")
    print(f"Total jobs: {len(jobs)}")


def main():
    logs_path = Path(output_dir)
    logs_path.mkdir(exist_ok=True)

    exe_path = f"{EXP_DIR}/execute_experiment.sh"

    benchmarks = get_prefix_problem_combinations()
    jobs = []

    for category, instance, model_path in benchmarks:
        for algo in RL_ALGORITHMS:
            for repair in [True]:
                args, job_name = build_job(category, instance, model_path, algo, repair)
                jobs.append((args, job_name))

    submit_file_path = logs_path / submit_filename
    create_submit_file(jobs, submit_file_path, exe_path)

    print(f"\nTo submit: condor_submit {submit_file_path}")


if __name__ == "__main__":
    main()
