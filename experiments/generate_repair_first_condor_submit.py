#! /usr/bin/env python3
"""
Generate Condor submit file for the "repair-first, then RL" experiment.

Workflow per job:
  1. Bootstrap a random policy (pipeline.py with --bootstrap_timesteps 1)
  2. Repair it on 1000 fixed start states until convergence or OOM
  3. Run RL training starting from the repaired checkpoint

Results are stored under artifacts/repair_first_pipeline/ to separate
them from the original pipeline experiments.
"""

import os
import os.path
from pathlib import Path

GRB_LICENSE_FILE = Path(os.environ.get("GRB_LICENSE_FILE", "/home/neuronet_team119/gurobi.lic"))

BENCHMARKS_DIR = "/home/neuronet_team119/jani_env/benchmarks_generator/benchmarks"
ARTIFACTS_DIR  = "/home/neuronet_team119/jani_env/artifacts/repair_first_pipeline"
EXP_DIR        = os.path.dirname(os.path.realpath(__file__))

time_limit    = 60 * 60 * 6   # 6 hours
memory_limit  = "128G"

output_dir      = EXP_DIR
submit_filename = "repair_first_experiments.sub"

REPAIR_METHODS       = ["milp", "spec"]
ALGOS                = ["mask_ppo", "safe_dqn", "ppo_lag"]
MAX_ITERATIONS       = 1000
REPAIR_START_STATES  = 1000       # repair on 1000 fixed start states
RL_TIMESTEPS         = 500_000


def get_prefix_problem_combinations(base_dir=BENCHMARKS_DIR):
    combinations = []
    base_path = Path(base_dir)
    for model_path in base_path.glob("*/*/model.jani"):
        if model_path.is_file():
            category = model_path.parent.parent.name
            instance = model_path.parent.name
            combinations.append((category, instance, model_path))
    return combinations


def build_job(category: str, instance: str, model_path: Path,
              algo: str, repair_method: str) -> str:
    """
    Build the argument string for execute_repair_then_rl.sh.

    Format:  --rl_algo <algo> <repair_args> --- <rl_args>
    """
    jani_model    = str(model_path)
    jani_property = str(model_path)
    start_states  = str(model_path.parent / "pa_model_random_starts_100000.jani")

    # Output directory — keeps results clearly separated
    base_output = str(Path(ARTIFACTS_DIR) / category / instance / algo / repair_method)
    repair_output = f"{base_output}/repair"
    rl_output     = f"{base_output}/rl"

    # ── Phase 1: Repair args (passed to pipeline.py) ──
    repair_args = (
        f"--jani_model {jani_model} "
        f"--jani_property {jani_property} "
        f"--start_states {start_states} "
        f"--max_steps 1100 "
        f"--traces_per_iteration {REPAIR_START_STATES} "
        f"--max_iterations {MAX_ITERATIONS} "
        f"--output_dir {repair_output} "
        f"--device cpu "
        f"--accumulate_faults "
        f"--repair_method {repair_method} "
        f"--bootstrap_timesteps 1"
    )

    # ── Phase 2: RL args (passed to the chosen RL trainer) ──
    rl_args = (
        f"--jani_model {jani_model} "
        f"--jani_property {jani_property} "
        f"--start_states {start_states} "
        f"--max_steps 1100 "
        f"--total_timesteps {RL_TIMESTEPS} "
        f"--model_save_dir {rl_output}/models "
        f"--device cpu "
        f"--use_oracle "
        f"--disable_eval "
        f"--disable_wandb "
        f"--seed 42 "
        f"--perf_file {rl_output}/perf.csv"
    )
    if algo == "safe_dqn":
        rl_args += f" --log_dir {rl_output}/logs"

    # Combine:  --rl_algo <algo>  <repair_args>  ---  <rl_args>
    return f"--rl_algo {algo} {repair_args} --- {rl_args}"


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

    # Ensure the shared log directory exists
    Path("/home/neuronet_team119/jani_env/test_logs").mkdir(exist_ok=True)

    exe_path = f"{EXP_DIR}/execute_repair_then_rl.sh"

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
