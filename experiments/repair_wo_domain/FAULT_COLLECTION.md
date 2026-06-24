# Fault collection for `transport/linetrack_20_10`

## Experiment definition

This experiment evaluates the fixed MaskedPPO seed-0 policy on every initial
state in `pa_model_random_starts_100000.jani`. It does not train, repair, or
otherwise modify the policy.

At every visited state, the collector checks the action selected by the policy
with `JANIEnv.is_state_action_fault`. The underlying oracle defines a fault as
a safe source state and an action for which at least one successor is unsafe.
Every unique `(observation, action)` fault is retained. Repeated occurrences
are counted, and no `break` is used after the first fault in a trace.

This is exhaustive over the 100,000 supplied initial-state policy rollouts,
up to the configured 2,000-step and cycle limits. It is not a claim that every
state in the complete reachable state graph has been enumerated.

## Cluster execution

From `experiments/repair_wo_domain`, run:

```bash
mkdir -p logs
condor_submit_dag transport_linetrack_20_10_collect_faults.dag
```

The workflow launches 20 disjoint shards of approximately 5,000 initial states and runs the
merge only after every shard succeeds.

## Final outputs

The merged result is written below:

```text
artifacts/fault_collection/transport_linetrack_20_10/
  mask_ppo_best_params_seed0/merged/
    faults.csv
    faults.jsonl
    summary.json
```

- `faults.csv` is the human-readable table.
- `faults.jsonl` preserves typed arrays for later programs.
- `summary.json` records coverage, policy hash, termination counts, oracle
  calls, unique fault counts, repeated occurrences, and counts by action.

If faults for every applicable action at each policy-visited state are wanted
instead, change `--action-scope policy` to `--action-scope all-applicable` in
the collection submit file. That broader setting still explores states reached
by the fixed-policy rollouts; it does not enumerate the complete state graph.
