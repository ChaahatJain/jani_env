# Shared-backbone action classifier shield

This workflow implements the requested experiment:

1. Run the fixed seed-0 policy from all 100,000 supplied initial states.
2. At every policy-visited state, query the oracle for every applicable action.
3. Retain both fault and non-fault labels, grouped by action.
4. Train one multitask model: a shared state encoder with a separate binary
   output head per action.
5. During evaluation, use:

   ```text
   shielded_mask = applicability_mask AND NOT predicted_fault_mask
   ```

6. Report `%Goal`, `%Avoid`, `%Cycle`, action-block counts, and states where the
   classifiers block every applicable action.

An episode stops with termination `all_blocked` when the shield mask is empty.
No classifier-blocked action is silently restored.

## Submit the complete workflow

From `experiments/repair_wo_domain` on the cluster:

```bash
mkdir -p logs
condor_submit_dag submissions/action_shield/transport_linetrack_20_10_action_shield.dag
```

The DAG runs these stages in order:

```text
40 collection shards
  -> merge collection
  -> train shared-backbone multitask classifier
  -> 40 evaluation shards
  -> merge evaluation
```

The collection uses `--action-scope all-applicable`; using the older
policy-action-only dataset would not provide valid training data for all four
action heads.

## Outputs

```text
artifacts/action_shield/transport_linetrack_20_10/
  mask_ppo_best_params_seed0/
    collection/
      shard_*/
        faults.jsonl
        labels_action_0.npz
        labels_action_1.npz
        labels_action_2.npz
        labels_action_3.npz
        summary.json
      merged/
        faults.csv
        faults.jsonl
        action_labels_manifest.json
        summary.json
    classifiers/
      multitask_model.pth
      multitask_training_metrics.json
      action_0_metrics.json
      ...
      manifest.json
    evaluation/
      shard_*/
      merged/
        summary.json
        episodes.csv
        all_blocked_states.npz
```

The main result is `evaluation/merged/summary.json`.

The training sampler balances the action tasks so an action with many more
collected states does not dominate the shared representation. For actions with
observed faults, the threshold is selected on validation data
to reach at least 99% validation recall while maximizing precision. Test
precision, recall, false-negative rate, false-positive rate, and confusion
matrix are saved per action.

If an action has no observed faults in the exhaustive collected rollout data,
its classifier is recorded explicitly as a constant `False` classifier. This
is an empirical statement about the collected coverage, not a proof that the
action can never be faulty elsewhere.
