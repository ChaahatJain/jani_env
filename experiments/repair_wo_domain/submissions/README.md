# HTCondor submissions

Run submission commands from `experiments/repair_wo_domain` so that job logs
continue to be written to the shared `logs/` directory.

- `repair/`: policy-repair jobs
- `policy_comparison/`: before/after policy comparisons
- `fault_collection/`: fault-collection DAG and jobs
- `action_shield/`: action-shield DAGs and jobs

Examples:

```bash
condor_submit submissions/repair/transport_linetrack_20_10_repair.sub
condor_submit submissions/policy_comparison/transport_linetrack_17_10_compare_milp.sub
condor_submit_dag submissions/fault_collection/transport_linetrack_20_10_collect_faults.dag
condor_submit_dag submissions/action_shield/transport_linetrack_20_10_action_shield.dag
```
