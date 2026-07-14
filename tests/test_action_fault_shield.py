import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from experiments.repair_wo_domain.action_fault_shield import (
    ActionFaultClassifier,
    ActionFaultShield,
    MultiTaskActionFaultClassifier,
    summarize_per_action_evaluation,
)
from experiments.repair_wo_domain.collect_policy_faults import (
    merge as merge_fault_collection,
    write_action_label_datasets,
)
from experiments.repair_wo_domain.train_action_fault_classifiers import (
    load_action_dataset,
)
from experiments.repair_wo_domain.evaluate_action_fault_shield import (
    merge as merge_shield_evaluation,
    shielded_rollout,
)


def test_action_label_files_include_faults_and_non_faults(tmp_path):
    classifications = {
        ((1, 2), 0): False,
        ((3, 4), 0): True,
        ((1, 2), 1): False,
    }
    metadata = write_action_label_datasets(
        tmp_path, classifications, n_actions=2, obs_dim=2
    )

    assert metadata["0"]["examples"] == 2
    assert metadata["0"]["faults"] == 1
    assert metadata["0"]["non_faults"] == 1
    with np.load(tmp_path / "labels_action_0.npz") as data:
        assert data["observations"].shape == (2, 2)
        assert sorted(data["labels"].tolist()) == [0, 1]


def test_training_loader_deduplicates_cross_shard_states(tmp_path):
    shards = []
    for shard_id, observations, labels in [
        (0, [[1, 1], [2, 2]], [1, 0]),
        (1, [[1, 1], [3, 3]], [1, 0]),
    ]:
        directory = tmp_path / f"shard_{shard_id}"
        directory.mkdir()
        np.savez_compressed(
            directory / "labels_action_0.npz",
            observations=np.asarray(observations, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.uint8),
            action=np.asarray(0, dtype=np.int64),
        )
        shards.append(
            {
                "shard_id": shard_id,
                "directory": str(directory),
                "actions": {
                    "0": {
                        "file": "labels_action_0.npz",
                        "examples": 2,
                        "faults": 1,
                        "non_faults": 1,
                    }
                },
            }
        )
    manifest = {
        "observation_dim": 2,
        "action_names": ["move"],
        "policy_sha256": "test",
        "shards": shards,
    }

    features, labels, counts = load_action_dataset(
        manifest, action=0, max_negatives=100, seed=0
    )

    assert len(features) == 3
    assert int(labels.sum()) == 1
    assert counts["unique_faults"] == 1
    assert counts["sampled_unique_non_faults"] == 2


def test_collection_merge_writes_action_label_manifest(tmp_path):
    collection = tmp_path / "collection"
    for shard_id in range(2):
        shard = collection / f"shard_{shard_id}"
        shard.mkdir(parents=True)
        np.savez_compressed(
            shard / "labels_action_0.npz",
            observations=np.asarray([[shard_id, 1]], dtype=np.float32),
            labels=np.asarray([shard_id], dtype=np.uint8),
            action=np.asarray(0, dtype=np.int64),
        )
        (shard / "faults.jsonl").write_text("", encoding="utf-8")
        summary = {
            "format_version": 1,
            "fault_definition": "test",
            "action_scope": "all-applicable",
            "coverage": "test",
            "jani_model": "/model",
            "jani_property": "/property",
            "start_states": "/starts",
            "policy": "/policy",
            "policy_sha256": "digest",
            "start_state_pool_size": 2,
            "num_shards": 2,
            "shard_id": shard_id,
            "processed_start_states": 1,
            "total_steps": 1,
            "action_evaluations": 1,
            "oracle_queries": 1,
            "classification_cache_hits": 0,
            "terminations": {"goal": 1},
            "action_names": ["move"],
            "action_labels_saved": True,
            "action_label_files": {
                "0": {
                    "file": "labels_action_0.npz",
                    "examples": 1,
                    "faults": shard_id,
                    "non_faults": 1 - shard_id,
                }
            },
        }
        (shard / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    output = collection / "merged"
    merge_fault_collection(
        Namespace(
            input_dir=str(collection),
            output_dir=str(output),
            expected_shards=2,
            allow_incomplete=False,
        )
    )

    manifest = json.loads(
        (output / "action_labels_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["observation_dim"] == 2
    assert len(manifest["shards"]) == 2
    assert (output / "summary.json").is_file()


def test_shield_combines_applicability_and_classifier_predictions(tmp_path):
    model = ActionFaultClassifier(input_dim=2, hidden_dims=[], dropout=0.0)
    with torch.no_grad():
        model.network[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.network[0].bias.zero_()

    torch.save(
        {
            "kind": "neural",
            "action": 0,
            "action_name": "move",
            "input_dim": 2,
            "hidden_dims": [],
            "dropout": 0.0,
            "model_state_dict": model.state_dict(),
            "normalization_mean": np.zeros(2, dtype=np.float32),
            "normalization_scale": np.ones(2, dtype=np.float32),
            "threshold": 0.5,
        },
        tmp_path / "action_0.pth",
    )
    torch.save(
        {
            "kind": "constant",
            "action": 1,
            "action_name": "wait",
            "constant_probability": 0.0,
            "threshold": 0.5,
        },
        tmp_path / "action_1.pth",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "classifiers": [
                    {"action": 0, "checkpoint": "action_0.pth"},
                    {"action": 1, "checkpoint": "action_1.pth"},
                ]
            }
        ),
        encoding="utf-8",
    )

    shield = ActionFaultShield.load(tmp_path)
    shielded, blocked, probabilities = shield.masks(
        np.asarray([2.0, 0.0], dtype=np.float32),
        np.asarray([True, True]),
    )

    assert probabilities[0] > 0.5
    assert blocked.tolist() == [True, False]
    assert shielded.tolist() == [False, True]


def test_multitask_shield_uses_shared_backbone_and_separate_heads(tmp_path):
    model = MultiTaskActionFaultClassifier(
        input_dim=2, hidden_dims=[2], dropout=0.0, n_actions=2
    )
    with torch.no_grad():
        model.backbone[0].weight.copy_(torch.eye(2))
        model.backbone[0].bias.zero_()
        model.heads[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.heads[0].bias.zero_()
        model.heads[1].weight.copy_(torch.tensor([[-1.0, 0.0]]))
        model.heads[1].bias.zero_()
    torch.save(
        {
            "input_dim": 2,
            "hidden_dims": [2],
            "dropout": 0.0,
            "n_actions": 2,
            "model_state_dict": model.state_dict(),
            "normalization_mean": np.zeros(2, dtype=np.float32),
            "normalization_scale": np.ones(2, dtype=np.float32),
        },
        tmp_path / "multitask_model.pth",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_type": "shared_multitask",
                "checkpoint": "multitask_model.pth",
                "classifiers": [
                    {
                        "action": 0,
                        "action_name": "right",
                        "kind": "neural",
                        "threshold": 0.5,
                    },
                    {
                        "action": 1,
                        "action_name": "left",
                        "kind": "neural",
                        "threshold": 0.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    shield = ActionFaultShield.load(tmp_path)
    probabilities = shield.fault_probabilities(np.asarray([2.0, 0.0]))

    assert shield.shared_model is not None
    assert probabilities[0] > 0.5
    assert probabilities[1] < 0.5


def test_per_action_evaluation_reports_faults_fixed_and_missed():
    result = summarize_per_action_evaluation(
        [
            {
                "action": 0,
                "action_name": "move",
                "kind": "neural",
                "dataset_counts": {
                    "unique_faults": 100,
                    "policy_selected_unique_faults": 7,
                },
                "test_metrics": {
                    "faults": 20,
                    "precision": 0.9,
                    "confusion_matrix": [[75, 5], [2, 18]],
                },
            },
            {
                "action": 1,
                "action_name": "wait",
                "kind": "constant",
                "dataset_counts": {"unique_faults": 0},
            },
        ]
    )

    assert result[0]["collected_unique_faults"] == 100
    assert result[0]["policy_selected_unique_faults"] == 7
    assert result[0]["held_out_faults_fixed"] == 18
    assert result[0]["held_out_faults_missed"] == 2
    assert result[0]["held_out_fix_rate_percent"] == 90.0
    assert result[0]["held_out_non_faults_wrongly_blocked"] == 5
    assert result[0]["held_out_safe_actions_wrongly_blocked"] == 5
    assert result[1]["held_out_fix_rate_percent"] is None
    assert result[1]["policy_selected_unique_faults"] == 0


def test_training_script_writes_loadable_per_action_models(tmp_path):
    collection = tmp_path / "collection"
    shard = collection / "shard_0"
    merged = collection / "merged"
    shard.mkdir(parents=True)
    merged.mkdir()

    fault_states = np.column_stack(
        [
            np.linspace(1.0, 2.0, 20, dtype=np.float32),
            np.ones(20, dtype=np.float32),
        ]
    )
    safe_states = np.column_stack(
        [
            np.linspace(-2.0, -1.0, 80, dtype=np.float32),
            np.zeros(80, dtype=np.float32),
        ]
    )
    observations = np.concatenate([fault_states, safe_states])
    labels = np.concatenate(
        [np.ones(len(fault_states), dtype=np.uint8), np.zeros(len(safe_states), dtype=np.uint8)]
    )
    np.savez_compressed(
        shard / "labels_action_0.npz",
        observations=observations,
        labels=labels,
        action=np.asarray(0, dtype=np.int64),
    )
    np.savez_compressed(
        shard / "labels_action_1.npz",
        observations=safe_states,
        labels=np.zeros(len(safe_states), dtype=np.uint8),
        action=np.asarray(1, dtype=np.int64),
    )
    manifest = {
        "format_version": 1,
        "observation_dim": 2,
        "action_names": ["move", "wait"],
        "policy_sha256": "synthetic",
        "shards": [
            {
                "shard_id": 0,
                "directory": str(shard),
                "actions": {
                    "0": {
                        "file": "labels_action_0.npz",
                        "examples": 100,
                        "faults": 20,
                        "non_faults": 80,
                    },
                    "1": {
                        "file": "labels_action_1.npz",
                        "examples": 80,
                        "faults": 0,
                        "non_faults": 80,
                    },
                },
            }
        ],
    }
    (merged / "action_labels_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output = tmp_path / "models"
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "repair_wo_domain"
        / "train_action_fault_classifiers.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--collection-dir",
            str(merged),
            "--output-dir",
            str(output),
            "--hidden-dims",
            "8",
            "--epochs",
            "3",
            "--patience",
            "2",
            "--batch-size",
            "32",
            "--minimum-positive-examples",
            "5",
            "--seed",
            "0",
        ],
        check=True,
    )

    shield = ActionFaultShield.load(output)
    output_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert output_manifest["model_type"] == "shared_multitask"
    assert (output / "multitask_model.pth").is_file()
    assert len(output_manifest["per_action_evaluation"]) == 2
    assert output_manifest["per_action_evaluation"][0]["collected_unique_faults"] == 20
    assert output_manifest["per_action_evaluation"][1]["collected_unique_faults"] == 0
    assert len(shield.classifiers) == 2
    assert shield.shared_model is not None
    assert shield.classifiers[0].kind == "neural"
    assert shield.classifiers[1].kind == "constant"


def test_all_applicable_actions_blocked_stops_episode():
    class FakeActionSpace:
        n = 2

    class FakeEnv:
        action_space = FakeActionSpace()

        def reset(self, options=None):
            return np.asarray([1.0, 2.0], dtype=np.float32), {}

        def action_mask(self):
            return np.asarray([True, True])

        def is_state_action_fault(self, observation, action):
            return action == 1

        def step(self, action):
            raise AssertionError("The environment must not step through an all-blocked state")

    class FakeShield:
        def masks(self, observation, applicable):
            return (
                np.asarray([False, False]),
                np.asarray([True, True]),
                np.asarray([0.9, 0.8], dtype=np.float32),
            )

    result, steps, stats = shielded_rollout(
        FakeEnv(),
        torch.nn.Identity(),
        FakeShield(),
        start_index=0,
        args=Namespace(max_steps=10, max_state_visits=3),
        device=torch.device("cpu"),
    )

    assert result == "all_blocked"
    assert steps == 0
    assert stats["decision_states"] == 1
    assert stats["blocked_action_occurrences"] == [1, 1]
    assert stats["policy_fault_blocked_occurrences"] == [0, 1]
    assert stats["policy_fault_missed_occurrences"] == [0, 0]
    assert stats["safe_policy_action_blocked_occurrences"] == [0, 0]


def test_evaluation_merge_reports_goal_avoid_and_unique_all_blocked(tmp_path):
    evaluation = tmp_path / "evaluation"
    common = {
        "shield_rule": "test",
        "jani_model": "/model",
        "start_states": "/starts",
        "policy": "/policy",
        "policy_sha256": "policy",
        "classifiers_dir": "/classifiers",
        "classifiers_manifest_sha256": "classifiers",
        "classifier_metadata": [
            {"action": 0, "action_name": "move", "kind": "neural", "threshold": 0.5}
        ],
        "start_state_pool_size": 2,
        "num_shards": 2,
        "max_steps": 10,
        "max_state_visits": 3,
    }
    for shard_id, blocked_state in [(0, [1.0, 2.0]), (1, [1.0, 2.0])]:
        shard = evaluation / f"shard_{shard_id}"
        shard.mkdir(parents=True)
        summary = {
            **common,
            "shard_id": shard_id,
            "processed_start_states": 1,
            "baseline_terminations": {"goal": 1},
            "shielded_terminations": (
                {"all_blocked": 1} if shard_id == 0 else {"failure": 1}
            ),
            "shielded_decision_states": 2,
            "states_with_any_classifier_block": 1,
            "blocked_action_occurrences": {"move": 1},
            "policy_fault_blocked_steps": 2,
            "policy_fault_missed_steps": 3,
            "safe_policy_action_blocked_steps": 4,
            "policy_fault_blocked_occurrences": {"move": 2},
            "policy_fault_missed_occurrences": {"move": 3},
            "safe_policy_action_blocked_occurrences": {"move": 4},
        }
        (shard / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        np.savez_compressed(
            shard / "all_blocked_states.npz",
            observations=(
                np.asarray([blocked_state], dtype=np.float32)
                if shard_id == 0
                else np.empty((0, 2), dtype=np.float32)
            ),
        )
        (shard / "episodes.csv").write_text(
            "start_state_index,baseline_termination,baseline_steps,"
            "shielded_termination,shielded_steps,shielded_decision_states,"
            "states_with_any_classifier_block,all_actions_blocked\n"
            f"{shard_id},goal,1,"
            f"{'all_blocked' if shard_id == 0 else 'failure'},1,2,1,"
            f"{1 if shard_id == 0 else 0}\n",
            encoding="utf-8",
        )

    output = evaluation / "merged"
    merge_shield_evaluation(
        Namespace(
            input_dir=str(evaluation),
            output_dir=str(output),
            expected_shards=2,
            allow_incomplete=False,
        )
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_metrics"]["goal_percent"] == 100.0
    assert summary["shielded_metrics"]["avoid_percent"] == 50.0
    assert summary["unique_all_actions_blocked_states"] == 1
    assert summary["policy_fault_blocked_steps"] == 4
    assert summary["policy_fault_missed_steps"] == 6
    assert summary["safe_policy_action_blocked_steps"] == 8
