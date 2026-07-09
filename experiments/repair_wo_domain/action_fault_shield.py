"""Per-action fault classifiers and runtime action shielding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def summarize_per_action_evaluation(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn per-action test metrics into a compact, user-facing evaluation."""
    evaluations: list[dict[str, Any]] = []
    for report in sorted(reports, key=lambda value: int(value["action"])):
        dataset_counts = report.get("dataset_counts", {})
        test_metrics = report.get("test_metrics")
        if test_metrics is None:
            test_faults = fixed = missed = false_blocks = 0
            fix_rate_percent = None
            precision_percent = None
        else:
            matrix = test_metrics["confusion_matrix"]
            if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
                raise ValueError(
                    f"Malformed confusion matrix for action {report['action']}"
                )
            false_blocks = int(matrix[0][1])
            missed = int(matrix[1][0])
            fixed = int(matrix[1][1])
            test_faults = int(test_metrics["faults"])
            if fixed + missed != test_faults:
                raise ValueError(
                    f"Confusion matrix fault count disagrees for action {report['action']}"
                )
            fix_rate_percent = (
                100.0 * fixed / test_faults if test_faults else None
            )
            precision_percent = 100.0 * float(test_metrics["precision"])
        evaluations.append(
            {
                "action": int(report["action"]),
                "action_name": str(report["action_name"]),
                "classifier_kind": str(report["kind"]),
                "collected_unique_faults": int(
                    dataset_counts.get("unique_faults", 0)
                ),
                "policy_selected_unique_faults": int(
                    dataset_counts.get("policy_selected_unique_faults", 0)
                ),
                "held_out_test_faults": test_faults,
                "held_out_faults_fixed": fixed,
                "held_out_faults_missed": missed,
                "held_out_fix_rate_percent": fix_rate_percent,
                "held_out_non_faults_wrongly_blocked": false_blocks,
                "held_out_safe_actions_wrongly_blocked": false_blocks,
                "held_out_precision_percent": precision_percent,
            }
        )
    return evaluations


class ActionFaultClassifier(nn.Module):
    """Binary MLP returning a fault logit for one fixed action."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


class MultiTaskActionFaultClassifier(nn.Module):
    """Shared state encoder with one binary fault-prediction head per action."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        n_actions: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(previous, 1) for _ in range(n_actions)])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.backbone(observations)
        return torch.cat([head(features) for head in self.heads], dim=-1)


@dataclass
class LoadedActionClassifier:
    action: int
    action_name: str
    kind: str
    threshold: float
    model: ActionFaultClassifier | None
    mean: torch.Tensor | None
    scale: torch.Tensor | None
    constant_probability: float = 0.0

    def probability(self, observation: torch.Tensor) -> float:
        if self.kind == "constant":
            return self.constant_probability
        if self.model is None or self.mean is None or self.scale is None:
            raise RuntimeError(f"Neural classifier for action {self.action} is incomplete")
        normalized = (observation - self.mean) / self.scale
        with torch.no_grad():
            return float(torch.sigmoid(self.model(normalized.unsqueeze(0)))[0].item())


class ActionFaultShield:
    """Blocks applicable actions predicted bad by oracle-fault classifiers."""

    def __init__(
        self,
        classifiers: list[LoadedActionClassifier],
        device: torch.device,
        cache_predictions: bool = True,
        shared_model: MultiTaskActionFaultClassifier | None = None,
        shared_mean: torch.Tensor | None = None,
        shared_scale: torch.Tensor | None = None,
    ):
        self.classifiers = classifiers
        self.device = device
        self.cache_predictions = cache_predictions
        self.shared_model = shared_model
        self.shared_mean = shared_mean
        self.shared_scale = shared_scale
        self._cache: dict[bytes, np.ndarray] = {}

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: str | torch.device = "cpu",
        cache_predictions: bool = True,
    ) -> "ActionFaultShield":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        torch_device = torch.device(device)
        classifiers: list[LoadedActionClassifier] = []

        if manifest.get("model_type") == "shared_multitask":
            checkpoint_path = directory / manifest["checkpoint"]
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=torch_device, weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=torch_device)
            model = MultiTaskActionFaultClassifier(
                input_dim=int(checkpoint["input_dim"]),
                hidden_dims=[int(value) for value in checkpoint["hidden_dims"]],
                dropout=float(checkpoint["dropout"]),
                n_actions=int(checkpoint["n_actions"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(torch_device)
            model.eval()
            for item in sorted(
                manifest["classifiers"], key=lambda value: int(value["action"])
            ):
                kind = str(item["kind"])
                if kind not in {"neural", "constant"}:
                    raise ValueError(f"Unknown multitask classifier kind {kind!r}")
                classifiers.append(
                    LoadedActionClassifier(
                        action=int(item["action"]),
                        action_name=str(item["action_name"]),
                        kind=kind,
                        threshold=float(item["threshold"]),
                        model=None,
                        mean=None,
                        scale=None,
                        constant_probability=float(item.get("constant_probability", 0.0)),
                    )
                )
            expected_actions = list(range(len(classifiers)))
            actual_actions = [classifier.action for classifier in classifiers]
            if actual_actions != expected_actions:
                raise ValueError(
                    f"Classifier actions must be contiguous: expected {expected_actions}, "
                    f"got {actual_actions}"
                )
            if int(checkpoint["n_actions"]) != len(classifiers):
                raise ValueError("Multitask checkpoint and manifest disagree on action count")
            return cls(
                classifiers,
                torch_device,
                cache_predictions,
                shared_model=model,
                shared_mean=torch.as_tensor(
                    checkpoint["normalization_mean"],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                shared_scale=torch.as_tensor(
                    checkpoint["normalization_scale"],
                    dtype=torch.float32,
                    device=torch_device,
                ),
            )

        for item in sorted(manifest["classifiers"], key=lambda value: int(value["action"])):
            checkpoint_path = directory / item["checkpoint"]
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=torch_device, weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=torch_device)

            kind = str(checkpoint["kind"])
            if kind == "constant":
                loaded = LoadedActionClassifier(
                    action=int(checkpoint["action"]),
                    action_name=str(checkpoint["action_name"]),
                    kind=kind,
                    threshold=float(checkpoint["threshold"]),
                    model=None,
                    mean=None,
                    scale=None,
                    constant_probability=float(checkpoint["constant_probability"]),
                )
            elif kind == "neural":
                model = ActionFaultClassifier(
                    input_dim=int(checkpoint["input_dim"]),
                    hidden_dims=[int(value) for value in checkpoint["hidden_dims"]],
                    dropout=float(checkpoint["dropout"]),
                )
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.to(torch_device)
                model.eval()
                loaded = LoadedActionClassifier(
                    action=int(checkpoint["action"]),
                    action_name=str(checkpoint["action_name"]),
                    kind=kind,
                    threshold=float(checkpoint["threshold"]),
                    model=model,
                    mean=torch.as_tensor(
                        checkpoint["normalization_mean"],
                        dtype=torch.float32,
                        device=torch_device,
                    ),
                    scale=torch.as_tensor(
                        checkpoint["normalization_scale"],
                        dtype=torch.float32,
                        device=torch_device,
                    ),
                )
            else:
                raise ValueError(f"Unknown classifier kind {kind!r} in {checkpoint_path}")
            classifiers.append(loaded)

        expected_actions = list(range(len(classifiers)))
        actual_actions = [classifier.action for classifier in classifiers]
        if actual_actions != expected_actions:
            raise ValueError(
                f"Classifier actions must be contiguous: expected {expected_actions}, "
                f"got {actual_actions}"
            )
        return cls(classifiers, torch_device, cache_predictions)

    def fault_probabilities(self, observation: np.ndarray) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32)
        key = observation_array.tobytes()
        if self.cache_predictions and key in self._cache:
            return self._cache[key].copy()

        observation_tensor = torch.as_tensor(
            observation_array, dtype=torch.float32, device=self.device
        )
        if self.shared_model is not None:
            if self.shared_mean is None or self.shared_scale is None:
                raise RuntimeError("Multitask classifier normalization is incomplete")
            normalized = (observation_tensor - self.shared_mean) / self.shared_scale
            with torch.no_grad():
                probabilities = (
                    torch.sigmoid(self.shared_model(normalized.unsqueeze(0)))[0]
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            for classifier in self.classifiers:
                if classifier.kind == "constant":
                    probabilities[classifier.action] = classifier.constant_probability
        else:
            probabilities = np.asarray(
                [
                    classifier.probability(observation_tensor)
                    for classifier in self.classifiers
                ],
                dtype=np.float32,
            )
        if self.cache_predictions:
            self._cache[key] = probabilities
        return probabilities.copy()

    def masks(
        self, observation: np.ndarray, applicable_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        applicable = np.asarray(applicable_mask, dtype=bool)
        if applicable.shape != (len(self.classifiers),):
            raise ValueError(
                f"Expected action mask shape {(len(self.classifiers),)}, "
                f"got {applicable.shape}"
            )
        probabilities = self.fault_probabilities(observation)
        predicted_faults = np.asarray(
            [
                probabilities[action] >= classifier.threshold
                for action, classifier in enumerate(self.classifiers)
            ],
            dtype=bool,
        )
        blocked = applicable & predicted_faults
        shielded = applicable & ~predicted_faults
        return shielded, blocked, probabilities

    def metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "action": classifier.action,
                "action_name": classifier.action_name,
                "kind": classifier.kind,
                "threshold": classifier.threshold,
            }
            for classifier in self.classifiers
        ]
