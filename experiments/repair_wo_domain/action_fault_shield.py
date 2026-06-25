"""Per-action fault classifiers and runtime action shielding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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
    """Combines environment applicability with per-action fault predictions."""

    def __init__(
        self,
        classifiers: list[LoadedActionClassifier],
        device: torch.device,
        cache_predictions: bool = True,
    ):
        self.classifiers = classifiers
        self.device = device
        self.cache_predictions = cache_predictions
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
