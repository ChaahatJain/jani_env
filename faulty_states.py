import json

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FAULTY_STATE_FORMAT_VERSION = 1


def load_faulty_states(path: str | Path) -> np.ndarray:
    """Load a faulty-state pool from the JSON format written by the repair pipeline."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    states = payload.get("states") if isinstance(payload, dict) else payload
    if not isinstance(states, list) or not states:
        raise ValueError(f"Faulty-state file contains no states: {source}")

    array = np.asarray(states, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"Faulty states must be a non-empty 2D array: {source}")
    if not np.isfinite(array).all():
        raise ValueError(f"Faulty states must contain only finite values: {source}")
    return array


def save_faulty_states(
    path: str | Path,
    states: Iterable[Iterable[float]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist restart states in a stable, inspectable JSON format."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized_states = [np.asarray(state, dtype=np.float64).tolist() for state in states]
    payload = {
        "format_version": FAULTY_STATE_FORMAT_VERSION,
        "states": serialized_states,
        "metadata": metadata or {},
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    temporary.replace(destination)


def restart_states_from_trace(trace: dict[str, Any], history: int) -> list[np.ndarray]:
    """Return states closest to a failure or detected cycle, excluding failure terminals."""
    reason = trace.get("termination_reason")
    if reason not in {"failure", "cycle"}:
        return []
    if history <= 0:
        raise ValueError("history must be positive")

    candidates = list(trace.get("observations", []))
    if reason == "cycle" and trace.get("final_observation") is not None:
        candidates.append(trace["final_observation"])
    return candidates[-history:]


class RecentFaultyStatePool:
    """A bounded, de-duplicated pool that favors states from recent bad rollouts."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._states: OrderedDict[tuple[float, ...], list[float]] = OrderedDict()

    def add(self, states: Iterable[Iterable[float]]) -> int:
        added = 0
        for state in states:
            values = np.asarray(state, dtype=np.float64)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError("Faulty restart states must be finite one-dimensional vectors")
            key = tuple(values.tolist())
            if key in self._states:
                self._states.move_to_end(key)
                continue
            self._states[key] = values.tolist()
            added += 1
            if len(self._states) > self._max_size:
                self._states.popitem(last=False)
        return added

    def __len__(self) -> int:
        return len(self._states)

    def states(self) -> list[list[float]]:
        return list(self._states.values())
