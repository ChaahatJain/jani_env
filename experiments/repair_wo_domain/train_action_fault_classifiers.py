#!/usr/bin/env python3
"""Train a shared fault-classification backbone with one head per action."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.repair_wo_domain.action_fault_shield import (
    MultiTaskActionFaultClassifier,
    summarize_per_action_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multitask state encoder with one fault head per action."
    )
    parser.add_argument("--collection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--target-validation-recall",
        type=float,
        default=0.99,
        help="Choose the most precise threshold attaining at least this recall.",
    )
    parser.add_argument(
        "--max-negative-examples-per-action",
        type=int,
        default=250_000,
        help="Uniformly sample at most this many unique non-fault states per action.",
    )
    parser.add_argument(
        "--minimum-positive-examples",
        type=int,
        default=10,
        help="Fail rather than fit an unreliable neural classifier below this count.",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def unique_rows(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    return np.unique(np.ascontiguousarray(values, dtype=np.float32), axis=0)


def row_keys(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def load_action_dataset(
    manifest: dict[str, Any],
    action: int,
    max_negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    shards = manifest["shards"]
    raw_negative_counts = [
        int(shard["actions"][str(action)]["non_faults"]) for shard in shards
    ]
    total_raw_negatives = sum(raw_negative_counts)
    positive_chunks: list[np.ndarray] = []
    negative_chunks: list[np.ndarray] = []

    for shard_position, (shard, raw_negative_count) in enumerate(
        zip(shards, raw_negative_counts)
    ):
        action_meta = shard["actions"][str(action)]
        path = Path(shard["directory"]) / action_meta["file"]
        with np.load(path) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.uint8)
            stored_action = int(np.asarray(data["action"]).item())
        if stored_action != action:
            raise ValueError(f"{path} stores action {stored_action}, expected {action}")
        if observations.ndim != 2 or labels.shape != (len(observations),):
            raise ValueError(f"Malformed action label arrays in {path}")

        positive_chunks.append(observations[labels == 1])
        negatives = observations[labels == 0]
        if max_negatives > 0 and total_raw_negatives > max_negatives:
            proportional = max_negatives * raw_negative_count / total_raw_negatives
            quota = min(len(negatives), max(1, int(math.ceil(proportional * 1.25)) + 50))
            if len(negatives) > quota:
                rng = np.random.default_rng(seed + action * 100_003 + shard_position)
                negatives = negatives[rng.choice(len(negatives), size=quota, replace=False)]
        negative_chunks.append(negatives)

    obs_dim = int(manifest["observation_dim"])
    positives = unique_rows(
        np.concatenate(positive_chunks, axis=0)
        if positive_chunks
        else np.empty((0, obs_dim), dtype=np.float32)
    )
    negatives = unique_rows(
        np.concatenate(negative_chunks, axis=0)
        if negative_chunks
        else np.empty((0, obs_dim), dtype=np.float32)
    )

    if len(positives) and len(negatives):
        conflicts = np.isin(row_keys(negatives), row_keys(positives))
        if conflicts.any():
            raise ValueError(
                f"Oracle labels disagree across shards for action {action} on "
                f"{int(conflicts.sum())} sampled states"
            )
    if max_negatives > 0 and len(negatives) > max_negatives:
        rng = np.random.default_rng(seed + action * 1_000_003)
        negatives = negatives[
            rng.choice(len(negatives), size=max_negatives, replace=False)
        ]

    features = np.concatenate([positives, negatives], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(positives), dtype=np.uint8),
            np.zeros(len(negatives), dtype=np.uint8),
        ]
    )
    return features, labels, {
        "unique_faults": int(len(positives)),
        "sampled_unique_non_faults": int(len(negatives)),
        "raw_non_fault_examples_before_cross_shard_deduplication": int(
            total_raw_negatives
        ),
    }


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, ...]:
    holdout_fraction = validation_fraction + test_fraction
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features,
        labels,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=labels,
    )
    relative_test = test_fraction / holdout_fraction
    val_x, test_x, val_y, test_y = train_test_split(
        holdout_x,
        holdout_y,
        test_size=relative_test,
        random_state=seed + 1,
        stratify=holdout_y,
    )
    return train_x, val_x, test_x, train_y, val_y, test_y


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "examples": int(len(labels)),
        "faults": int(labels.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, target_recall: float
) -> tuple[float, dict[str, Any]]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        threshold = 0.5
    else:
        eligible = np.flatnonzero(recall[:-1] >= target_recall)
        if len(eligible) == 0:
            threshold = 0.0
        else:
            eligible_precision = precision[:-1][eligible]
            best_precision = eligible_precision.max()
            best = eligible[eligible_precision == best_precision]
            threshold = float(thresholds[best[-1]])
    return threshold, classification_metrics(labels, probabilities, threshold)


def predict_probabilities(
    model: MultiTaskActionFaultClassifier,
    action: int,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = (features[start : start + batch_size] - mean) / scale
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
            output.append(torch.sigmoid(model(tensor)[:, action]).cpu().numpy())
    return np.concatenate(output) if output else np.empty((0,), dtype=np.float32)


def train_multitask_classifier(
    datasets: list[dict[str, Any]],
    n_actions: int,
    input_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    """Jointly optimize all non-constant action tasks through one backbone."""
    active = [dataset for dataset in datasets if dataset["kind"] == "neural"]
    if not active:
        all_features = np.concatenate([dataset["features"] for dataset in datasets])
        if len(all_features):
            mean = all_features.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = all_features.std(axis=0, dtype=np.float64).astype(np.float32)
        else:
            mean = np.zeros(input_dim, dtype=np.float32)
            scale = np.ones(input_dim, dtype=np.float32)
        scale[scale < 1e-6] = 1.0
        model = MultiTaskActionFaultClassifier(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            n_actions=n_actions,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        return (
            {
                "format_version": 2,
                "kind": "shared_multitask",
                "input_dim": int(input_dim),
                "hidden_dims": [int(value) for value in args.hidden_dims],
                "dropout": float(args.dropout),
                "n_actions": int(n_actions),
                "model_state_dict": model.state_dict(),
                "normalization_mean": mean,
                "normalization_scale": scale,
            },
            {},
            {
                "best_epoch": 0,
                "best_validation_macro_average_precision": None,
                "history": [],
                "reason": "No action had observed faults; all heads are constant.",
            },
        )

    for dataset in active:
        action = int(dataset["action"])
        dataset["split"] = split_dataset(
            dataset["features"],
            dataset["labels"],
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed + action * 101,
        )

    all_train_x = np.concatenate([dataset["split"][0] for dataset in active])
    mean = all_train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = all_train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    normalized_train = np.concatenate(
        [(dataset["split"][0] - mean) / scale for dataset in active]
    )
    train_actions = np.concatenate(
        [
            np.full(len(dataset["split"][3]), dataset["action"], dtype=np.int64)
            for dataset in active
        ]
    )
    train_labels = np.concatenate([dataset["split"][3] for dataset in active])
    dataset = TensorDataset(
        torch.as_tensor(normalized_train, dtype=torch.float32),
        torch.as_tensor(train_actions, dtype=torch.int64),
        torch.as_tensor(train_labels, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(args.seed)
    action_counts = np.bincount(train_actions, minlength=n_actions)
    sample_weights = np.asarray(
        [1.0 / action_counts[action] for action in train_actions], dtype=np.float64
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    model = MultiTaskActionFaultClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        n_actions=n_actions,
    ).to(device)
    positive_weights = torch.ones(n_actions, dtype=torch.float32, device=device)
    for item in active:
        action = int(item["action"])
        train_y = item["split"][3]
        positive_weights[action] = float(
            (train_y == 0).sum() / max(1, (train_y == 1).sum())
        )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_average_precision = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_x, batch_actions, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_actions = batch_actions.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            losses = criterion(logits, batch_y)
            losses = losses * torch.where(
                batch_y > 0.5, positive_weights[batch_actions], 1.0
            )
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_examples += len(batch_x)

        per_action_ap: dict[str, float] = {}
        for item in active:
            action = int(item["action"])
            val_x, val_y = item["split"][1], item["split"][4]
            val_probabilities = predict_probabilities(
                model, action, val_x, mean, scale, device, args.batch_size
            )
            per_action_ap[str(action)] = float(
                average_precision_score(val_y, val_probabilities)
            )
        val_ap = float(np.mean(list(per_action_ap.values())))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, total_examples),
                "validation_macro_average_precision": val_ap,
                "validation_average_precision_by_action": per_action_ap,
            }
        )
        if val_ap > best_average_precision + 1e-8:
            best_average_precision = val_ap
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Multitask training produced no checkpoint")
    model.load_state_dict(best_state)

    results: dict[int, dict[str, Any]] = {}
    for item in active:
        action = int(item["action"])
        train_x, val_x, test_x, train_y, val_y, test_y = item["split"]
        val_probabilities = predict_probabilities(
            model, action, val_x, mean, scale, device, args.batch_size
        )
        threshold, validation_metrics = select_threshold(
            val_y, val_probabilities, args.target_validation_recall
        )
        test_probabilities = predict_probabilities(
            model, action, test_x, mean, scale, device, args.batch_size
        )
        results[action] = {
            "threshold": threshold,
            "split_counts": {
                "train": int(len(train_y)),
                "validation": int(len(val_y)),
                "test": int(len(test_y)),
                "train_faults": int(train_y.sum()),
                "validation_faults": int(val_y.sum()),
                "test_faults": int(test_y.sum()),
            },
            "positive_weight": float(positive_weights[action].item()),
            "validation_metrics": validation_metrics,
            "test_metrics": classification_metrics(
                test_y, test_probabilities, threshold
            ),
        }

    checkpoint = {
        "format_version": 2,
        "kind": "shared_multitask",
        "input_dim": int(input_dim),
        "hidden_dims": [int(value) for value in args.hidden_dims],
        "dropout": float(args.dropout),
        "n_actions": int(n_actions),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "normalization_mean": mean,
        "normalization_scale": scale,
    }
    training_summary = {
        "best_epoch": best_epoch,
        "best_validation_macro_average_precision": best_average_precision,
        "history": history,
    }
    return checkpoint, results, training_summary


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("test-fraction must be between zero and one")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation-fraction + test-fraction must be below one")
    if not 0.0 < args.target_validation_recall <= 1.0:
        raise ValueError("target-validation-recall must be in (0, 1]")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_dir = Path(args.collection_dir)
    manifest = json.loads(
        (collection_dir / "action_labels_manifest.json").read_text(encoding="utf-8")
    )

    action_names = [str(value) for value in manifest["action_names"]]
    datasets: list[dict[str, Any]] = []
    for action, action_name in enumerate(action_names):
        print(f"Loading labels for action {action}: {action_name}", flush=True)
        features, labels, dataset_counts = load_action_dataset(
            manifest,
            action=action,
            max_negatives=args.max_negative_examples_per_action,
            seed=args.seed,
        )
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        print(
            f"  unique faults={positives}, sampled unique non-faults={negatives}",
            flush=True,
        )
        if 0 < positives < args.minimum_positive_examples:
            raise RuntimeError(
                f"Action {action} ({action_name}) has only {positives} unique "
                f"faults; at least {args.minimum_positive_examples} are required."
            )
        if positives > 0 and negatives == 0:
            raise RuntimeError(
                f"Action {action} ({action_name}) has no non-fault examples."
            )
        datasets.append(
            {
                "action": action,
                "action_name": action_name,
                "kind": "neural" if positives else "constant",
                "features": features,
                "labels": labels,
                "dataset_counts": dataset_counts,
            }
        )

    checkpoint, neural_results, training_summary = train_multitask_classifier(
        datasets=datasets,
        n_actions=len(action_names),
        input_dim=int(manifest["observation_dim"]),
        args=args,
        device=device,
    )
    checkpoint_name = "multitask_model.pth"
    temporary = output_dir / f"{checkpoint_name}.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output_dir / checkpoint_name)

    classifiers: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for dataset in datasets:
        action = int(dataset["action"])
        if dataset["kind"] == "constant":
            reason = "No oracle-labelled faults were observed for this action."
            classifier = {
                "action": action,
                "action_name": dataset["action_name"],
                "kind": "constant",
                "constant_probability": 0.0,
                "threshold": 0.5,
            }
            report = {
                "kind": "constant",
                "action": action,
                "action_name": dataset["action_name"],
                "reason": reason,
            }
        else:
            result = neural_results[action]
            classifier = {
                "action": action,
                "action_name": dataset["action_name"],
                "kind": "neural",
                "threshold": float(result["threshold"]),
            }
            report = {
                "kind": "neural",
                "action": action,
                "action_name": dataset["action_name"],
                "split_counts": result["split_counts"],
                "positive_weight": result["positive_weight"],
                "target_validation_recall": float(args.target_validation_recall),
                "validation_metrics": result["validation_metrics"],
                "test_metrics": result["test_metrics"],
                "shared_training": {
                    "best_epoch": training_summary["best_epoch"],
                    "best_validation_macro_average_precision": training_summary[
                        "best_validation_macro_average_precision"
                    ],
                },
            }
        report["dataset_counts"] = dataset["dataset_counts"]
        write_json(output_dir / f"action_{action}_metrics.json", report)
        classifiers.append(classifier)
        reports.append(report)

    write_json(output_dir / "multitask_training_metrics.json", training_summary)

    output_manifest = {
        "format_version": 2,
        "model_type": "shared_multitask",
        "checkpoint": checkpoint_name,
        "classifier_definition": (
            "A shared state encoder feeds one binary output head per action. "
            "The positive class is oracle-fault/bad for that (state, action) pair."
        ),
        "training_collection": str(collection_dir.resolve()),
        "source_policy_sha256": manifest["policy_sha256"],
        "observation_dim": int(manifest["observation_dim"]),
        "action_names": action_names,
        "classifiers": classifiers,
        "per_action_evaluation_definition": (
            "A held-out oracle-labelled fault is fixed when its action head predicts "
            "fault/bad, so the shield would block that action. Held-out non-fault "
            "actions predicted as fault are false blocks."
        ),
        "per_action_evaluation": summarize_per_action_evaluation(reports),
        "training_arguments": vars(args),
        "reports": reports,
    }
    write_json(output_dir / "manifest.json", output_manifest)
    print(
        f"Saved one shared model with {len(classifiers)} action heads to "
        f"{output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
