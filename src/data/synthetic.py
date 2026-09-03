"""Deterministic synthetic hardness instances for Pin-FS optimization studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.matrices import data_hash
from src.utils.serialization import write_json


@dataclass(frozen=True)
class SyntheticParameters:
    n_samples: int
    n_features: int
    informative_ratio: float
    redundant_ratio: float
    correlation_strength: float
    positive_class_fraction: float
    feature_budget_ratio: float
    seed: int


@dataclass
class SyntheticInstanceData:
    X: np.ndarray
    y: np.ndarray
    parameters: SyntheticParameters
    data_hash: str
    informative_indices: list[int]
    redundant_indices: list[int]
    redundant_sources: dict[int, int]
    feature_budget: int

    def metadata(self) -> dict[str, Any]:
        return {
            "parameters": asdict(self.parameters),
            "data_hash": self.data_hash,
            "informative_indices": self.informative_indices,
            "redundant_indices": self.redundant_indices,
            "redundant_sources": {
                str(key): value for key, value in sorted(self.redundant_sources.items())
            },
            "feature_budget": self.feature_budget,
            "shape": [int(self.X.shape[0]), int(self.X.shape[1])],
            "class_counts": {
                "-1": int(np.count_nonzero(self.y == -1)),
                "+1": int(np.count_nonzero(self.y == 1)),
            },
        }


def generate_synthetic_instance(
    *,
    n_samples: int,
    n_features: int,
    informative_ratio: float,
    redundant_ratio: float,
    correlation_strength: float,
    positive_class_fraction: float,
    feature_budget_ratio: float,
    seed: int,
) -> SyntheticInstanceData:
    """Generate a clean synthetic base; corruption is applied after splitting."""
    n_samples = _integer_at_least(n_samples, "n_samples", 4)
    n_features = _integer_at_least(n_features, "n_features", 2)
    informative_ratio = _closed_rate(informative_ratio, "informative_ratio")
    redundant_ratio = _closed_rate(redundant_ratio, "redundant_ratio")
    if informative_ratio <= 0:
        raise ValueError("informative_ratio must be positive")
    if informative_ratio + redundant_ratio > 1:
        raise ValueError("informative_ratio + redundant_ratio cannot exceed 1")
    correlation_strength = _closed_rate(correlation_strength, "correlation_strength")
    positive_class_fraction = _open_rate(
        positive_class_fraction, "positive_class_fraction"
    )
    feature_budget_ratio = _open_rate(feature_budget_ratio, "feature_budget_ratio")
    n_informative = max(1, int(round(n_features * informative_ratio)))
    n_redundant = int(round(n_features * redundant_ratio))
    n_redundant = min(n_redundant, n_features - n_informative)
    n_noise = n_features - n_informative - n_redundant
    rng = np.random.default_rng(int(seed))

    informative = rng.normal(size=(n_samples, n_informative))
    weights = rng.normal(size=n_informative)
    norm = float(np.linalg.norm(weights))
    if norm <= 1e-15:
        weights[0] = 1.0
        norm = 1.0
    scores = informative @ (weights / norm) + rng.normal(scale=0.25, size=n_samples)
    positive_count = min(
        n_samples - 1,
        max(1, int(round(n_samples * positive_class_fraction))),
    )
    ranked = np.lexsort((np.arange(n_samples), -scores))
    y = np.full(n_samples, -1, dtype=int)
    y[ranked[:positive_count]] = 1

    redundant = np.empty((n_samples, n_redundant), dtype=float)
    sources: list[int] = []
    residual_scale = float(np.sqrt(max(0.0, 1.0 - correlation_strength**2)))
    for index in range(n_redundant):
        source = index % n_informative
        sources.append(source)
        redundant[:, index] = (
            correlation_strength * informative[:, source]
            + residual_scale * rng.normal(size=n_samples)
        )
    noise = rng.normal(size=(n_samples, n_noise))
    X_unshuffled = np.column_stack([informative, redundant, noise])

    permutation = rng.permutation(n_features)
    X = np.asarray(X_unshuffled[:, permutation], dtype=float)
    old_to_new = {int(old): int(new) for new, old in enumerate(permutation)}
    informative_indices = sorted(old_to_new[index] for index in range(n_informative))
    redundant_indices = sorted(
        old_to_new[n_informative + index] for index in range(n_redundant)
    )
    redundant_sources = {
        old_to_new[n_informative + index]: old_to_new[source]
        for index, source in enumerate(sources)
    }
    parameters = SyntheticParameters(
        n_samples=n_samples,
        n_features=n_features,
        informative_ratio=informative_ratio,
        redundant_ratio=redundant_ratio,
        correlation_strength=correlation_strength,
        positive_class_fraction=positive_class_fraction,
        feature_budget_ratio=feature_budget_ratio,
        seed=int(seed),
    )
    return SyntheticInstanceData(
        X=X,
        y=y,
        parameters=parameters,
        data_hash=data_hash(X, y),
        informative_indices=informative_indices,
        redundant_indices=redundant_indices,
        redundant_sources=redundant_sources,
        feature_budget=max(1, min(n_features, int(round(n_features * feature_budget_ratio)))),
    )

def save_synthetic_instance(
    instance: SyntheticInstanceData,
    directory: str | Path,
    *,
    instance_id: str,
) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / f"{instance_id}.npz"
    metadata_path = directory / f"{instance_id}.json"
    np.savez_compressed(array_path, X=instance.X, y=instance.y)
    write_json(metadata_path, {"instance_id": instance_id, **instance.metadata()})
    return array_path, metadata_path

def _integer_at_least(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _closed_rate(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _open_rate(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return value
