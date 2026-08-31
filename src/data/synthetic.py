"""Deterministic synthetic hardness instances for Pin-FS optimization studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.serialization import read_json, write_json


@dataclass(frozen=True)
class SyntheticParameters:
    n_samples: int
    n_features: int
    informative_ratio: float
    redundant_ratio: float
    correlation_strength: float
    positive_class_fraction: float
    label_noise_rate: float
    outlier_sample_rate: float
    outlier_feature_rate: float
    outlier_scale: float
    feature_budget_ratio: float
    seed: int
    split: str


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
    generation_mode: str = "legacy_embedded_corruption"

    def metadata(self) -> dict[str, Any]:
        return {
            "generation_mode": self.generation_mode,
            "research_split": self.parameters.split,
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
    label_noise_rate: float,
    outlier_sample_rate: float,
    outlier_feature_rate: float,
    outlier_scale: float,
    feature_budget_ratio: float,
    seed: int,
    split: str,
) -> SyntheticInstanceData:
    """Legacy generator including pre-split corruption; not for final scientific use."""
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
    label_noise_rate = _closed_rate(label_noise_rate, "label_noise_rate")
    outlier_sample_rate = _closed_rate(outlier_sample_rate, "outlier_sample_rate")
    outlier_feature_rate = _closed_rate(outlier_feature_rate, "outlier_feature_rate")
    outlier_scale = float(outlier_scale)
    if not np.isfinite(outlier_scale) or outlier_scale < 0:
        raise ValueError("outlier_scale must be finite and non-negative")
    feature_budget_ratio = _open_rate(feature_budget_ratio, "feature_budget_ratio")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")

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

    flip_count = int(round(n_samples * label_noise_rate))
    if flip_count:
        flipped = np.sort(rng.choice(n_samples, size=flip_count, replace=False))
        y[flipped] *= -1
    if set(np.unique(y)) != {-1, 1}:
        raise RuntimeError("requested label noise removed one of the two classes")

    outlier_samples = int(round(n_samples * outlier_sample_rate))
    outlier_features = int(round(n_features * outlier_feature_rate))
    if outlier_samples and outlier_features and outlier_scale:
        sample_indices = np.sort(
            rng.choice(n_samples, size=outlier_samples, replace=False)
        )
        feature_indices = np.sort(
            rng.choice(n_features, size=outlier_features, replace=False)
        )
        perturbation = rng.normal(
            scale=outlier_scale,
            size=(outlier_samples, outlier_features),
        )
        X_unshuffled[np.ix_(sample_indices, feature_indices)] += perturbation

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
        label_noise_rate=label_noise_rate,
        outlier_sample_rate=outlier_sample_rate,
        outlier_feature_rate=outlier_feature_rate,
        outlier_scale=outlier_scale,
        feature_budget_ratio=feature_budget_ratio,
        seed=int(seed),
        split=split,
    )
    data_hash = synthetic_hash(X, y, parameters)
    return SyntheticInstanceData(
        X=X,
        y=y,
        parameters=parameters,
        data_hash=data_hash,
        informative_indices=informative_indices,
        redundant_indices=redundant_indices,
        redundant_sources=redundant_sources,
        feature_budget=max(1, min(n_features, int(round(n_features * feature_budget_ratio)))),
    )


def generate_clean_synthetic_instance(*, n_samples, n_features, informative_ratio,
        redundant_ratio, correlation_strength, positive_class_fraction,
        feature_budget_ratio, seed, research_split):
    """Generate a clean base only; experimental corruption belongs after splitting."""
    result = generate_synthetic_instance(n_samples=n_samples, n_features=n_features,
        informative_ratio=informative_ratio, redundant_ratio=redundant_ratio,
        correlation_strength=correlation_strength, positive_class_fraction=positive_class_fraction,
        feature_budget_ratio=feature_budget_ratio, seed=seed, split=research_split,
        label_noise_rate=0., outlier_sample_rate=0., outlier_feature_rate=0., outlier_scale=0.)
    result.generation_mode = "clean_base"
    return result


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


def load_synthetic_instance(
    directory: str | Path,
    *,
    instance_id: str,
) -> SyntheticInstanceData:
    directory = Path(directory)
    metadata = read_json(directory / f"{instance_id}.json")
    with np.load(directory / f"{instance_id}.npz", allow_pickle=False) as arrays:
        X = np.asarray(arrays["X"], dtype=float)
        y = np.asarray(arrays["y"], dtype=int)
    parameters = SyntheticParameters(**metadata["parameters"])
    expected_hash = synthetic_hash(X, y, parameters)
    if expected_hash != metadata["data_hash"]:
        raise ValueError(f"synthetic instance {instance_id!r} failed its data-hash check")
    return SyntheticInstanceData(
        X=X,
        y=y,
        parameters=parameters,
        data_hash=expected_hash,
        informative_indices=[int(value) for value in metadata["informative_indices"]],
        redundant_indices=[int(value) for value in metadata["redundant_indices"]],
        redundant_sources={
            int(key): int(value) for key, value in metadata["redundant_sources"].items()
        },
        feature_budget=int(metadata["feature_budget"]),
        generation_mode=metadata.get("generation_mode", "legacy_embedded_corruption"),
    )


def synthetic_hash(
    X: np.ndarray,
    y: np.ndarray,
    parameters: SyntheticParameters,
) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(X).view(np.uint8))
    digest.update(np.ascontiguousarray(y).view(np.uint8))
    digest.update(repr(sorted(asdict(parameters).items())).encode("utf-8"))
    return digest.hexdigest()


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
