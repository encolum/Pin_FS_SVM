"""Cached, split-aware evaluation of frozen VeraPin policy candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import classification_metrics
from src.utils.serialization import read_json, write_json

from ..kernel_engine import run_kernel_search
from ..objectives import primal_integral
from ..policies.frozen_verapin import FrozenVeraPinPolicy
from ..progress import SolverProgressRecord, time_to_target_gap
from ..signals import LPRelaxationCache
from .schemas import PolicyCandidate


@dataclass(frozen=True)
class PolicyInstance:
    instance_id: str
    split: str
    X: np.ndarray
    y: np.ndarray
    B: int
    C: float
    tau: float
    coefficient_bounds: tuple[float, float]
    reference_objective: float | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    base_instance_id: str | None = None
    outer_fold: int | None = None

    @property
    def instance_hash(self) -> str:
        digest = sha256()
        digest.update(self.instance_id.encode("utf-8"))
        digest.update(self.split.encode("utf-8"))
        digest.update(np.ascontiguousarray(self.X).view(np.uint8))
        digest.update(np.ascontiguousarray(self.y).view(np.uint8))
        if self.X_test is not None:
            digest.update(np.ascontiguousarray(self.X_test).view(np.uint8))
        if self.y_test is not None:
            digest.update(np.ascontiguousarray(self.y_test).view(np.uint8))
        digest.update(
            repr(
                (
                    self.B,
                    self.C,
                    self.tau,
                    self.coefficient_bounds,
                    self.base_instance_id,
                    self.outer_fold,
                )
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class FitnessWeights:
    primal_integral: float
    final_gap: float
    failure_rate: float
    overhead: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in asdict(self).values())
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("fitness weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one fitness weight must be positive")


@dataclass(frozen=True)
class FitnessNormalization:
    primal_integral_scale: float
    final_gap_scale: float
    overhead_scale: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class PolicyEvaluation:
    policy_id: str
    mean_fitness: float
    mean_primal_integral: float
    mean_final_gap: float
    mean_time_to_target_gap: float | None
    failure_rate: float
    mean_overhead: float
    per_instance: list[dict]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PolicyEvaluation":
        return cls(**value)


@dataclass
class PolicyEvaluationCache:
    root: Path | None = None
    memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    lp_relaxations: LPRelaxationCache = field(default_factory=LPRelaxationCache)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = Path(self.root)
            self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        if key in self.memory:
            self.hits += 1
            return dict(self.memory[key])
        path = None if self.root is None else self.root / f"{key}.json"
        if path is not None and path.is_file():
            value = read_json(path)
            self.memory[key] = value
            self.hits += 1
            return dict(value)
        self.misses += 1
        return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.memory[key] = dict(value)
        if self.root is not None:
            write_json(self.root / f"{key}.json", value)


def evaluate_policy(
    candidate: PolicyCandidate,
    instances: list[PolicyInstance],
    *,
    required_split: str,
    solver_config: dict[str, Any],
    fitness_weights: FitnessWeights,
    normalization: FitnessNormalization,
    target_gap: float,
    cache: PolicyEvaluationCache | None = None,
) -> PolicyEvaluation:
    """Evaluate on exactly one declared split; lower fitness is better."""
    if not instances:
        raise ValueError("policy evaluation requires at least one instance")
    invalid = sorted({instance.split for instance in instances if instance.split != required_split})
    if invalid:
        raise ValueError(
            f"evaluation requested split {required_split!r} but received instances from {invalid}"
        )
    if required_split not in {"train", "validation", "test"}:
        raise ValueError("required_split must be train, validation, or test")
    target_gap = float(target_gap)
    if target_gap < 0:
        raise ValueError("target_gap must be non-negative")
    cache = cache or PolicyEvaluationCache()
    config_hash = _canonical_hash(solver_config)
    policy = FrozenVeraPinPolicy(candidate)
    rows: list[dict[str, Any]] = []

    for instance in instances:
        key = sha256(
            f"{candidate.policy_hash}:{instance.instance_hash}:{config_hash}".encode("utf-8")
        ).hexdigest()
        cached = cache.get(key)
        if cached is not None:
            rows.append(cached)
            continue
        try:
            result = run_kernel_search(
                instance.X,
                instance.y,
                policy=policy,
                B=instance.B,
                C=instance.C,
                tau=instance.tau,
                coefficient_bounds=instance.coefficient_bounds,
                lp_cache=cache.lp_relaxations,
                **solver_config,
            )
            trajectory = [
                SolverProgressRecord(**record)
                for record in result.metadata.get("route_progress", [])
            ]
            final_gap = trajectory[-1].relative_gap if trajectory else None
            overhead = float(result.metadata.get("signal_overhead", 0.0)) + float(
                result.metadata.get("policy_overhead", 0.0)
            ) + float(result.metadata.get("lp_relaxation_overhead", 0.0)) + float(
                result.metadata.get("mip_start_overhead", 0.0)
            )
            integral = primal_integral(
                trajectory,
                horizon=result.total_runtime,
                reference_objective=instance.reference_objective,
            )
            row = {
                "instance_id": instance.instance_id,
                "instance_hash": instance.instance_hash,
                "split": instance.split,
                "failed": False,
                "primal_integral": integral,
                "final_gap": 1.0 if final_gap is None else float(final_gap),
                "time_to_target_gap": time_to_target_gap(trajectory, target_gap),
                "overhead": overhead,
                "objective": result.best_result.objective,
                "total_runtime": result.total_runtime,
                "selected_feature_count": len(result.best_result.support),
                "selected_feature_indices": sorted(result.best_result.support),
            }
            if instance.X_test is not None and instance.y_test is not None:
                predictions = np.where(
                    instance.X_test @ result.best_result.coefficients
                    + result.best_result.intercept
                    >= 0,
                    1,
                    -1,
                )
                row.update(classification_metrics(instance.y_test, predictions))
                row["classification_scope"] = "outer_test"
        except Exception as exc:
            row = {
                "instance_id": instance.instance_id,
                "instance_hash": instance.instance_hash,
                "split": instance.split,
                "failed": True,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "primal_integral": normalization.primal_integral_scale,
                "final_gap": normalization.final_gap_scale,
                "time_to_target_gap": None,
                "overhead": normalization.overhead_scale,
            }
        cache.put(key, row)
        rows.append(row)

    failure_rate = float(np.mean([float(row["failed"]) for row in rows]))
    mean_integral = float(np.mean([float(row["primal_integral"]) for row in rows]))
    mean_gap = float(np.mean([float(row["final_gap"]) for row in rows]))
    mean_overhead = float(np.mean([float(row["overhead"]) for row in rows]))
    target_times = [
        float(row["time_to_target_gap"])
        for row in rows
        if row.get("time_to_target_gap") is not None
    ]
    mean_time = float(np.mean(target_times)) if target_times else None
    mean_fitness = (
        fitness_weights.primal_integral
        * mean_integral
        / normalization.primal_integral_scale
        + fitness_weights.final_gap * mean_gap / normalization.final_gap_scale
        + fitness_weights.failure_rate * failure_rate
        + fitness_weights.overhead * mean_overhead / normalization.overhead_scale
    )
    return PolicyEvaluation(
        policy_id=candidate.policy_id,
        mean_fitness=float(mean_fitness),
        mean_primal_integral=mean_integral,
        mean_final_gap=mean_gap,
        mean_time_to_target_gap=mean_time,
        failure_rate=failure_rate,
        mean_overhead=mean_overhead,
        per_instance=rows,
    )


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()
