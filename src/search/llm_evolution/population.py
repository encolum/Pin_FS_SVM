"""Fitness-first population updates with deterministic structural diversity."""

from __future__ import annotations

from dataclasses import dataclass
import json

from .evaluator import PolicyEvaluation
from .schemas import PolicyCandidate


@dataclass
class PopulationMember:
    candidate: PolicyCandidate
    training_evaluation: PolicyEvaluation


def select_strong_diverse(
    members: list[PopulationMember],
    *,
    count: int,
    maximum_similarity: float,
) -> list[PopulationMember]:
    if int(count) < 1:
        raise ValueError("selection count must be positive")
    if not 0 <= float(maximum_similarity) <= 1:
        raise ValueError("maximum_similarity must lie in [0, 1]")
    ranked = sorted(
        members,
        key=lambda member: (
            member.training_evaluation.mean_fitness,
            member.candidate.policy_hash,
        ),
    )
    selected: list[PopulationMember] = []
    deferred: list[PopulationMember] = []
    for member in ranked:
        if len(selected) >= int(count):
            break
        tokens = _policy_tokens(member.candidate)
        if all(
            _jaccard(tokens, _policy_tokens(existing.candidate)) <= maximum_similarity
            for existing in selected
        ):
            selected.append(member)
        else:
            deferred.append(member)
    for member in deferred:
        if len(selected) >= int(count):
            break
        selected.append(member)
    return selected


def update_population(
    existing: list[PopulationMember],
    candidates: list[PopulationMember],
    *,
    population_size: int,
    maximum_similarity: float,
) -> list[PopulationMember]:
    unique: dict[str, PopulationMember] = {}
    for member in existing + candidates:
        key = member.candidate.policy_hash
        incumbent = unique.get(key)
        if incumbent is None or (
            member.training_evaluation.mean_fitness
            < incumbent.training_evaluation.mean_fitness
        ):
            unique[key] = member
    return select_strong_diverse(
        list(unique.values()),
        count=min(int(population_size), len(unique)),
        maximum_similarity=maximum_similarity,
    )


def _policy_tokens(candidate: PolicyCandidate) -> set[str]:
    payload = json.dumps(
        {
            "initial_kernel_size": candidate.initial_kernel_size,
            "initial_score": candidate.initial_score,
            "add_score": candidate.add_score,
            "keep_score": candidate.keep_score,
            "target_kernel_size": candidate.target_kernel_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {payload[index : index + 12] for index in range(max(1, len(payload) - 11))}


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)
