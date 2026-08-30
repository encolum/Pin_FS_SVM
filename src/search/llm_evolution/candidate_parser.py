"""Strict parsing and validation of LLM-produced policy candidates."""

from __future__ import annotations

import json
import math
from typing import Any

from .sandbox import compile_expression
from .schemas import PolicyCandidate
from ..states import SearchState


class CandidateValidationError(ValueError):
    pass


def parse_candidates(text: str) -> list[PolicyCandidate]:
    """Parse one candidate, a list, or ``{"candidates": [...]}`` from JSON."""
    payload = _strip_json_fence(str(text).strip())
    try:
        value = json.loads(payload, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise CandidateValidationError(f"invalid candidate JSON: {exc}") from exc
    if isinstance(value, dict) and set(value) == {"candidates"}:
        value = value["candidates"]
    raw_candidates = value if isinstance(value, list) else [value]
    if not raw_candidates:
        raise CandidateValidationError("candidate list must not be empty")
    candidates = []
    for index, raw in enumerate(raw_candidates):
        try:
            candidate = validate_candidate(PolicyCandidate.from_dict(raw))
        except (TypeError, ValueError) as exc:
            raise CandidateValidationError(f"candidate {index} rejected: {exc}") from exc
        candidates.append(candidate)
    hashes = [candidate.policy_hash for candidate in candidates]
    if len(hashes) != len(set(hashes)):
        raise CandidateValidationError("candidate response contains duplicate policies")
    return candidates


def validate_candidate(candidate: PolicyCandidate) -> PolicyCandidate:
    compile_expression(candidate.initial_score)
    compile_expression(candidate.add_score)
    compile_expression(candidate.keep_score)
    target = compile_expression(
        candidate.target_kernel_size, allow_feature_signals=False
    )
    _validate_target_behavior(target)
    return candidate


def _strip_json_fence(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1])
    return text


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _validate_target_behavior(target) -> None:
    for total_features, feature_budget, kernel_size, stagnation, improved in (
        (10, 2, 4, 0, True),
        (100, 10, 25, 4, False),
    ):
        search = SearchState(
            iteration=2,
            current_objective=12.0,
            best_objective=10.0,
            current_gap=0.2,
            best_bound=8.0,
            kernel_size=kernel_size,
            feature_budget=feature_budget,
            total_features=total_features,
            stagnation_iterations=stagnation,
            elapsed_seconds=2.0,
            remaining_seconds=8.0,
            C=1.0,
            tau=0.5,
            improved_last_iteration=improved,
        )
        size = int(math.ceil(target(None, search)))
        if size < feature_budget or size > total_features:
            raise ValueError(
                "target_kernel_size must remain within [feature_budget, total_features]"
            )
