"""Deterministic prompts that expose training summaries but never held-out results."""

from __future__ import annotations

import json
from typing import Any

from .sandbox import ALLOWED_OPERATIONS, FEATURE_SIGNALS, SEARCH_SIGNALS


def build_evolution_prompt(
    *,
    generation: int,
    training_summary: list[dict[str, Any]],
    parent_policies: list[dict[str, Any]],
    failure_summary: list[dict[str, Any]],
    requested_candidates: int,
) -> str:
    if int(requested_candidates) < 1:
        raise ValueError("requested_candidates must be positive")
    envelope = {
        "task": "Generate deterministic VeraPin kernel-policy mutations as typed JSON only.",
        "generation": int(generation),
        "requested_candidates": int(requested_candidates),
        "scientific_constraints": [
            "The policy controls kernel ranking and size only.",
            "It must not modify the Pin-FS objective or constraints.",
            "It receives training/search signals only and no held-out results.",
            "Return a JSON object with a candidates list and no executable code.",
        ],
        "dsl": {
            "allowed_operations": sorted(ALLOWED_OPERATIONS),
            "feature_signals": sorted(FEATURE_SIGNALS),
            "search_signals": sorted(SEARCH_SIGNALS),
            "candidate_fields": [
                "schema_version",
                "policy_id",
                "name",
                "initial_kernel_size",
                "initial_score",
                "add_score",
                "keep_score",
                "target_kernel_size",
                "metadata",
            ],
        },
        "training_evaluations": training_summary,
        "parents": parent_policies,
        "failures": failure_summary,
    }
    return json.dumps(envelope, sort_keys=True, indent=2)
