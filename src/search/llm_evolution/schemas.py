"""Typed JSON contracts for safe VeraPin policy candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from numbers import Integral
from typing import Any


@dataclass(frozen=True)
class PolicyCandidate:
    policy_id: str
    name: str
    initial_kernel_size: int
    initial_score: dict[str, Any]
    add_score: dict[str, Any]
    keep_score: dict[str, Any]
    target_kernel_size: dict[str, Any]
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or int(self.schema_version) != 1
        ):
            raise ValueError("only VeraPin policy schema_version=1 is supported")
        if not isinstance(self.policy_id, str) or not isinstance(self.name, str):
            raise ValueError("policy_id and name must be strings")
        if not self.policy_id.strip() or not self.name.strip():
            raise ValueError("policy_id and name must not be empty")
        if (
            isinstance(self.initial_kernel_size, bool)
            or not isinstance(self.initial_kernel_size, Integral)
            or int(self.initial_kernel_size) < 1
        ):
            raise ValueError("initial_kernel_size must be a positive integer")
        for name in ("initial_score", "add_score", "keep_score", "target_kernel_size"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be a JSON expression object")
        if not isinstance(self.metadata, dict):
            raise ValueError("policy metadata must be a mapping")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PolicyCandidate":
        if not isinstance(value, dict):
            raise ValueError("policy candidate must be a JSON object")
        allowed = {
            "schema_version",
            "policy_id",
            "name",
            "initial_kernel_size",
            "initial_score",
            "add_score",
            "keep_score",
            "target_kernel_size",
            "metadata",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown policy fields: {unknown}")
        required = allowed - {"schema_version", "metadata"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"missing policy fields: {missing}")
        return cls(
            schema_version=value.get("schema_version", 1),
            policy_id=value["policy_id"],
            name=value["name"],
            initial_kernel_size=value["initial_kernel_size"],
            initial_score=value["initial_score"],
            add_score=value["add_score"],
            keep_score=value["keep_score"],
            target_kernel_size=value["target_kernel_size"],
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "name": self.name,
            "initial_kernel_size": self.initial_kernel_size,
            "initial_score": self.initial_score,
            "add_score": self.add_score,
            "keep_score": self.keep_score,
            "target_kernel_size": self.target_kernel_size,
            "metadata": self.metadata,
        }

    @property
    def policy_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()
