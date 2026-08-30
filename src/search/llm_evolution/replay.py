"""Load immutable evolution records for deterministic offline replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.serialization import read_json

from .provider import ReplayProvider
from .schemas import PolicyCandidate


def load_replay_provider(run_dir: str | Path) -> ReplayProvider:
    path = Path(run_dir) / "provider_records.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError("provider_records.json must contain a list")
    return ReplayProvider(records)


def load_frozen_policy(run_dir: str | Path) -> PolicyCandidate:
    path = Path(run_dir) / "policies" / "frozen_verapin_policy.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return PolicyCandidate.from_dict(read_json(path))


def evolution_audit(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    checkpoint = read_json(run_dir / "checkpoint.json")
    frozen = load_frozen_policy(run_dir)
    return {
        "generation": int(checkpoint["generation"]),
        "population_size": len(checkpoint["population"]),
        "provider_calls": len(checkpoint.get("provider_records", [])),
        "frozen_policy_id": frozen.policy_id,
        "selected_on": frozen.metadata.get("selected_on"),
    }
