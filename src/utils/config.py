"""Configuration loading shared by the VeraPin command-line workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML, falling back to PyYAML for YAML syntax."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("non-JSON YAML requires PyYAML") from exc
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("the configuration root must be a mapping")
    config["_config_path"] = str(path.resolve())
    return config
