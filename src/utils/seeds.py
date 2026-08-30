"""Stable derived seeds independent of Python hash randomization."""

from __future__ import annotations

from hashlib import sha256


def derive_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big")
