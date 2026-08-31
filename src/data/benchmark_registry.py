"""Strict, data-only registry for the six retained benchmarks."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import yaml

from .benchmark_loaders import BENCHMARK_LOADERS


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "benchmark_registry.yaml"
SPLIT_PARTITIONS = {"hill_valley": ("train", "test"), "madelon": ("train", "validation")}
PARTITION_POLICIES = {"pool", "merge_labeled", "official_holdout"}
PREPROCESSING_POLICIES = {"standard", "max_abs", "passthrough_upstream_normalized"}


class _RegistryLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            if key in result:
                raise ValueError(f"duplicate registry key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=True)
        except TypeError as exc:
            raise ValueError("registry keys must be scalar values") from exc
    return result


_RegistryLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def validate_partition_policy(dataset: str, policy: str) -> None:
    if not isinstance(policy, str) or policy not in PARTITION_POLICIES:
        raise ValueError(f"{dataset}: explicit partition policy must be one of {sorted(PARTITION_POLICIES)}")
    if dataset in SPLIT_PARTITIONS and policy == "pool":
        raise ValueError(f"{dataset}: choose merge_labeled or official_holdout explicitly")
    if dataset not in SPLIT_PARTITIONS and policy == "official_holdout":
        raise ValueError(f"{dataset}: official_holdout unavailable; no official split indices supplied")


def read_benchmark_registry(path=DEFAULT_REGISTRY_PATH) -> tuple[dict, str]:
    """Return a validated registry and the hash of exactly the bytes parsed."""
    content = Path(path).read_bytes()
    try:
        registry = yaml.load(content, Loader=_RegistryLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid benchmark registry YAML: {exc}") from exc
    if not isinstance(registry, dict) or set(registry) != set(BENCHMARK_LOADERS):
        raise ValueError("benchmark registry must define exactly the six retained benchmarks")
    fields = {"loader", "source_partition_policy", "label_mapping", "storage", "preprocessing", "expected_features"}
    for name, entry in registry.items():
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"{name}: registry entry requires exactly {sorted(fields)}")
        if entry["loader"] != name:
            raise ValueError(f"{name}: loader must name the same retained benchmark")
        validate_partition_policy(name, entry["source_partition_policy"])
        if entry["storage"] not in ("dense", "csr"):
            raise ValueError(f"{name}: storage must be dense or csr")
        if entry["preprocessing"] not in tuple(PREPROCESSING_POLICIES):
            raise ValueError(f"{name}: unknown preprocessing policy")
        if entry["storage"] == "csr" and entry["preprocessing"] == "standard":
            raise ValueError(f"{name}: centered standard scaling is not sparse-safe")
        if name == "colon" and entry["preprocessing"] != "passthrough_upstream_normalized":
            raise ValueError("colon: preserve the upstream-normalized preprocessing declaration")
        if type(entry["expected_features"]) is not int or entry["expected_features"] <= 0:
            raise ValueError(f"{name}: expected_features must be a positive integer")
        mapping = entry["label_mapping"]
        if (not isinstance(mapping, dict) or len(mapping) != 2
                or any(type(value) is not int or value not in (-1, 1) for value in mapping.values())
                or set(mapping.values()) != {-1, 1}):
            raise ValueError(f"{name}: label_mapping must explicitly map two native labels to -1/+1")
        key_type = str if name in {"gina", "hiva"} else int
        if any(type(key) is not key_type for key in mapping):
            raise ValueError(f"{name}: native mapping keys must be {key_type.__name__}")
    return registry, sha256(content).hexdigest()
