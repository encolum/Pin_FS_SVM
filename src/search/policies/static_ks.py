"""Deterministic one-ranking Static Kernel Search policy."""

from __future__ import annotations

import math

from ..states import FeatureState, SearchState


_STATIC_SIGNALS = {
    "fisher_score",
    "mutual_information",
    "mean_abs_correlation",
    "max_abs_correlation",
    "lp_activation",
    "lp_abs_coefficient",
}


class StaticKSPolicy:
    """Rank once, then expose deterministic prefixes as fixed buckets."""

    name = "static_ks"

    def __init__(
        self,
        *,
        score_name: str,
        initial_kernel_size: int,
        bucket_size: int,
        maximum_kernel_size: int | None = None,
    ) -> None:
        if score_name not in _STATIC_SIGNALS:
            raise ValueError(f"unsupported static score: {score_name}")
        if int(initial_kernel_size) < 1 or int(bucket_size) < 1:
            raise ValueError("initial_kernel_size and bucket_size must be positive")
        if maximum_kernel_size is not None and int(maximum_kernel_size) < 1:
            raise ValueError("maximum_kernel_size must be positive when provided")
        self.score_name = score_name
        self.initial_kernel_size = int(initial_kernel_size)
        self.bucket_size = int(bucket_size)
        self.maximum_kernel_size = (
            None if maximum_kernel_size is None else int(maximum_kernel_size)
        )
        self._rank: dict[int, int] = {}

    def stable_order(self, features: list[FeatureState]) -> list[int]:
        return [
            feature.index
            for feature in sorted(
                features,
                key=lambda feature: (-float(getattr(feature, self.score_name)), feature.index),
            )
        ]

    def initialize_kernel(
        self,
        features: list[FeatureState],
        search: SearchState,
    ) -> set[int]:
        order = self.stable_order(features)
        self._rank = {feature: rank for rank, feature in enumerate(order)}
        size = max(search.feature_budget, self.initial_kernel_size)
        size = min(search.total_features, size)
        if self.maximum_kernel_size is not None:
            size = min(size, max(search.feature_budget, self.maximum_kernel_size))
        return set(order[:size])

    def add_score(self, feature: FeatureState, search: SearchState) -> float:
        return self._rank_score(feature)

    def keep_score(self, feature: FeatureState, search: SearchState) -> float:
        return self._rank_score(feature)

    def target_kernel_size(self, search: SearchState) -> int:
        target = self.initial_kernel_size + (search.iteration + 1) * self.bucket_size
        target = max(search.feature_budget, target)
        target = min(search.total_features, target)
        if self.maximum_kernel_size is not None:
            target = min(target, max(search.feature_budget, self.maximum_kernel_size))
        return int(math.ceil(target))

    def _rank_score(self, feature: FeatureState) -> float:
        if self._rank:
            return float(-self._rank[feature.index])
        return float(getattr(feature, self.score_name))
