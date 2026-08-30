"""Frozen, deterministic VeraPin policy compiled from the safe JSON DSL."""

from __future__ import annotations

import math

from ..llm_evolution.sandbox import compile_expression
from ..llm_evolution.schemas import PolicyCandidate
from ..states import FeatureState, SearchState


class FrozenVeraPinPolicy:
    def __init__(self, candidate: PolicyCandidate) -> None:
        self.candidate = candidate
        self.name = f"verapin_ks:{candidate.policy_id}"
        self._initial = compile_expression(candidate.initial_score)
        self._add = compile_expression(candidate.add_score)
        self._keep = compile_expression(candidate.keep_score)
        self._target = compile_expression(
            candidate.target_kernel_size,
            allow_feature_signals=False,
        )

    def initialize_kernel(
        self,
        features: list[FeatureState],
        search: SearchState,
    ) -> set[int]:
        size = max(search.feature_budget, self.candidate.initial_kernel_size)
        size = min(search.total_features, size)
        ranked = sorted(
            features,
            key=lambda feature: (-self._initial(feature, search), feature.index),
        )
        return {feature.index for feature in ranked[:size]}

    def add_score(self, feature: FeatureState, search: SearchState) -> float:
        return self._add(feature, search)

    def keep_score(self, feature: FeatureState, search: SearchState) -> float:
        return self._keep(feature, search)

    def target_kernel_size(self, search: SearchState) -> int:
        value = self._target(None, search)
        if value <= 0:
            raise ValueError("frozen policy returned a non-positive target kernel size")
        return int(math.ceil(value))
