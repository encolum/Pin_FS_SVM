"""Deterministic handcrafted Adaptive Kernel Search policy."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from ..states import FeatureState, SearchState


@dataclass(frozen=True)
class ADKSWeights:
    """Explicit weights; no scientific defaults are guessed by the implementation."""

    initial_fisher: float
    initial_mutual_information: float
    initial_lp_activation: float

    keep_selected: float
    keep_abs_coefficient: float
    keep_selection_frequency: float
    keep_slack_association: float
    keep_lp_activation: float
    keep_redundancy_penalty: float
    keep_inactivity_penalty: float
    keep_kernel_age_penalty: float

    add_fisher: float
    add_mutual_information: float
    add_lp_activation: float
    add_slack_association: float
    add_nonredundancy: float
    add_selection_stability: float

    def __post_init__(self) -> None:
        for field_info in fields(self):
            value = float(getattr(self, field_info.name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"ADKS weight {field_info.name} must be finite and non-negative")


class HandcraftedADKSPolicy:
    name = "handcrafted_adks"

    def __init__(
        self,
        *,
        weights: ADKSWeights,
        initial_kernel_size: int,
        minimum_kernel_size: int,
        maximum_kernel_size: int | None,
        stagnation_threshold: int,
        focus_fraction: float,
        expansion_fraction: float,
    ) -> None:
        if min(int(initial_kernel_size), int(minimum_kernel_size)) < 1:
            raise ValueError("initial and minimum kernel sizes must be positive")
        if int(stagnation_threshold) < 1:
            raise ValueError("stagnation_threshold must be positive")
        if not 0 <= float(focus_fraction) < 1:
            raise ValueError("focus_fraction must lie in [0, 1)")
        if float(expansion_fraction) <= 0:
            raise ValueError("expansion_fraction must be positive")
        if maximum_kernel_size is not None and int(maximum_kernel_size) < 1:
            raise ValueError("maximum_kernel_size must be positive when provided")
        self.weights = weights
        self.initial_kernel_size = int(initial_kernel_size)
        self.minimum_kernel_size = int(minimum_kernel_size)
        self.maximum_kernel_size = (
            None if maximum_kernel_size is None else int(maximum_kernel_size)
        )
        self.stagnation_threshold = int(stagnation_threshold)
        self.focus_fraction = float(focus_fraction)
        self.expansion_fraction = float(expansion_fraction)

    def initialize_kernel(
        self,
        features: list[FeatureState],
        search: SearchState,
    ) -> set[int]:
        target = self._bounded_size(
            max(self.initial_kernel_size, self.minimum_kernel_size), search
        )
        ranked = sorted(
            features,
            key=lambda feature: (-self._initial_score(feature), feature.index),
        )
        return {feature.index for feature in ranked[:target]}

    def keep_score(self, feature: FeatureState, search: SearchState) -> float:
        weights = self.weights
        return float(
            weights.keep_selected * float(feature.is_selected)
            + weights.keep_abs_coefficient * feature.abs_coefficient
            + weights.keep_selection_frequency * feature.selection_frequency
            + weights.keep_slack_association * feature.slack_association
            + weights.keep_lp_activation * feature.lp_activation
            - weights.keep_redundancy_penalty * feature.support_redundancy
            - weights.keep_inactivity_penalty * feature.inactive_iterations
            - weights.keep_kernel_age_penalty
            * (feature.kernel_age if not feature.is_selected else 0)
        )

    def add_score(self, feature: FeatureState, search: SearchState) -> float:
        weights = self.weights
        return float(
            weights.add_fisher * feature.fisher_score
            + weights.add_mutual_information * feature.mutual_information
            + weights.add_lp_activation * feature.lp_activation
            + weights.add_slack_association * feature.slack_association
            + weights.add_nonredundancy * (1.0 - feature.support_redundancy)
            + weights.add_selection_stability * feature.selection_frequency
        )

    def target_kernel_size(self, search: SearchState) -> int:
        target = search.kernel_size
        if search.improved_last_iteration:
            target = math.ceil(target * (1.0 - self.focus_fraction))
        elif search.stagnation_iterations >= self.stagnation_threshold:
            target = math.ceil(target * (1.0 + self.expansion_fraction))
        return self._bounded_size(target, search)

    def _initial_score(self, feature: FeatureState) -> float:
        weights = self.weights
        return float(
            weights.initial_fisher * feature.fisher_score
            + weights.initial_mutual_information * feature.mutual_information
            + weights.initial_lp_activation * feature.lp_activation
        )

    def _bounded_size(self, size: int, search: SearchState) -> int:
        minimum = max(search.feature_budget, self.minimum_kernel_size)
        maximum = search.total_features
        if self.maximum_kernel_size is not None:
            maximum = min(maximum, max(minimum, self.maximum_kernel_size))
        return max(minimum, min(maximum, int(size)))
