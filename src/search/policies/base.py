"""Policy contract kept deliberately separate from the generic search engine."""

from __future__ import annotations

from typing import Protocol

from ..states import FeatureState, SearchState


class KernelPolicy(Protocol):
    name: str

    def initialize_kernel(
        self,
        features: list[FeatureState],
        search: SearchState,
    ) -> set[int]: ...

    def add_score(self, feature: FeatureState, search: SearchState) -> float: ...

    def keep_score(self, feature: FeatureState, search: SearchState) -> float: ...

    def target_kernel_size(self, search: SearchState) -> int: ...
