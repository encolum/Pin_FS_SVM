"""Selection frequencies and pairwise Jaccard stability."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def feature_stability(selected_sets: list[list[int]], n_features: int) -> dict[str, object]:
    sets = [set(values) for values in selected_sets]
    frequencies = np.zeros(n_features, dtype=float)
    for values in sets:
        for index in values:
            if not 0 <= index < n_features:
                raise ValueError(f"feature index out of bounds: {index}")
            frequencies[index] += 1
    if sets:
        frequencies /= len(sets)
    similarities = [jaccard(a, b) for a, b in combinations(sets, 2)]
    return {
        "selection_frequency": frequencies.tolist(),
        "mean_pairwise_jaccard": float(np.mean(similarities)) if similarities else 1.0,
    }
