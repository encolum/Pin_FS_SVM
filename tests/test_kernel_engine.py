from types import SimpleNamespace

import numpy as np
import pytest

from src.data.synthetic import generate_synthetic_instance
from src.search.kernel_engine import _update_kernel, run_kernel_search
from src.search.policies.static_ks import StaticKSPolicy


def _instance(seed=7):
    return generate_synthetic_instance(
        n_samples=24,
        n_features=8,
        informative_ratio=0.25,
        redundant_ratio=0.25,
        correlation_strength=0.9,
        positive_class_fraction=0.5,
        feature_budget_ratio=0.25,
        seed=seed,
    )


def _run(*, final_refinement=False):
    instance = _instance()
    return run_kernel_search(
        instance.X,
        instance.y,
        policy=StaticKSPolicy(
            score_name="fisher_score",
            initial_kernel_size=2,
            bucket_size=2,
        ),
        B=instance.feature_budget,
        C=1.0,
        tau=0.5,
        coefficient_bounds=(-4.0, 4.0),
        total_time_limit=3.0,
        subproblem_time_limit=0.5,
        max_iterations=3,
        backend="scipy",
        threads=1,
        final_full_refinement=final_refinement,
        final_refinement_fraction=0.2 if final_refinement else 0.0,
        seed=11,
        mip_gap=0.0,
        signal_options={"use_lp": False},
    )


def test_kernel_engine_retains_incumbent_and_never_worsens_best():
    result = _run()
    previous_support = set()
    best_values = []
    for record in result.history:
        kernel = set(record["kernel_features"])
        assert previous_support.issubset(kernel)
        assert 2 <= len(kernel) <= 8
        previous_support = set(record["support_features"])
        best_values.append(record["best_objective"])
    assert all(
        later <= earlier + 1e-9
        for earlier, later in zip(best_values, best_values[1:])
    )
    assert result.total_runtime <= 3.05
    assert result.metadata["time_budget_exceeded"] is False
    assert all(
        record["best_bound"] is None and record["relative_gap"] is None
        for record in result.metadata["route_progress"]
    )
    assert all(record["gap_scope"] == "restricted_kernel" for record in result.history)


def test_kernel_retains_active_support_but_not_selector_only_features():
    class IndexPolicy:
        def keep_score(self, feature, search):
            return -float(feature.index)

        def add_score(self, feature, search):
            return -float(feature.index)

    incumbent = SimpleNamespace(v=np.array([1, 1, 0]), support={0})
    search = SimpleNamespace(feature_budget=1, total_features=3)
    features = [SimpleNamespace(index=index) for index in range(3)]
    kernel = _update_kernel(
        features,
        current_kernel={0, 1},
        incumbent=incumbent,
        target_size=1,
        search=search,
        policy=IndexPolicy(),
    )
    assert kernel == {0}


def test_final_full_refinement_removes_all_kernel_restrictions():
    result = _run(final_refinement=True)
    assert result.final_kernel == set(range(8))
    assert result.history[-1]["iteration"] == "final_refinement"
    assert result.metadata["final_full_refinement"] is True
    assert result.best_result.objective <= result.history[0]["objective"] + 1e-7
    assert result.history[-1]["gap_scope"] == "full_model"
    assert all(
        record["gap_scope"] == "restricted_kernel" for record in result.history[:-1]
    )


def test_kernel_search_is_deterministic_for_fixed_policy_and_seed():
    first = _run()
    second = _run()
    assert first.initial_kernel == second.initial_kernel
    assert first.final_kernel == second.final_kernel
    assert first.best_result.support == second.best_result.support
    assert first.best_result.objective == pytest.approx(second.best_result.objective)
    assert [record["kernel_features"] for record in first.history] == [
        record["kernel_features"] for record in second.history
    ]
