import pytest

from src.search.policies.static_ks import StaticKSPolicy
from src.search.states import FeatureState, SearchState


def _feature(index, fisher):
    return FeatureState(
        index=index,
        in_kernel=False,
        is_selected=False,
        abs_coefficient=0.0,
        fisher_score=fisher,
        mutual_information=0.0,
        mean_abs_correlation=0.0,
        max_abs_correlation=0.0,
        lp_activation=0.0,
        lp_abs_coefficient=0.0,
        slack_association=0.0,
        selection_frequency=0.0,
        inactive_iterations=0,
        kernel_age=0,
    )


def _search(iteration=0):
    return SearchState(
        iteration=iteration,
        current_objective=0.0,
        best_objective=0.0,
        current_gap=None,
        best_bound=None,
        kernel_size=2,
        feature_budget=2,
        total_features=6,
        stagnation_iterations=0,
        elapsed_seconds=0.0,
        remaining_seconds=10.0,
        C=1.0,
        tau=0.5,
    )


def test_static_ranking_and_bucket_order_are_stable():
    features = [_feature(i, score) for i, score in enumerate([0.2, 0.9, 0.9, 0.1, 0.5, 0.4])]
    policy = StaticKSPolicy(
        score_name="fisher_score",
        initial_kernel_size=2,
        bucket_size=2,
    )
    assert policy.stable_order(features) == [1, 2, 4, 5, 0, 3]
    assert policy.initialize_kernel(features, _search()) == {1, 2}
    assert policy.target_kernel_size(_search(iteration=0)) == 4
    assert policy.target_kernel_size(_search(iteration=1)) == 6


def test_static_policy_rejects_unknown_signal():
    with pytest.raises(ValueError, match="unsupported static score"):
        StaticKSPolicy(score_name="heldout_accuracy", initial_kernel_size=2, bucket_size=1)
