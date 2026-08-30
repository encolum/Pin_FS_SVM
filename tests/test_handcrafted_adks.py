from dataclasses import replace

import pytest

from src.search.policies.handcrafted_adks import ADKSWeights, HandcraftedADKSPolicy
from src.search.states import FeatureState, SearchState


def _weights():
    return ADKSWeights(
        initial_fisher=1.0,
        initial_mutual_information=1.0,
        initial_lp_activation=1.0,
        keep_selected=4.0,
        keep_abs_coefficient=2.0,
        keep_selection_frequency=1.0,
        keep_slack_association=1.0,
        keep_lp_activation=1.0,
        keep_redundancy_penalty=1.0,
        keep_inactivity_penalty=0.1,
        keep_kernel_age_penalty=0.1,
        add_fisher=1.0,
        add_mutual_information=1.0,
        add_lp_activation=1.0,
        add_slack_association=1.0,
        add_nonredundancy=1.0,
        add_selection_stability=1.0,
    )


def _policy():
    return HandcraftedADKSPolicy(
        weights=_weights(),
        initial_kernel_size=8,
        minimum_kernel_size=4,
        maximum_kernel_size=20,
        stagnation_threshold=2,
        focus_fraction=0.25,
        expansion_fraction=0.5,
    )


def _search(**updates):
    state = SearchState(
        iteration=1,
        current_objective=10.0,
        best_objective=10.0,
        current_gap=0.2,
        best_bound=8.0,
        kernel_size=8,
        feature_budget=3,
        total_features=20,
        stagnation_iterations=0,
        elapsed_seconds=1.0,
        remaining_seconds=9.0,
        C=1.0,
        tau=0.5,
        improved_last_iteration=False,
    )
    return replace(state, **updates)


def _feature(**updates):
    state = FeatureState(
        index=0,
        in_kernel=True,
        is_selected=False,
        abs_coefficient=0.0,
        fisher_score=0.5,
        mutual_information=0.5,
        mean_abs_correlation=0.2,
        max_abs_correlation=0.3,
        lp_activation=0.5,
        lp_abs_coefficient=0.5,
        slack_association=0.5,
        selection_frequency=0.5,
        inactive_iterations=0,
        kernel_age=0,
        support_redundancy=0.2,
    )
    return replace(state, **updates)


def test_adks_focuses_after_improvement_and_expands_after_stagnation():
    policy = _policy()
    assert policy.target_kernel_size(_search(improved_last_iteration=True)) == 6
    assert policy.target_kernel_size(_search(stagnation_iterations=2)) == 12
    assert policy.target_kernel_size(
        _search(kernel_size=4, improved_last_iteration=True)
    ) == 4


def test_adks_keep_score_rewards_selected_and_penalizes_unused_redundancy():
    policy = _policy()
    selected = _feature(is_selected=True, abs_coefficient=1.0, selection_frequency=1.0)
    unused = _feature(
        support_redundancy=1.0,
        inactive_iterations=8,
        kernel_age=8,
    )
    assert policy.keep_score(selected, _search()) > policy.keep_score(unused, _search())
    assert policy.add_score(_feature(support_redundancy=0.0), _search()) > policy.add_score(
        _feature(support_redundancy=1.0), _search()
    )


def test_adks_weights_must_be_explicit_and_finite():
    values = _weights().__dict__.copy()
    values["add_fisher"] = float("nan")
    with pytest.raises(ValueError, match="add_fisher"):
        ADKSWeights(**values)
