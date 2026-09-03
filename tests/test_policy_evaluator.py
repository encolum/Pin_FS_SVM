import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.synthetic import generate_synthetic_instance
from src.search.llm_evolution.candidate_parser import parse_candidates
from src.search.llm_evolution.evaluator import (
    FitnessNormalization,
    FitnessWeights,
    PolicyEvaluationCache,
    PolicyInstance,
    evaluate_policy,
)


def _candidate():
    value = {
        "schema_version": 1,
        "policy_id": "evaluator-policy",
        "name": "evaluator-policy",
        "initial_kernel_size": 2,
        "initial_score": {"feature": "fisher_score"},
        "add_score": {"feature": "mutual_information"},
        "keep_score": {"feature": "is_selected"},
        "target_kernel_size": {
            "op": "clip",
            "value": {"op": "add", "args": [{"search": "kernel_size"}, 1]},
            "lower": {"search": "feature_budget"},
            "upper": {"search": "total_features"},
        },
        "metadata": {},
    }
    return parse_candidates(json.dumps(value))[0]


def _instance(split="train", instance_id="instance-a", seed=3):
    generated = generate_synthetic_instance(
        n_samples=20,
        n_features=6,
        informative_ratio=0.34,
        redundant_ratio=0.33,
        correlation_strength=0.9,
        positive_class_fraction=0.5,
        feature_budget_ratio=0.34,
        seed=seed,
    )
    return PolicyInstance(
        instance_id=instance_id,
        research_split=split,
        X=generated.X,
        y=generated.y,
        B=generated.feature_budget,
        C=1.0,
        tau=0.5,
        coefficient_bounds=(-4.0, 4.0),
        reference_objective=20.0,  # Fixed feasible zero-classifier reference for this fixture.
        fitness_horizon=3.0,
    )


def _solver_config():
    return {
        "total_time_limit": 3.0,
        "subproblem_time_limit": 1.0,
        "max_iterations": 1,
        "backend": "scipy",
        "threads": 1,
        "final_full_refinement": False,
        "final_refinement_fraction": 0.0,
        "seed": 5,
        "mip_gap": 0.0,
        "signal_options": {"use_lp": False},
    }


def test_duplicate_policy_evaluations_are_cached(tmp_path):
    cache = PolicyEvaluationCache(tmp_path / "cache")
    kwargs = {
        "required_split": "train",
        "solver_config": _solver_config(),
        "fitness_weights": FitnessWeights(1.0, 1.0, 2.0, 0.5),
        "normalization": FitnessNormalization(3.0, 1.0, 1.0),
        "target_gap": 0.01,
        "cache": cache,
    }
    first = evaluate_policy(_candidate(), [_instance()], **kwargs)
    second = evaluate_policy(_candidate(), [_instance()], **kwargs)
    assert first.mean_fitness == pytest.approx(second.mean_fitness)
    assert first.failure_rate == 0.0
    assert cache.misses == 1
    assert cache.hits == 1
    assert first.per_instance[0]["research_split"] == "train"


def test_policy_evaluator_rejects_cross_split_leakage(tmp_path):
    with pytest.raises(ValueError, match="received instances"):
        evaluate_policy(
            _candidate(),
            [_instance(split="validation")],
            required_split="train",
            solver_config=_solver_config(),
            fitness_weights=FitnessWeights(1.0, 1.0, 1.0, 1.0),
            normalization=FitnessNormalization(3.0, 1.0, 1.0),
            target_gap=0.01,
            cache=PolicyEvaluationCache(tmp_path / "cache"),
        )


def _evaluation_kwargs(cache=None, horizon=10.):
    return dict(required_split="train", solver_config={**_solver_config(), "total_time_limit": horizon},
                fitness_weights=FitnessWeights(1., 0., 1., 0.),
                normalization=FitnessNormalization(horizon, 1., 1.), target_gap=.01, cache=cache)


def _mock_result(objective=20., runtime=2., extra_progress=()):
    return SimpleNamespace(total_runtime=runtime,
        metadata={"route_progress": [{"elapsed_seconds": 1., "incumbent_objective": objective,
            "best_bound": None, "relative_gap": None, "node_count": 0, "solution_count": 1}, *extra_progress]},
        best_result=SimpleNamespace(objective=objective, support={0}, coefficients=np.ones(6), intercept=0.))


def test_policies_use_one_reference_and_full_budget_even_when_they_stop_early(monkeypatch):
    results = iter([_mock_result(runtime=2.), _mock_result(runtime=7.), _mock_result(objective=10.)])
    monkeypatch.setattr("src.search.llm_evolution.evaluator.run_kernel_search", lambda *a, **k: next(results))
    instance = replace(_instance(), reference_objective=10., fitness_horizon=10.)
    evaluations = [evaluate_policy(_candidate(), [instance], **_evaluation_kwargs()) for _ in range(3)]
    # Same anytime objective, different actual runtimes: both pay error until T=10.
    assert [row.mean_primal_integral for row in evaluations] == pytest.approx([10., 10., 1.])
    assert evaluations[0].mean_fitness == evaluations[1].mean_fitness
    assert evaluations[2].mean_fitness < evaluations[0].mean_fitness
    assert all(row.per_instance[0]["reference_objective"] == 10. for row in evaluations)
    assert all(row.per_instance[0]["horizon"] == 10. for row in evaluations)


def test_late_progress_cannot_improve_in_budget_fitness(monkeypatch):
    late = {"elapsed_seconds": 12., "incumbent_objective": 10., "best_bound": 10.,
            "relative_gap": 0., "node_count": 1, "solution_count": 2}
    monkeypatch.setattr("src.search.llm_evolution.evaluator.run_kernel_search",
                        lambda *a, **k: _mock_result(runtime=12., extra_progress=[late]))
    result = evaluate_policy(_candidate(), [replace(_instance(), reference_objective=10., fitness_horizon=10.)],
                             **_evaluation_kwargs())
    assert result.mean_primal_integral == pytest.approx(10.)
    assert result.mean_final_gap == 1.
    assert result.mean_time_to_target_gap is None


@pytest.mark.parametrize("updates", [{"reference_objective": None}, {"reference_objective": float("nan")},
                                    {"fitness_horizon": None}, {"fitness_horizon": 9.}])
def test_invalid_fitness_anchors_fail_before_solver(monkeypatch, updates):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid protocol started solver")
    monkeypatch.setattr("src.search.llm_evolution.evaluator.run_kernel_search", forbidden)
    instance = replace(_instance(), fitness_horizon=10., **{k: v for k, v in updates.items() if k != "fitness_horizon"})
    if "fitness_horizon" in updates:
        instance = replace(instance, fitness_horizon=updates["fitness_horizon"])
    with pytest.raises(ValueError, match="reference_objective|fitness_horizon"):
        evaluate_policy(_candidate(), [instance], **_evaluation_kwargs())


def test_cache_and_instance_hash_include_reference_horizon_and_scoring_context(monkeypatch, tmp_path):
    calls = []
    def solve(*args, **kwargs):
        calls.append(kwargs)
        return _mock_result()
    monkeypatch.setattr("src.search.llm_evolution.evaluator.run_kernel_search", solve)
    cache = PolicyEvaluationCache(tmp_path / "cache")
    base = replace(_instance(), reference_objective=10., fitness_horizon=10.)
    other_reference = replace(base, reference_objective=5.)
    other_horizon = replace(base, fitness_horizon=20.)
    assert len({base.instance_hash, other_reference.instance_hash, other_horizon.instance_hash}) == 3
    kwargs = _evaluation_kwargs(cache)
    first = evaluate_policy(_candidate(), [base], **kwargs)
    evaluate_policy(_candidate(), [base], **{**kwargs, "cache": PolicyEvaluationCache(tmp_path / "cache")})
    assert len(calls) == 1  # Disk cache replay, not only in-memory reuse.
    second = evaluate_policy(_candidate(), [other_reference], **kwargs)
    third = evaluate_policy(_candidate(), [other_horizon], **_evaluation_kwargs(cache, horizon=20.))
    evaluate_policy(_candidate(), [base], **{**kwargs, "target_gap": .02})
    evaluate_policy(_candidate(), [base], **{**kwargs, "normalization": FitnessNormalization(5., 1., 1.)})
    assert len(calls) == 5
    assert first.mean_primal_integral != second.mean_primal_integral
    assert first.mean_primal_integral != third.mean_primal_integral
