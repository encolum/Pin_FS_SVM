import json

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
        label_noise_rate=0.0,
        outlier_sample_rate=0.0,
        outlier_feature_rate=0.0,
        outlier_scale=0.0,
        feature_budget_ratio=0.34,
        seed=seed,
        split=split,
    )
    return PolicyInstance(
        instance_id=instance_id,
        split=split,
        X=generated.X,
        y=generated.y,
        B=generated.feature_budget,
        C=1.0,
        tau=0.5,
        coefficient_bounds=(-4.0, 4.0),
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
    assert first.per_instance[0]["split"] == "train"


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
