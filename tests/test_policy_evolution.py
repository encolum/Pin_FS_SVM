import json
from dataclasses import replace

import pytest

from src.data.synthetic import generate_synthetic_instance
from src.search.llm_evolution.candidate_parser import parse_candidates
from src.search.llm_evolution.evaluator import (
    FitnessNormalization,
    FitnessWeights,
    PolicyInstance,
)
from src.search.llm_evolution.evolution import EvolutionConfig, run_evolution
from src.search.llm_evolution.provider import MockProvider
from src.utils.serialization import read_json


def _candidate(policy_id, signal="fisher_score"):
    value = {
        "schema_version": 1,
        "policy_id": policy_id,
        "name": policy_id,
        "initial_kernel_size": 2,
        "initial_score": {"feature": signal},
        "add_score": {"feature": signal},
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


def _instance(instance_id, split, seed):
    generated = generate_synthetic_instance(
        n_samples=18,
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
        instance_id,
        split,
        generated.X,
        generated.y,
        generated.feature_budget,
        1.0,
        0.5,
        (-4.0, 4.0),
        reference_objective=18.0,  # Fixed fixture reference, never derived from a candidate.
        fitness_horizon=2.0,
    )


def _solver_config():
    return {
        "total_time_limit": 2.0,
        "subproblem_time_limit": 0.5,
        "max_iterations": 1,
        "backend": "scipy",
        "threads": 1,
        "final_full_refinement": False,
        "final_refinement_fraction": 0.0,
        "seed": 4,
        "mip_gap": 0.0,
        "signal_options": {"use_lp": False},
    }


def _run(tmp_path, provider, generations, resume=False):
    return run_evolution(
        seed_candidates=[_candidate("seed")],
        training_instances=[_instance("train-a", "train", 1)],
        validation_instances=[_instance("validation-a", "validation", 2)],
        provider=provider,
        evolution_config=EvolutionConfig(
            generations=generations,
            population_size=2,
            parent_count=1,
            candidates_per_generation=1,
            maximum_similarity=0.99,
            seed=20,
        ),
        solver_config=_solver_config(),
        fitness_weights=FitnessWeights(1.0, 1.0, 2.0, 0.5),
        normalization=FitnessNormalization(2.0, 1.0, 1.0),
        target_gap=0.01,
        run_dir=tmp_path / "evolution",
        resume=resume,
    )


def test_mock_evolution_is_reproducible_and_freezes_on_validation(tmp_path):
    response = json.dumps({"candidates": [_candidate("mutation", "mutual_information").to_dict()]})
    first = _run(tmp_path, MockProvider([response]), generations=1)
    frozen = read_json(first.run_dir / "policies" / "frozen_verapin_policy.json")
    assert frozen["metadata"]["selected_on"] == "validation"
    assert frozen["metadata"]["training_instance_ids"] == ["train-a"]
    assert frozen["metadata"]["validation_instance_ids"] == ["validation-a"]
    assert "test" not in json.dumps(frozen).lower()
    assert (first.run_dir / "offline_llm_summary.json").is_file()


def test_mock_provider_is_reproducible_by_seed():
    provider = MockProvider(["first", "second"])
    assert provider.generate("prompt", seed=7) == "second"
    assert provider.generate("prompt", seed=7) == "second"


def test_resume_restores_population_and_continues_generation(tmp_path):
    response_one = json.dumps({"candidates": [_candidate("mutation-1").to_dict()]})
    _run(tmp_path, MockProvider([response_one]), generations=1)
    response_two = json.dumps({"candidates": [_candidate("mutation-2", "lp_activation").to_dict()]})
    result = _run(tmp_path, MockProvider([response_two]), generations=2, resume=True)
    checkpoint = read_json(result.run_dir / "checkpoint.json")
    assert checkpoint["generation"] == 2
    assert len(checkpoint["provider_records"]) == 2


def test_resume_rejects_generation_limit_below_checkpoint(tmp_path):
    response = json.dumps({"candidates": [_candidate("mutation").to_dict()]})
    _run(tmp_path, MockProvider([response]), generations=2)
    with pytest.raises(ValueError, match="cannot be lower"):
        _run(
            tmp_path,
            MockProvider([response]),
            generations=1,
            resume=True,
        )


def test_resume_rejects_changed_fitness_reference(tmp_path, monkeypatch):
    response = json.dumps({"candidates": [_candidate("mutation").to_dict()]})
    _run(tmp_path, MockProvider([response]), generations=1)
    original_instance = _instance
    def changed(*args, **kwargs):
        return replace(original_instance(*args, **kwargs), reference_objective=17.)
    monkeypatch.setitem(globals(), "_instance", changed)
    provider = MockProvider([response])
    with pytest.raises(ValueError, match="instance hashes differ"):
        _run(tmp_path, provider, generations=2, resume=True)
    assert not provider.records


def test_evolution_rejects_overlapping_train_validation_ids(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        run_evolution(
            seed_candidates=[_candidate("seed")],
            training_instances=[_instance("same", "train", 1)],
            validation_instances=[_instance("same", "validation", 2)],
            provider=MockProvider([json.dumps(_candidate("mutation").to_dict())]),
            evolution_config=EvolutionConfig(1, 1, 1, 1, 1.0, 1),
            solver_config=_solver_config(),
            fitness_weights=FitnessWeights(1.0, 1.0, 1.0, 1.0),
            normalization=FitnessNormalization(2.0, 1.0, 1.0),
            target_gap=0.01,
            run_dir=tmp_path / "bad",
        )
