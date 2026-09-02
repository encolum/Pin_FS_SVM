from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.config import load_config
from src.experiments.verapin import _engine_config
from src.search.llm_evolution.evaluator import PolicyInstance
from src.search.llm_evolution import references
from src.utils.serialization import read_json


def _inputs(tmp_path):
    config = load_config("configs/adks_real_pilot.yaml")
    config["solver"].update(backend="scipy", total_time_limit=3., subproblem_time_limit=1.)
    config["search"].update(max_iterations=1, final_full_refinement=False, final_refinement_fraction=0.)
    config["search"]["signal_options"].update(use_lp=False)
    config["adks_policy"].update(initial_kernel_size=2, minimum_kernel_size=2, maximum_kernel_size=4)
    rng = np.random.default_rng(7)
    instance = PolicyInstance("tiny", "train", rng.normal(size=(12, 4)), np.tile([-1, 1], 6),
        2, 1., .5, (-4., 4.), X_test=np.zeros((2, 4)), y_test=np.array([-1, 1]))
    return [instance], dict(solver_config=_engine_config(config), adks_config=config["adks_policy"],
                            output_path=tmp_path / "fitness_references.json")


def test_references_precomputed_once_on_training_only_and_resume_reuses_anchors(tmp_path, monkeypatch):
    instances, kwargs = _inputs(tmp_path)
    calls = []
    def baseline(instance, *, route, solver_config, adks_config):
        assert instance.X_test is instance.y_test is None
        calls.append((route, solver_config["total_time_limit"]))
        return {"route": route, "objective": 10. if route == "cold_full" else 8.}
    monkeypatch.setattr(references, "_reference_route", baseline)
    prepared = references.prepare_fitness_references(instances, **kwargs)
    assert calls == [("cold_full", 3.), ("handcrafted_adks", 3.)]
    assert prepared[0].reference_objective == 8. and prepared[0].fitness_horizon == 3.
    assert prepared[0].X_test is instances[0].X_test
    assert instances[0].reference_objective is None  # Immutable input.
    assert read_json(kwargs["output_path"])["status"] == "complete"
    replay = references.prepare_fitness_references(instances, **kwargs, reuse=True)
    assert len(calls) == 2
    assert replay[0].instance_hash == prepared[0].instance_hash


@pytest.mark.parametrize("change", ["data", "budget", "adks"])
def test_reference_resume_rejects_changed_problem_or_protocol(tmp_path, monkeypatch, change):
    instances, kwargs = _inputs(tmp_path)
    monkeypatch.setattr(references, "_reference_route", lambda *a, **k: {"objective": 10.})
    references.prepare_fitness_references(instances, **kwargs)
    if change == "data":
        instances = [replace(instances[0], X=instances[0].X + 1.)]
    elif change == "budget":
        kwargs["solver_config"]["total_time_limit"] += 1.
    else:
        kwargs["adks_config"]["initial_kernel_size"] += 1
    with pytest.raises(ValueError, match="configuration/data differ"):
        references.prepare_fitness_references(instances, **kwargs, reuse=True)


def test_missing_resume_references_cannot_be_recomputed(tmp_path, monkeypatch):
    instances, kwargs = _inputs(tmp_path)
    def forbidden(*a, **k):
        pytest.fail("resume must not rerun reference baselines")
    monkeypatch.setattr(references, "_reference_route", forbidden)
    with pytest.raises(ValueError, match="original fitness_references"):
        references.prepare_fitness_references(instances, **kwargs, reuse=True)


def test_no_baseline_incumbent_aborts_and_leaves_failure_evidence(tmp_path, monkeypatch):
    instances, kwargs = _inputs(tmp_path)
    monkeypatch.setattr(references, "_reference_route", lambda *a, **k: {"objective": None, "status": "failed"})
    with pytest.raises(RuntimeError, match="evolution aborted"):
        references.prepare_fitness_references(instances, **kwargs)
    assert read_json(kwargs["output_path"])["status"] == "failed"


def test_tiny_real_solvers_supply_a_finite_fixed_reference(tmp_path):
    instances, kwargs = _inputs(tmp_path)
    prepared = references.prepare_fitness_references(instances, **kwargs)
    baselines = prepared[0].metadata["fitness_reference"]["baselines"]
    assert {row["route"] for row in baselines} == {"cold_full", "handcrafted_adks"}
    assert all(row["status"] == "feasible" for row in baselines)
    assert prepared[0].reference_objective == min(row["objective"] for row in baselines)
    assert np.isfinite(prepared[0].reference_objective)


def test_reference_baseline_ignores_incumbent_discovered_after_budget(tmp_path, monkeypatch):
    instances, kwargs = _inputs(tmp_path)
    progress = [{"elapsed_seconds": 1., "incumbent_objective": 10.},
                {"elapsed_seconds": 4., "incumbent_objective": 5.}]
    monkeypatch.setattr(references, "run_kernel_search", lambda *a, **k:
        SimpleNamespace(total_runtime=4., metadata={"route_progress": progress},
                        best_result=SimpleNamespace(objective=5.)))
    row = references._reference_route(instances[0], route="handcrafted_adks",
        solver_config=kwargs["solver_config"], adks_config=kwargs["adks_config"])
    assert row["objective"] == 10.
