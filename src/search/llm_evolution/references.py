"""Policy-independent, fixed-budget fitness anchors, persisted before evolution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from src.search.kernel_engine import run_kernel_search
from src.search.policies.handcrafted_adks import ADKSWeights, HandcraftedADKSPolicy
from src.search.restricted_solver import solve_restricted_pin_fs
from src.utils.serialization import read_json, write_json

from .evaluator import (
    FITNESS_PROTOCOL_VERSION, PolicyInstance, _canonical_hash, validate_fitness_protocol,
)


def prepare_fitness_references(instances, *, solver_config, adks_config, output_path, reuse=False):
    """Run cold full MILP + fixed ADKS once per instance; never use test labels.

    Each baseline receives the same explicit total_time_limit as each policy.
    Their offline cost is recorded separately, not charged to candidate fitness.
    References remain frozen even when a candidate subsequently beats them.
    Resume must reuse the original anchors, not run stochastic/time-limited
    baselines again. Old checkpoints without this protocol are not compatible.
    """
    horizon = float(solver_config["total_time_limit"])
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("reference budget must be finite and positive")
    if not instances or any(instance.split not in {"train", "validation"} for instance in instances):
        raise ValueError("fitness references accept only research train/validation instances")
    if len({instance.instance_id for instance in instances}) != len(instances):
        raise ValueError("fitness reference instance IDs must be unique")
    # Hash the prepared problem identity, without any previous scoring anchors.
    identities = [
        {"instance_id": instance.instance_id,
         "problem_hash": replace(instance, reference_objective=None, fitness_horizon=None).instance_hash}
        for instance in instances
    ]
    protocol = {"version": FITNESS_PROTOCOL_VERSION, "instances": identities,
                "horizon": horizon, "solver": solver_config, "adks_policy": adks_config}
    signature = _canonical_hash(protocol)
    output_path = Path(output_path)
    if reuse:
        if not output_path.is_file():
            raise ValueError("resume requires the original fitness_references.json; start a new run")
        artifact = read_json(output_path)
        if artifact.get("signature") != signature or artifact.get("status") != "complete":
            raise ValueError("fitness reference configuration/data differ or references are incomplete")
    else:
        artifact = {"signature": signature, "protocol": protocol, "status": "preparing", "instances": []}
        write_json(output_path, artifact)
        for instance, identity in zip(instances, identities):
            # Deliberately remove outer-test arrays: neither baseline has access.
            training = replace(instance, X_test=None, y_test=None)
            baselines = [
                _reference_route(training, route=route, solver_config=solver_config, adks_config=adks_config)
                for route in ("cold_full", "handcrafted_adks")
            ]
            objectives = [row["objective"] for row in baselines if row["objective"] is not None]
            artifact["instances"].append({**identity, "horizon": horizon,
                "reference_objective": min(objectives) if objectives else None,
                "baselines": baselines})
            if not objectives:
                artifact["status"] = "failed"
                write_json(output_path, artifact)
                raise RuntimeError(f"{instance.instance_id}: no feasible baseline objective within reference budget; evolution aborted")
            write_json(output_path, artifact)
        artifact["status"] = "complete"
        write_json(output_path, artifact)
    rows = artifact.get("instances", [])
    if [{key: row.get(key) for key in ("instance_id", "problem_hash")} for row in rows] != identities:
        raise ValueError("fitness reference artifact has mismatched instances")
    prepared = []
    for instance, row in zip(instances, rows):
        objectives = [baseline["objective"] for baseline in row["baselines"]
                      if baseline["objective"] is not None]
        if not objectives or row["reference_objective"] != min(objectives) or row["horizon"] != horizon:
            raise ValueError("invalid persisted fitness reference")
        prepared.append(replace(instance, reference_objective=float(row["reference_objective"]),
            fitness_horizon=horizon, metadata={**instance.metadata, "fitness_reference": row}))
    validate_fitness_protocol(prepared, solver_config)
    return prepared


def _reference_route(instance: PolicyInstance, *, route, solver_config, adks_config):
    horizon = float(solver_config["total_time_limit"])
    started = perf_counter()
    try:
        problem = dict(B=instance.B, C=instance.C, tau=instance.tau,
                       coefficient_bounds=instance.coefficient_bounds)
        if route == "cold_full":
            result = solve_restricted_pin_fs(instance.X, instance.y,
                kernel=set(range(instance.X.shape[1])), **problem,
                backend=solver_config["backend"], time_limit=horizon,
                mip_gap=solver_config["mip_gap"], threads=solver_config["threads"],
                collect_progress=True, deadline=started + horizon)
            runtime = perf_counter() - started
            # Solver progress uses solver-local time; account for model setup.
            offset = max(0., runtime - max((record.elapsed_seconds for record in result.progress), default=0.))
            objectives = [record.incumbent_objective for record in result.progress
                          if record.elapsed_seconds + offset <= horizon]
            final_objective = result.objective
        else:
            policy_config = dict(adks_config)
            policy = HandcraftedADKSPolicy(weights=ADKSWeights(**policy_config.pop("weights")), **policy_config)
            result = run_kernel_search(instance.X, instance.y, policy=policy, **problem, **solver_config)
            runtime = result.total_runtime
            objectives = [record["incumbent_objective"] for record in result.metadata.get("route_progress", [])
                          if 0 <= record["elapsed_seconds"] <= horizon]
            final_objective = result.best_result.objective
        if runtime <= horizon:
            objectives.append(final_objective)
        objectives = [float(value) for value in objectives if value is not None and np.isfinite(value)]
        return {"route": route, "backend": solver_config["backend"], "time_budget": horizon,
                "total_runtime": perf_counter() - started, "objective": min(objectives) if objectives else None,
                "status": "feasible" if objectives else "no_in_budget_incumbent"}
    except Exception as exc:
        return {"route": route, "backend": solver_config["backend"], "time_budget": horizon,
                "total_runtime": perf_counter() - started, "objective": None,
                "status": "failed", "exception_type": type(exc).__name__, "message": str(exc)}
