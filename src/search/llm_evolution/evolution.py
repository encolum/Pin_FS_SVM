"""Checkpointed train-only policy evolution and validation-only freezing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.serialization import read_json, write_json

from .candidate_parser import CandidateValidationError, parse_candidates, validate_candidate
from .evaluator import (
    FitnessNormalization,
    FitnessWeights,
    PolicyEvaluation,
    PolicyEvaluationCache,
    PolicyInstance,
    evaluate_policy,
)
from .population import PopulationMember, select_strong_diverse, update_population
from .prompt_builder import build_evolution_prompt
from .provider import LLMProvider
from .schemas import PolicyCandidate


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int
    population_size: int
    parent_count: int
    candidates_per_generation: int
    maximum_similarity: float
    seed: int

    def __post_init__(self) -> None:
        if min(
            int(self.generations),
            int(self.population_size),
            int(self.parent_count),
            int(self.candidates_per_generation),
        ) < 1:
            raise ValueError("evolution counts must all be positive")
        if int(self.parent_count) > int(self.population_size):
            raise ValueError("parent_count cannot exceed population_size")
        if not 0 <= float(self.maximum_similarity) <= 1:
            raise ValueError("maximum_similarity must lie in [0, 1]")


@dataclass
class EvolutionResult:
    frozen_candidate: PolicyCandidate
    validation_evaluation: PolicyEvaluation
    population: list[PopulationMember]
    generations_completed: int
    run_dir: Path


def run_evolution(
    *,
    seed_candidates: list[PolicyCandidate],
    training_instances: list[PolicyInstance],
    validation_instances: list[PolicyInstance],
    provider: LLMProvider,
    evolution_config: EvolutionConfig,
    solver_config: dict[str, Any],
    fitness_weights: FitnessWeights,
    normalization: FitnessNormalization,
    target_gap: float,
    run_dir: str | Path,
    resume: bool = False,
) -> EvolutionResult:
    """Evolve on training only, select on validation only, and never accept test data."""
    _validate_partitions(training_instances, validation_instances)
    seed_candidates = [validate_candidate(candidate) for candidate in seed_candidates]
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    cache = PolicyEvaluationCache(run_dir / "evaluation_cache")
    failures: list[dict[str, Any]] = []
    run_signature = _run_signature(
        seed_candidates=seed_candidates,
        training_instances=training_instances,
        validation_instances=validation_instances,
        evolution_config=evolution_config,
        solver_config=solver_config,
        fitness_weights=fitness_weights,
        normalization=normalization,
        target_gap=target_gap,
    )

    if resume and checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if checkpoint.get("run_signature") != run_signature:
            raise ValueError("resume configuration or instance hashes differ from the checkpoint")
        prior_provider_records = list(checkpoint.get("provider_records", []))
        completed = int(checkpoint["generation"])
        if completed > int(evolution_config.generations):
            raise ValueError(
                "resume generation limit cannot be lower than the checkpoint generation"
            )
        population = [
            PopulationMember(
                candidate=validate_candidate(
                    PolicyCandidate.from_dict(item["candidate"])
                ),
                training_evaluation=PolicyEvaluation.from_dict(item["training_evaluation"]),
            )
            for item in checkpoint["population"]
        ]
        failures = list(checkpoint.get("failures", []))
    else:
        if not seed_candidates:
            raise ValueError("evolution requires at least one seed policy")
        population = [
            PopulationMember(
                candidate=candidate,
                training_evaluation=evaluate_policy(
                    candidate,
                    training_instances,
                    required_split="train",
                    solver_config=solver_config,
                    fitness_weights=fitness_weights,
                    normalization=normalization,
                    target_gap=target_gap,
                    cache=cache,
                ),
            )
            for candidate in seed_candidates
        ]
        population = update_population(
            [],
            population,
            population_size=evolution_config.population_size,
            maximum_similarity=evolution_config.maximum_similarity,
        )
        completed = 0
        prior_provider_records = []
        _write_checkpoint(
            checkpoint_path,
            completed,
            population,
            failures,
            provider,
            prior_provider_records=prior_provider_records,
            run_signature=run_signature,
        )

    for generation in range(completed + 1, evolution_config.generations + 1):
        parents = select_strong_diverse(
            population,
            count=min(evolution_config.parent_count, len(population)),
            maximum_similarity=evolution_config.maximum_similarity,
        )
        prompt = build_evolution_prompt(
            generation=generation,
            training_summary=[
                member.training_evaluation.to_dict() for member in population
            ],
            parent_policies=[member.candidate.to_dict() for member in parents],
            failure_summary=failures[-20:],
            requested_candidates=evolution_config.candidates_per_generation,
        )
        response = provider.generate(prompt, seed=evolution_config.seed + generation)
        try:
            generated = parse_candidates(response)
        except CandidateValidationError as exc:
            failures.append(
                {"generation": generation, "stage": "parse", "message": str(exc)}
            )
            generated = []
        evaluated: list[PopulationMember] = []
        for candidate in generated[: evolution_config.candidates_per_generation]:
            evaluation = evaluate_policy(
                candidate,
                training_instances,
                required_split="train",
                solver_config=solver_config,
                fitness_weights=fitness_weights,
                normalization=normalization,
                target_gap=target_gap,
                cache=cache,
            )
            evaluated.append(PopulationMember(candidate, evaluation))
        population = update_population(
            population,
            evaluated,
            population_size=evolution_config.population_size,
            maximum_similarity=evolution_config.maximum_similarity,
        )
        _write_checkpoint(
            checkpoint_path,
            generation,
            population,
            failures,
            provider,
            prior_provider_records=prior_provider_records,
            run_signature=run_signature,
        )

    validation_members: list[tuple[PopulationMember, PolicyEvaluation]] = []
    for member in population:
        validation = evaluate_policy(
            member.candidate,
            validation_instances,
            required_split="validation",
            solver_config=solver_config,
            fitness_weights=fitness_weights,
            normalization=normalization,
            target_gap=target_gap,
            cache=cache,
        )
        validation_members.append((member, validation))
    selected_member, selected_validation = min(
        validation_members,
        key=lambda pair: (pair[1].mean_fitness, pair[0].candidate.policy_hash),
    )
    frozen = selected_member.candidate.to_dict()
    frozen["metadata"] = {
        **frozen.get("metadata", {}),
        "frozen": True,
        "selected_on": "validation",
        "training_instance_ids": sorted(instance.instance_id for instance in training_instances),
        "validation_instance_ids": sorted(
            instance.instance_id for instance in validation_instances
        ),
        "validation_evaluation": selected_validation.to_dict(),
    }
    frozen_candidate = PolicyCandidate.from_dict(frozen)
    policy_path = run_dir / "policies" / "frozen_verapin_policy.json"
    write_json(policy_path, frozen_candidate.to_dict())
    write_json(
        run_dir / "validation_selection.json",
        {
            "selected_policy_id": frozen_candidate.policy_id,
            "evaluations": [
                {
                    "policy_id": member.candidate.policy_id,
                    "validation": evaluation.to_dict(),
                }
                for member, evaluation in validation_members
            ],
        },
    )
    return EvolutionResult(
        frozen_candidate=frozen_candidate,
        validation_evaluation=selected_validation,
        population=population,
        generations_completed=evolution_config.generations,
        run_dir=run_dir,
    )


def _validate_partitions(
    training: list[PolicyInstance], validation: list[PolicyInstance]
) -> None:
    if not training or not validation:
        raise ValueError("training and validation partitions must both be non-empty")
    if any(instance.split != "train" for instance in training):
        raise ValueError("training_instances may contain only split='train'")
    if any(instance.split != "validation" for instance in validation):
        raise ValueError("validation_instances may contain only split='validation'")
    train_ids = {instance.instance_id for instance in training}
    validation_ids = {instance.instance_id for instance in validation}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError(f"training and validation instance IDs overlap: {overlap}")


def _write_checkpoint(
    path: Path,
    generation: int,
    population: list[PopulationMember],
    failures: list[dict[str, Any]],
    provider: LLMProvider,
    *,
    prior_provider_records: list[dict[str, Any]],
    run_signature: dict[str, Any],
) -> None:
    records = list(prior_provider_records) + [
        record.to_dict() for record in getattr(provider, "records", [])
    ]
    write_json(
        path,
        {
            "generation": generation,
            "run_signature": run_signature,
            "population": [
                {
                    "candidate": member.candidate.to_dict(),
                    "training_evaluation": member.training_evaluation.to_dict(),
                }
                for member in population
            ],
            "failures": failures,
            "provider_records": records,
        },
    )
    write_json(path.parent / "provider_records.json", records)
    write_json(
        path.parent / "offline_llm_summary.json",
        {
            "calls": len(records),
            "input_tokens": sum(int(record.get("input_tokens") or 0) for record in records),
            "output_tokens": sum(int(record.get("output_tokens") or 0) for record in records),
            "latency_seconds": sum(
                float(record.get("latency_seconds") or 0.0) for record in records
            ),
            "estimated_cost": (
                None
                if any(record.get("estimated_cost") is None for record in records)
                else sum(float(record["estimated_cost"]) for record in records)
            ),
            "reported_separately_from_online_solver_time": True,
        },
    )


def _run_signature(
    *,
    seed_candidates: list[PolicyCandidate],
    training_instances: list[PolicyInstance],
    validation_instances: list[PolicyInstance],
    evolution_config: EvolutionConfig,
    solver_config: dict[str, Any],
    fitness_weights: FitnessWeights,
    normalization: FitnessNormalization,
    target_gap: float,
) -> dict[str, Any]:
    evolution = asdict(evolution_config)
    evolution.pop("generations", None)
    return {
        "evolution_without_generation_limit": evolution,
        "solver_config": solver_config,
        "fitness_weights": asdict(fitness_weights),
        "fitness_normalization": asdict(normalization),
        "target_gap": float(target_gap),
        "seed_policy_hashes": sorted(candidate.policy_hash for candidate in seed_candidates),
        "training_instances": [
            {"id": instance.instance_id, "hash": instance.instance_hash}
            for instance in training_instances
        ],
        "validation_instances": [
            {"id": instance.instance_id, "hash": instance.instance_hash}
            for instance in validation_instances
        ],
    }
