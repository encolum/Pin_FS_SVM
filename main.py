"""Safe, thin command-line entry point for corrected experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.data.loaders import DATASET_SPECS, audit_datasets
from src.experiments.config import load_config, validate_config
from src.experiments.runner import run_experiment
from src.experiments.sensitivity import run_sensitivity
from src.experiments.search import estimate_model_fits, estimate_top_level_fits, parameter_candidates
from src.reporting.plots import plot_balanced_accuracy
from src.reporting.tables import create_markdown_table
from src.statistics.wilcoxon import run_wilcoxon_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Corrected Pin-FS-SVM experiment pipeline")
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate", help="audit all dataset files and optionally a config")
    validate.add_argument("--config")

    benchmarks = subparsers.add_parser(
        "validate-datasets", help="check curated dataset/ inputs without running experiments"
    )
    benchmarks.add_argument("--data-root")

    for command, default in (
        ("pilot", "configs/pilot.yaml"),
        ("sensitivity", "configs/sensitivity.yaml"),
        ("ablation", "configs/ablation.yaml"),
    ):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default=default)
        sub.add_argument("--resume", metavar="RUN_DIR")

    full = subparsers.add_parser("run", help="run the full benchmark only with explicit confirmation")
    full.add_argument("--config", default="configs/full.yaml")
    full.add_argument("--confirm-full-run", action="store_true")
    full.add_argument("--resume", metavar="RUN_DIR")

    statistics = subparsers.add_parser("statistics", help="optional post-hoc Wilcoxon/BH analysis")
    statistics.add_argument("--run-dir", required=True)
    statistics.add_argument("--proposed-model", default="pin_fs_svm")

    analyze = subparsers.add_parser("analyze", help="create Markdown tables from a completed run")
    analyze.add_argument("--run-dir", required=True)

    plot = subparsers.add_parser("plot", help="plot saved Balanced Accuracy values")
    plot.add_argument("--run-dir", required=True)

    for command, default in (
        ("hardness", "configs/hardness.yaml"),
        ("kernel-search", "configs/static_ks_pilot.yaml"),
        ("adks", "configs/adks_pilot.yaml"),
    ):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default=default)

    evolve = subparsers.add_parser("evolve-verapin")
    evolve.add_argument("--config", default="configs/verapin_evolution.yaml")
    evolve.add_argument("--resume", metavar="RUN_DIR")

    evaluate = subparsers.add_parser("evaluate-verapin")
    evaluate.add_argument("--config", default="configs/verapin_final.yaml")
    evaluate.add_argument("--confirm-full-run", action="store_true")

    verify_policy = subparsers.add_parser("verify-policy")
    verify_policy.add_argument("--policy", required=True)

    replay = subparsers.add_parser("replay-evolution")
    replay.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "validate-datasets":
        from src.data.benchmarks import audit_benchmark_datasets

        report = audit_benchmark_datasets(data_root=args.data_root)
        print(f"Validated {len(report)} curated partitions (SHA-256, shape, labels, finite values).")
        for row in report:
            print(f"  {row['dataset']}/{row['partition']}: {row['samples']} x {row['features']}")
        return 0
    if args.command == "validate":
        report = audit_datasets(include_archived_variants=True)
        if args.config:
            validate_config(load_config(args.config))
        print(f"Validated {len(report)} dataset-condition files with no shape or label errors.")
        for row in report:
            print(f"  {row['dataset']}/{row['condition']}: {row['samples']} x {row['features']}")
        return 0
    if args.command == "statistics":
        output = run_wilcoxon_analysis(args.run_dir, proposed_model=args.proposed_model)
        print(output)
        return 0
    if args.command == "analyze":
        print(create_markdown_table(args.run_dir))
        return 0
    if args.command == "plot":
        print(plot_balanced_accuracy(args.run_dir))
        return 0
    if args.command == "verify-policy":
        from src.experiments.verapin import verify_policy_file

        print(verify_policy_file(args.policy))
        return 0
    if args.command == "replay-evolution":
        from src.experiments.verapin import replay_evolution_audit

        print(replay_evolution_audit(args.run_dir))
        return 0
    if args.command in {
        "hardness",
        "kernel-search",
        "adks",
        "evolve-verapin",
        "evaluate-verapin",
    }:
        from src.experiments.verapin import (
            run_adks,
            run_hardness_benchmark,
            run_static_kernel_search,
            run_verapin_evolution,
            run_verapin_final,
            validate_verapin_config,
        )

        config = load_config(args.config)
        try:
            validate_verapin_config(config, command=args.command)
        except ValueError as exc:
            parser.error(str(exc))
        if args.command == "evaluate-verapin" and not args.confirm_full_run:
            parser.error("held-out VeraPin evaluation requires --confirm-full-run")
        print(f"VeraPin command: {args.command}")
        print(f"Instances: {[item['id'] for item in config['instances']]}")
        print(f"Solver: {config['solver']}")
        print(f"Output directory: {config.get('output', {}).get('root', 'results_verapin')}")
        sys.stdout.flush()
        if args.command == "hardness":
            output = run_hardness_benchmark(config)
        elif args.command == "kernel-search":
            output = run_static_kernel_search(config)
        elif args.command == "adks":
            output = run_adks(config)
        elif args.command == "evolve-verapin":
            output = run_verapin_evolution(config, resume_dir=args.resume)
        else:
            output = run_verapin_final(config)
        print(f"Completed VeraPin run: {output}")
        return 0

    config = load_config(args.config)
    try:
        validate_config(config)
        if args.command == "sensitivity":
            _validate_sensitivity(config)
    except ValueError as exc:
        parser.error(str(exc))
    _print_run_summary(args.command, config)
    sys.stdout.flush()
    if args.command == "run" and not args.confirm_full_run:
        parser.error("the full benchmark requires --confirm-full-run")
    if args.command == "sensitivity":
        run_dir = run_sensitivity(config, resume_dir=args.resume)
    else:
        run_dir = run_experiment(config, mode=args.command, resume_dir=args.resume)
    print(f"Completed run: {run_dir}")
    return 0


def _print_run_summary(mode: str, config: dict) -> None:
    grids = {
        dataset: {
            model: len(parameter_candidates(model, DATASET_SPECS[dataset].features, config))
            for model in config["models"]
        }
        for dataset in config["datasets"]
    }
    print(f"Run mode: {mode}")
    print(f"Datasets: {config['datasets']}")
    print(f"Models: {config['models']}")
    print(f"Conditions: {config['conditions']}")
    print(f"Hyperparameter grid sizes: {grids}")
    print(f"Outer folds: {config['outer_folds']}")
    print(f"Inner folds: {config['inner_folds']}")
    print(f"Random seeds: {config['seeds']}")
    print(f"Solver: {config.get('solver', {})}")
    print(f"Parallelism: {config.get('parallelism', {'max_workers': 1})}")
    print(f"Estimated top-level fit calls: {estimate_top_level_fits(config)}")
    print(f"Estimated underlying optimization fits (conservative): {estimate_model_fits(config)}")
    print(f"Output directory: {config.get('output', {}).get('root', 'results_v2')}")


def _validate_sensitivity(config: dict) -> None:
    parameter = config.get("sensitivity_parameter")
    if parameter not in {"B", "C", "tau"}:
        raise ValueError("sensitivity_parameter must be exactly one of B, C, or tau")
    if len(config["models"]) != 1:
        raise ValueError("a sensitivity run must contain exactly one model")
    grids = {"C": config["C_grid"], "tau": config["tau_grid"], "B": config.get("B_grid")}
    for name, values in grids.items():
        if not isinstance(values, list):
            raise ValueError("sensitivity grids must be explicit lists")
        if name == parameter and len(values) < 2:
            raise ValueError(f"{name} requires at least two sensitivity values")
        if name != parameter and len(values) != 1:
            raise ValueError(f"{name} must be held fixed at exactly one value")


if __name__ == "__main__":
    raise SystemExit(main())
