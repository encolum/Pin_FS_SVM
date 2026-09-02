"""Safe, thin command-line entry point for Pin-FS-SVM and VeraPin-KS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pin-FS-SVM and VeraPin-KS experiment pipeline")
    subparsers = parser.add_subparsers(dest="command")

    raw_data = subparsers.add_parser("validate-datasets", help="inspect original uploads; no training")
    raw_data.add_argument("--data-root")
    report_destination = raw_data.add_mutually_exclusive_group()
    report_destination.add_argument("--output", help="export a separate JSON validation report")
    report_destination.add_argument("--update-manifest", action="store_true",
                                    help="refresh only the validation block in dataset/manifest.json")
    raw_data.add_argument("--overwrite", action="store_true", help="replace an existing validation report")

    benchmarks = subparsers.add_parser("validate-benchmarks", help="audit solver-facing benchmark views; no training")
    from src.data.benchmark_registry import DEFAULT_REGISTRY_PATH

    benchmarks.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    benchmarks.add_argument("--data-root")
    benchmarks.add_argument("--output", help="export a separate JSON validation report")
    benchmarks.add_argument("--overwrite", action="store_true", help="replace an existing validation report")

    for command, default in (
        ("hardness", "configs/hardness.yaml"),
        ("kernel-search", "configs/static_ks_pilot.yaml"),
        ("adks", "configs/adks_pilot.yaml"),
    ):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default=default)
        sub.add_argument("--instance", action="append", help="run only the named instance (repeatable)")
        sub.add_argument("--validate-only", action="store_true", help="validate config without creating a run or solving")

    evolve = subparsers.add_parser("evolve-verapin")
    evolve.add_argument("--config", default="configs/verapin_evolution.yaml")
    evolve.add_argument("--resume", metavar="RUN_DIR")
    evolve.add_argument("--validate-only", action="store_true")

    evaluate = subparsers.add_parser("evaluate-verapin")
    evaluate.add_argument("--config", default="configs/verapin_final.yaml")
    evaluate.add_argument("--confirm-full-run", action="store_true")
    evaluate.add_argument("--validate-only", action="store_true")

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
    if args.command == "validate-benchmarks":
        from src.data.benchmark_adapter import audit_solver_ready_benchmarks
        from src.data.benchmark_validation import write_validation_manifest

        try:
            if args.output:
                output, registry = Path(args.output).resolve(), Path(args.registry).resolve()
                if output == registry or (output.exists() and registry.exists() and output.samefile(registry)):
                    raise ValueError("validation output must not overwrite the benchmark registry")
            report = audit_solver_ready_benchmarks(data_root=args.data_root, registry_path=args.registry)
            if args.output:
                write_validation_manifest(report, args.output, data_root=args.data_root,
                                          overwrite=args.overwrite)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        passed = sum(row["status"] == "passed" for row in report)
        print(f"Solver-ready benchmark validation: {passed}/{len(report)} passed; no training or preprocessing run.")
        for row in report:
            if row["status"] == "passed":
                print(f"  {row['dataset']}: shape=({row['samples']}, {row['features']}), "
                      f"positive={row['positive']}, negative={row['negative']}, "
                      f"storage={row['storage']}, density={row['density']:.6%}, "
                      f"partition_policy={row['partition_policy']}, "
                      f"label_mapping={row['label_mapping']}, preprocessing={row['preprocessing_policy']}")
                if row["holdout"] is not None:
                    holdout = row["holdout"]
                    print(f"    Separate {holdout['source_partition']} holdout: "
                          f"{holdout['samples']} rows; positive={holdout['positive']}, negative={holdout['negative']}")
                for warning in row["warnings"]:
                    print(f"    Warning: {warning}")
            for error in row["errors"]:
                print(f"  {row['dataset']}: failed: {error}")
        return 0 if passed == len(report) else 1
    if args.command == "validate-datasets":
        from src.data.benchmark_validation import (
            audit_benchmark_datasets, update_dataset_validation, write_validation_manifest,
        )

        try:
            report = audit_benchmark_datasets(data_root=args.data_root)
            if args.output:
                write_validation_manifest(report, args.output, data_root=args.data_root,
                                          overwrite=args.overwrite)
            elif args.update_manifest:
                update_dataset_validation(report, data_root=args.data_root)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f"Original benchmark validation: {report['status']}; {report['summary']}")
        for row in report["partitions"]:
            name = "/".join(str(row[key]) for key in ("dataset", "variant", "partition") if row[key])
            if row["X"] is not None:
                X, y = row["X"], row["y"]
                print(f"  {name}: {row['status']}; shape={X['shape']}, X={X['dtype']}, "
                      f"y={y['dtype']}, classes={y['class_counts']}, "
                      f"missing_X={X['missing_values']}, missing_y={y['missing_values']}, "
                      f"sparsity={X['sparsity']:.6%}")
            for error in row["errors"]:
                print(f"  {name}: {error}")
        for error in report["integrity"]["errors"]:
            print(f"  Integrity error: {error}")
        return 0 if report["status"] == "passed" else 1
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
        selected = getattr(args, "instance", None)
        if selected:
            known = {item["id"] for item in config.get("instances", [])}
            if set(selected) - known:
                parser.error("unknown instance selection")
            config["instances"] = [item for item in config["instances"] if item["id"] in selected]
        try:
            validate_verapin_config(config, command=args.command)
            if not args.validate_only:
                from src.experiments.readiness import check_execution_readiness
                check_execution_readiness(config, args.command)
        except ValueError as exc:
            parser.error(str(exc))
        if args.validate_only:
            print("VeraPin configuration valid; no run directory, solver or LLM call created.")
            return 0
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
        from src.utils.serialization import read_json
        return 0 if read_json(Path(output) / "manifest.json")["status"] == "complete" else 1

    parser.error("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
