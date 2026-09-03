"""Flatten kernel-search results into the VeraPin route/iteration schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.search.progress import solver_progress_summary
from src.search.progress import SolverProgressRecord
from src.search.states import KernelSearchResult
from src.utils.serialization import write_csv, write_json


def kernel_search_result_row(
    result: KernelSearchResult,
    *,
    route: str,
    instance_id: str,
    classification: dict[str, Any] | None = None,
    reference_objective: float | None = None,
) -> dict[str, Any]:
    trajectory = [
        SolverProgressRecord(**record)
        for record in result.metadata.get("route_progress", [])
    ]
    progress = solver_progress_summary(
        trajectory,
        horizon=result.total_runtime,
        reference_objective=reference_objective,
    )
    final_record = result.history[-1] if result.history else {}
    row = {
        "instance_id": instance_id,
        "route": route,
        "kernel_policy": result.method,
        "initial_kernel_size": len(result.initial_kernel),
        "final_kernel_size": len(result.final_kernel),
        "iterations": len(result.history),
        "improving_iterations": int(result.metadata.get("improving_iterations", 0)),
        "restricted_solves": int(result.metadata.get("restricted_solves", 0)),
        **progress,
        "final_objective": result.best_result.objective,
        "node_count": result.metadata.get(
            "total_node_count",
            final_record.get("node_count", result.best_result.diagnostics.node_count),
        ),
        "total_runtime": result.total_runtime,
        "signal_overhead": float(result.metadata.get("signal_overhead", 0.0)),
        "policy_overhead": float(result.metadata.get("policy_overhead", 0.0)),
        "lp_relaxation_overhead": float(
            result.metadata.get("lp_relaxation_overhead", 0.0)
        ),
        "mip_start_status": final_record.get(
            "mip_start_status", result.best_result.mip_start_status
        ),
        "solver_status": final_record.get(
            "solver_status", result.best_result.diagnostics.status
        ),
        "selected_feature_count": len(result.best_result.support),
        "selected_feature_indices": sorted(result.best_result.support),
    }
    if classification:
        row["classification_scope"] = "outer_test"
        row.update(classification)
    return row


def write_kernel_search_results(
    run_dir: str | Path,
    route_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    details: dict[str, Any],
) -> None:
    run_dir = Path(run_dir)
    write_csv(run_dir / "route_results.csv", route_rows)
    write_csv(run_dir / "iteration_results.csv", iteration_rows)
    write_json(run_dir / "search_details.json", details)
    lines = [
        "# Kernel-search summary",
        "",
        "| Instance | Route | Objective | Gap | Runtime | Initial K | Final K |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in route_rows:
        gap = row.get("final_gap")
        gap_text = "n/a" if gap is None else f"{float(gap):.6g}"
        lines.append(
            f"| {row['instance_id']} | {row['route']} | "
            f"{float(row['final_objective']):.8g} | {gap_text} | "
            f"{float(row['total_runtime']):.4f} | {row['initial_kernel_size']} | "
            f"{row['final_kernel_size']} |"
        )
    (run_dir / "kernel_search_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
