"""Machine-readable hardness summaries for cold CPLEX progress profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.serialization import write_csv, write_json


def summarize_hardness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"instances": 0, "nontrivial_instances": 0}
    successful = [row for row in rows if row.get("solver_status") in {"optimal", "feasible_with_gap"}
                  and row.get("final_objective") is not None]
    nontrivial = [
        row
        for row in successful
        if float(row.get("final_gap") or 0.0) > 0
        or int(row.get("node_count") or 0) > 0
        or row.get("first_feasible_time") is None
    ]
    return {
        "instances": len(rows),
        "successful_instances": len(successful),
        "failed_instances": len(rows) - len(successful),
        "nontrivial_instances": len(nontrivial),
        "nontrivial_instance_ids": [row["instance_id"] for row in nontrivial],
        "mean_runtime": float(np.mean([float(row["total_runtime"]) for row in rows])),
        "mean_final_gap": float(np.mean([float(row["final_gap"]) for row in successful
                                          if row.get("final_gap") is not None]))
                          if any(row.get("final_gap") is not None for row in successful) else None,
        "total_nodes": int(sum(int(row.get("node_count") or 0) for row in rows)),
        "go_no_go_requires_author_review": True,
    }


def write_solver_profiles(run_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    run_dir = Path(run_dir)
    write_csv(run_dir / "solver_profiles.csv", rows)
    write_json(run_dir / "hardness_summary.json", summarize_hardness(rows))
