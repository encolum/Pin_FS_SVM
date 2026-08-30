"""Render compact Markdown summaries from aggregate CSV files."""

from __future__ import annotations

import csv
from pathlib import Path


def create_markdown_table(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    source = run_dir / "aggregate" / "summary.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lines = [
        "# Outer-test performance summary", "",
        "| Dataset | Condition | Model | Balanced Accuracy | Weighted F1 | Accuracy | G-mean |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['model']} | "
            f"{float(row['mean_balanced_accuracy']):.4f} | {float(row['mean_weighted_f1']):.4f} | "
            f"{float(row['mean_accuracy']):.4f} | {float(row['mean_gmean']):.4f} |"
        )
    output = run_dir / "tables" / "summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
