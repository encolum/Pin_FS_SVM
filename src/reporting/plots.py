"""Balanced-accuracy plots generated only from saved numerical results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def plot_balanced_accuracy(run_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    with (run_dir / "aggregate" / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append((f"{row['dataset']}\n{row['condition']}", float(row["mean_balanced_accuracy"])))
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.35), 5))
    offset = 0
    labels: list[str] = []
    for model, values in sorted(grouped.items()):
        xs = list(range(offset, offset + len(values)))
        axis.plot(xs, [value for _, value in values], marker="o", label=model)
        labels.extend(label for label, _ in values)
        offset += len(values)
    axis.set_ylabel("Balanced Accuracy")
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    output = run_dir / "figures" / "balanced_accuracy.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
