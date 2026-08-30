import numpy as np
from statsmodels.stats.multitest import multipletests

from src.statistics.wilcoxon import benjamini_hochberg, run_wilcoxon_analysis
from src.utils.serialization import read_json, write_csv, write_json


def test_bh_matches_statsmodels():
    raw = [0.01, 0.04, 0.03, 0.2, 0.5]
    expected = multipletests(raw, method="fdr_bh")[1]
    assert np.allclose(benjamini_hochberg(raw), expected)


def test_wilcoxon_pipeline_uses_configured_metric_and_saves_family(tmp_path):
    rows = []
    for fold, proposed, baseline in ((1, 0.9, 0.6), (2, 0.8, 0.5), (3, 0.85, 0.55)):
        common = {"dataset": "toy", "condition": "clean", "outer_fold": fold, "random_seed": 42}
        rows.append({**common, "model": "pin_fs_svm", "balanced_accuracy": proposed, "accuracy": 0.5})
        rows.append({**common, "model": "l1_svm", "balanced_accuracy": baseline, "accuracy": 0.5})
    write_csv(tmp_path / "metrics" / "fold_metrics.csv", rows)
    write_json(
        tmp_path / "config.yaml",
        {"statistics": {"metric": "balanced_accuracy", "alternative": "greater"}},
    )
    output = run_wilcoxon_analysis(tmp_path)
    assert output.is_file()
    metadata = read_json(tmp_path / "aggregate" / "wilcoxon_metadata.json")
    assert metadata["settings"]["metric"] == "balanced_accuracy"
    assert metadata["correction_family"] == [{"dataset": "toy", "baseline": "l1_svm"}]
