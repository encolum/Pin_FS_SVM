import pytest

from src.data.loaders import DEFAULT_DATA_ROOT, DATASET_SPECS, audit_datasets, load_dataset


requires_archived_data = pytest.mark.skipif(
    not (DEFAULT_DATA_ROOT / "diabetes.csv").is_file(),
    reason="source-only checkout: restore Dataset/Dataset before running data integration tests",
)


def test_dataset_manifest_matches_manuscript_dimensions():
    assert {
        name: (spec.samples, spec.features, spec.positive, spec.negative)
        for name, spec in DATASET_SPECS.items()
    } == {
        "diabetes": (768, 8, 268, 500),
        "cleveland": (303, 13, 139, 164),
        "wdbc": (569, 30, 212, 357),
        "ionosphere": (351, 34, 225, 126),
        "sonar": (208, 60, 111, 97),
        "colon": (62, 2000, 22, 40),
    }


@requires_archived_data
def test_all_clean_and_archived_files_preserve_expected_shapes():
    report = audit_datasets(include_archived_variants=True)
    assert len(report) == 24
    for row in report:
        spec = DATASET_SPECS[row["dataset"]]
        assert row["samples"] == spec.samples
        assert row["features"] == spec.features


@requires_archived_data
def test_cleveland_combined_does_not_drop_numeric_first_observation():
    X, y = load_dataset("cleveland", "combined")
    assert X.shape == (303, 13)
    assert X[0, 0] == pytest.approx(63.0)
    assert y[0] == -1


@requires_archived_data
def test_clean_class_counts_match_manuscript():
    for dataset, spec in DATASET_SPECS.items():
        _, y = load_dataset(dataset, "clean", validate_classes=True)
        assert int((y == 1).sum()) == spec.positive
        assert int((y == -1).sum()) == spec.negative
