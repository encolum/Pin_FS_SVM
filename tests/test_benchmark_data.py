from hashlib import sha256
from io import BytesIO
import json
import pickle
import zipfile

import numpy as np
import pytest

from scripts.curate_datasets import compact_features, curate, read_export_labels
from src.data import audit_benchmark_datasets, load_benchmark_dataset
from src.data.benchmarks import DEFAULT_BENCHMARK_ROOT


PARTITIONS = [
    ("basehock", "pool", 1993, 4862, 999, 994),
    ("colon_libsvm", "pool", 62, 2000, 22, 40),
    ("gina", "pool", 3468, 970, 1705, 1763),
    ("hiva", "pool", 4229, 1617, 149, 4080),
    ("hill_valley_clean", "train", 606, 100, 301, 305),
    ("hill_valley_clean", "test", 606, 100, 311, 295),
    ("hill_valley_noisy", "train", 606, 100, 299, 307),
    ("hill_valley_noisy", "test", 606, 100, 307, 299),
    ("madelon", "train", 2000, 500, 1000, 1000),
    ("madelon", "validation", 600, 500, 300, 300),
]


@pytest.mark.parametrize("name,partition,n,p,positive,negative", PARTITIONS)
def test_committed_partition_shape_labels_and_integrity(name, partition, n, p, positive, negative):
    X, y = load_benchmark_dataset(name, partition=partition)
    assert X.shape == (n, p)
    assert y.shape == (n,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert np.isfinite(X).all()
    assert int((y == 1).sum()) == positive
    assert int((y == -1).sum()) == negative


def test_manifest_has_no_unaccounted_data_or_metadata():
    root = DEFAULT_BENCHMARK_ROOT
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {r["path"] for r in manifest["datasets"] + manifest["metadata"]}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert actual == expected | {"manifest.json", "README.md"}
    assert len(audit_benchmark_datasets()) == 10
    inventory = {r["path"]: r for r in manifest["original_inventory"]}
    for entry in manifest["datasets"]:
        for raw_path in entry["source_files"]:
            assert entry["path"] in inventory[raw_path]["curated_paths"]
    assert not inventory["madelon/MADELON/madelon_test.data"]["curated_paths"]


def test_partitions_must_be_explicit_and_test_labels_are_not_fabricated():
    with pytest.raises(TypeError):
        load_benchmark_dataset("gina")
    with pytest.raises(KeyError):
        load_benchmark_dataset("madelon", partition="test")
    with pytest.raises(KeyError):
        load_benchmark_dataset("colon", partition="pool")
    with pytest.raises(KeyError):
        load_benchmark_dataset("hill_valley_clean", partition="pool")


def test_hill_valley_variants_and_partitions_remain_distinct():
    clean, _ = load_benchmark_dataset("hill_valley_clean", partition="train")
    noisy, _ = load_benchmark_dataset("hill_valley_noisy", partition="train")
    test, _ = load_benchmark_dataset("hill_valley_clean", partition="test")
    assert not np.array_equal(clean, noisy)
    assert not np.array_equal(clean, test)


def _fixture(root, X=None, y=None):
    X = np.array([[0, 255], [1, 2]], dtype=np.uint8) if X is None else X
    y = np.array([-1, 1], dtype=np.int8) if y is None else y
    path = root / "pool.npz"
    np.savez_compressed(path, X=X, y=y)
    manifest = {"schema_version": 1, "metadata": [], "datasets": [{
        "dataset": "tiny", "partition": "pool", "path": "pool.npz",
        "samples": 2, "features": 2, "positive": 1, "negative": 1,
        "X_dtype": str(X.dtype), "y_dtype": str(y.dtype),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_tampered_file_is_rejected_before_loading(tmp_path):
    _fixture(tmp_path)
    with (tmp_path / "pool.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_benchmark_dataset("tiny", partition="pool", data_root=tmp_path)


@pytest.mark.parametrize("X,y", [
    (np.zeros((3, 2)), None),
    (np.array([[np.nan, 1], [0, 1]]), None),
    (np.array([[np.inf, 1], [0, 1]]), None),
    (None, np.array([0, 1], dtype=np.int8)),
    (None, np.array([1, 1], dtype=np.int8)),
    (None, np.array([-1.0, 1.0])),
    (None, np.array(["-1", "1"], dtype=object)),
])
def test_invalid_arrays_rejected_even_when_hash_matches(tmp_path, X, y):
    _fixture(tmp_path, X, y)
    with pytest.raises(ValueError):
        load_benchmark_dataset("tiny", partition="pool", data_root=tmp_path)


def test_manifest_cannot_escape_data_root(tmp_path):
    manifest = _fixture(tmp_path)
    manifest["datasets"][0]["path"] = "../outside.npz"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes data root"):
        load_benchmark_dataset("tiny", partition="pool", data_root=tmp_path)


def test_class_counts_are_checked(tmp_path):
    manifest = _fixture(tmp_path)
    manifest["datasets"][0]["positive"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="class counts mismatch"):
        load_benchmark_dataset("tiny", partition="pool", data_root=tmp_path)


def test_original_string_labels_are_decoded_without_unpickling(tmp_path, monkeypatch):
    path = tmp_path / "export.npz"
    labels = ["-1", "1", "1", "-1"] * 600  # crosses multiple pickle list batches
    # Match the uploaded protocol-4 format even on NumPy versions that export
    # object arrays with a different default pickle protocol.
    stream = BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        "shape": (len(labels),), "fortran_order": False, "descr": "|O",
    })
    stream.write(pickle.dumps(np.array(labels, dtype=object), protocol=4))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("y.npy", stream.getvalue())

    def forbidden(*args, **kwargs):
        raise AssertionError("pickle execution is forbidden")

    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    result = read_export_labels(path, len(labels))
    np.testing.assert_array_equal(result, np.array(labels, dtype=np.int8))


@pytest.mark.parametrize("labels", [["-1", "unknown"], ["-1", {}]])
def test_unsupported_object_label_payload_rejected(tmp_path, labels):
    path = tmp_path / "export.npz"
    np.savez(path, y=np.array(labels, dtype=object))
    with pytest.raises(ValueError):
        read_export_labels(path, len(labels))


@pytest.mark.parametrize("X", [
    np.array([[0, 255]], dtype=np.int64),
    np.array([[-200, 1200]], dtype=float),
    np.array([[0.125, -1.123456789]], dtype=float),
])
def test_dtype_compaction_preserves_every_value(X):
    np.testing.assert_array_equal(compact_features(X), X)


def test_curation_never_overwrites_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        curate(tmp_path, tmp_path)


def test_data_validation_cli_does_not_launch_experiments(monkeypatch, capsys):
    import main

    def forbidden(*args, **kwargs):
        raise AssertionError("validation must not train")

    monkeypatch.setattr(main, "run_experiment", forbidden)
    assert main.main(["validate-datasets"]) == 0
    assert "Validated 10 curated partitions" in capsys.readouterr().out
