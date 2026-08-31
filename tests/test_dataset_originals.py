"""Verify original upload bytes without parsing, transforming or unpickling data."""

from hashlib import sha256
import json
from pathlib import Path

import pytest


DATA_ROOT = Path(__file__).resolve().parents[1] / "dataset"
MANIFEST = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", MANIFEST["files"], ids=lambda entry: entry["path"])
def test_original_upload_bytes_unchanged(entry):
    path = DATA_ROOT / entry["path"]
    assert path.is_file()
    assert path.stat().st_size == entry["bytes"]
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == entry["sha256"]


def test_inventory_contains_originals_only():
    assert MANIFEST["schema_version"] == 3
    assert MANIFEST["representation"] == "original_uploads"
    assert MANIFEST["transformation"] == "none"
    paths = [entry["path"] for entry in MANIFEST["files"]]
    assert len(paths) == len(set(paths)) == 10
    expected = {
        "BASEHOCK.mat", "colon-cancer.bz2", "gina.npz", "hiva.npz",
        "hill_valley/train.data", "hill_valley/test.data",
        "madelon/train.data", "madelon/train.labels", "madelon/valid.data", "madelon/valid.labels",
        "README.md", "manifest.json",
    }
    assert set(paths) | {"README.md", "manifest.json"} == expected
    actual = {path.relative_to(DATA_ROOT).as_posix()
              for path in DATA_ROOT.rglob("*") if path.is_file()}
    assert actual == expected
    directories = {p.relative_to(DATA_ROOT).as_posix() for p in DATA_ROOT.rglob("*") if p.is_dir()}
    assert directories == {"hill_valley", "madelon"}
    assert {row["original_path"] for row in MANIFEST["removed_files"]} == {
        "download.py", "hill+valley/Hill-Valley.names",
        "hill+valley/Hill_Valley_sample_arff.text", "hill+valley/Hill_Valley_visual_examples.jpg",
        "hill+valley/Hill_Valley_with_noise_Training.data", "hill+valley/Hill_Valley_with_noise_Testing.data",
        "madelon/Dataset.pdf", "madelon/MADELON/madelon.param", "madelon/MADELON/madelon_test.data",
    }
