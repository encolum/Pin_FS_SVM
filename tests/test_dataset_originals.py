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
    assert MANIFEST["schema_version"] == 2
    assert MANIFEST["representation"] == "original_uploads"
    assert MANIFEST["transformation"] == "none"
    paths = [entry["path"] for entry in MANIFEST["files"]]
    assert len(paths) == len(set(paths)) == 19
    expected = set(paths) | {"README.md", "manifest.json"}
    ignored = {entry["path"] for entry in MANIFEST["excluded_from_git"]}
    actual = {path.relative_to(DATA_ROOT).as_posix()
              for path in DATA_ROOT.rglob("*") if path.is_file()}
    assert actual - ignored == expected
    assert "madelon/MADELON/madelon_test.data" in paths
    assert "hill+valley/Hill_Valley_sample_arff.text" in paths
    assert "download.py" in paths
