"""Read and checksum-verify the six retained benchmark file formats.

This module preserves native values, labels, partitions, and sparse storage. It
does not map labels, merge partitions, preprocess data, apply corruption, or
write source files.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickletools
import zipfile

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file


# Safe reader for the literal object labels in the retained GINA/HIVA exports.
def read_original_npz_labels(path: Path, count: int) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["X.npy", "y.npy"]:
            raise ValueError("expected exactly X.npy and y.npy in original export")
        with archive.open("y.npy") as stream:
            if np.lib.format.read_magic(stream) != (1, 0):
                raise ValueError("unsupported label NPY version")
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            if shape != (count,) or fortran or dtype != np.dtype(object):
                raise ValueError("expected the original one-dimensional object labels")
            payload = stream.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError("label payload exceeds supported size")
    try:
        disassembly = list(pickletools.genops(payload))
    except Exception as exc:
        raise ValueError("malformed label payload") from exc
    if not disassembly or disassembly[-1][2] + 1 != len(payload):
        raise ValueError("truncated or trailing label payload")
    operations = [(op.name, value) for op, value, _ in disassembly]
    count_opcode = "BININT1" if count < 256 else "BININT2" if count < 65536 else "BININT"
    # The only variable envelope fields are the NumPy module spelling, array
    # length, and frame byte length. These symbols are compared, never invoked.
    prefix = [
        ("PROTO", 4), ("FRAME", len(payload) - 11),
        ("SHORT_BINUNICODE", "numpy._core.multiarray"), ("MEMOIZE", None),
        ("SHORT_BINUNICODE", "_reconstruct"), ("MEMOIZE", None),
        ("STACK_GLOBAL", None), ("MEMOIZE", None),
        ("SHORT_BINUNICODE", "numpy"), ("MEMOIZE", None),
        ("SHORT_BINUNICODE", "ndarray"), ("MEMOIZE", None),
        ("STACK_GLOBAL", None), ("MEMOIZE", None),
        ("BININT1", 0), ("TUPLE1", None), ("MEMOIZE", None),
        ("SHORT_BINBYTES", b"b"), ("MEMOIZE", None), ("TUPLE3", None),
        ("MEMOIZE", None), ("REDUCE", None), ("MEMOIZE", None),
        ("MARK", None), ("BININT1", 1), (count_opcode, count),
        ("TUPLE1", None), ("MEMOIZE", None), ("BINGET", 3),
        ("SHORT_BINUNICODE", "dtype"), ("MEMOIZE", None),
        ("STACK_GLOBAL", None), ("MEMOIZE", None),
        ("SHORT_BINUNICODE", "O8"), ("MEMOIZE", None),
        ("NEWFALSE", None), ("NEWTRUE", None), ("TUPLE3", None),
        ("MEMOIZE", None), ("REDUCE", None), ("MEMOIZE", None),
        ("MARK", None), ("BININT1", 3), ("SHORT_BINUNICODE", "|"),
        ("MEMOIZE", None), ("NONE", None), ("NONE", None), ("NONE", None),
        ("BININT", -1), ("BININT", -1), ("BININT1", 63), ("TUPLE", None),
        ("MEMOIZE", None), ("BUILD", None), ("NEWFALSE", None),
        ("EMPTY_LIST", None), ("MEMOIZE", None),
    ]
    if len(operations) > 2 and operations[2] == ("SHORT_BINUNICODE", "numpy.core.multiarray"):
        prefix[2] = operations[2]
    suffix = [("TUPLE", None), ("MEMOIZE", None), ("BUILD", None), ("STOP", None)]
    if operations[:len(prefix)] != prefix or operations[-4:] != suffix:
        raise ValueError("unsupported NumPy label envelope; no pickle fallback is permitted")
    next_memo = sum(op == "MEMOIZE" for op, _ in prefix)
    memo: dict[int, str] = {}
    labels: list[str] = []
    batch = False
    literal: str | None = None
    for op, value in operations[len(prefix):-4]:
        if literal is not None and op != "MEMOIZE":
            raise ValueError("expected memoized label literal")
        if op == "MARK" and not batch:
            batch = True
        elif op == "APPENDS" and batch:
            batch = False
        elif op == "SHORT_BINUNICODE" and batch and value in ("-1", "1"):
            literal = value
            labels.append(value)
        elif op == "MEMOIZE" and batch and literal is not None:
            memo[next_memo] = literal
            next_memo += 1
            literal = None
        elif op == "BINGET" and batch and value in memo:
            labels.append(memo[value])
        else:
            raise ValueError(f"unsupported literal-label opcode: {op}")
    if batch or literal is not None or len(labels) != count:
        raise ValueError("incomplete or incorrect label count")
    # Retain object dtype and string values, as in the original y.npy.
    return np.asarray(labels, dtype=object)


# Original benchmark loading.
DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "dataset"


@dataclass(frozen=True)
class RawBenchmarkDataset:
    dataset: str
    partition: str
    variant: str | None
    X: np.ndarray | sparse.spmatrix
    y: np.ndarray | None
    source_files: tuple[dict, ...]
    source_y_shape: tuple[int, ...] | None
    source_format: str
    dtype_origin: str
    warnings: tuple[str, ...] = ()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_inventory(data_root: str | Path | None = None) -> tuple[Path, dict[str, dict]]:
    root = Path(data_root) if data_root is not None else DEFAULT_BENCHMARK_ROOT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 3
            or manifest.get("representation") != "original_uploads"
            or manifest.get("transformation") != "none"):
        raise ValueError("expected the original-upload integrity manifest")
    rows = manifest["files"]
    inventory = {row["path"]: row for row in rows}
    if len(inventory) != len(rows):
        raise ValueError("duplicate original file in manifest")
    return root, inventory


def verified_source(root: Path, inventory: dict[str, dict], relative: str) -> tuple[Path, dict]:
    if relative not in inventory:
        raise ValueError(f"source is not inventoried: {relative}")
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("source path escapes dataset root")
    entry = inventory[relative]
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"original source checksum/size mismatch: {relative}")
    return path, dict(entry)


def _sources(data_root, *relatives):
    root, inventory = original_inventory(data_root)
    verified = [verified_source(root, inventory, relative) for relative in relatives]
    return [pair[0] for pair in verified], tuple(pair[1] for pair in verified)


def _bundle(dataset, partition, X, y, sources, source_format, dtype_origin, *, variant=None, warnings=()):
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or X.dtype.kind not in "iuf":
        raise ValueError(f"{dataset}: expected a nonempty numeric feature matrix")
    source_y_shape = None if y is None else tuple(y.shape)
    if y is not None:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)  # interface shape only; values/dtype/order unchanged
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError(f"{dataset}: feature/label row mismatch")
    return RawBenchmarkDataset(dataset, partition, variant, X, y, sources,
                               source_y_shape, source_format, dtype_origin, tuple(warnings))


def load_basehock(*, data_root=None) -> RawBenchmarkDataset:
    paths, sources = _sources(data_root, "BASEHOCK.mat")
    data = loadmat(paths[0])
    if "X" not in data or "Y" not in data:
        raise ValueError("BASEHOCK MAT must contain X and Y")
    return _bundle("basehock", "pool", data["X"], data["Y"], sources, "mat_v5", "stored", warnings=(
        "No official split indices supplied; labels remain 1/2, not solver-ready -1/+1.",
    ))


def load_colon(*, data_root=None) -> RawBenchmarkDataset:
    paths, sources = _sources(data_root, "colon-cancer.bz2")
    # Explicit one-based feature indices, but infer width so validation detects
    # missing/extra columns instead of silently padding to the expected width.
    X, y = load_svmlight_file(str(paths[0]), zero_based=False, dtype=np.float64)
    return _bundle("colon", "pool", X, y, sources, "libsvm_bzip2", "parsed_float64", warnings=(
        "LIBSVM export is already normalized upstream; not raw for a train-only-scaling claim.",
        "This loader is separate from the manuscript colon.csv loader.",
    ))


def _load_openml_export(name, data_root):
    paths, sources = _sources(data_root, f"{name}.npz")
    with np.load(paths[0], allow_pickle=False) as archive:
        if set(archive.files) != {"X", "y"}:
            raise ValueError(f"{name}: expected X and y arrays")
        X = archive["X"]
    y = read_original_npz_labels(paths[0], len(X))
    return _bundle(name, "pool", X, y, sources, "numpy_npz", "stored", warnings=(
        "OpenML export combines original train and validation; no split indices inferred.",
        "Object labels remain strings '-1'/'1'; parsed as literals without executing pickle.",
    ))


def load_gina(*, data_root=None) -> RawBenchmarkDataset:
    return _load_openml_export("gina", data_root)


def load_hiva(*, data_root=None) -> RawBenchmarkDataset:
    return _load_openml_export("hiva", data_root)


def load_hill_valley(*, partition: str, variant: str = "without_noise", data_root=None) -> RawBenchmarkDataset:
    if variant != "without_noise" or partition not in {"train", "test"}:
        raise ValueError("Hill-Valley retains only without_noise; partition must be train/test")
    relative = f"hill_valley/{partition}.data"
    paths, sources = _sources(data_root, relative)
    with paths[0].open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
    if header != [f"X{i}" for i in range(1, 101)] + ["class"]:
        raise ValueError("unexpected Hill-Valley feature/label header")
    data = np.loadtxt(paths[0], delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if data.shape[1] != len(header):
        raise ValueError("Hill-Valley column count disagrees with its header")
    return _bundle("hill_valley", partition, data[:, :-1], data[:, -1], sources,
                   "csv", "parsed_float64", variant=variant, warnings=(
                       "Official partition preserved; labels remain 0/1, not solver-ready -1/+1.",
                       "Only the original without-noise variant is retained; no noise is generated by this loader.",
                   ))


def load_madelon(*, partition: str, data_root=None) -> RawBenchmarkDataset:
    suffixes = {"train": "train", "validation": "valid"}
    if partition not in suffixes:
        raise ValueError("Madelon retains only labeled train/validation partitions")
    relatives = (f"madelon/{suffixes[partition]}.data", f"madelon/{suffixes[partition]}.labels")
    paths, sources = _sources(data_root, *relatives)
    X = np.loadtxt(paths[0], dtype=np.float64, ndmin=2)
    y = np.loadtxt(paths[1], dtype=np.float64, ndmin=1)
    warning = "Official labeled partition preserved; train/validation are not automatically merged."
    return _bundle("madelon", partition, X, y, sources, "whitespace_matrix", "parsed_float64",
                   warnings=(warning,))


BENCHMARK_LOADERS = {
    "basehock": load_basehock, "colon": load_colon, "gina": load_gina,
    "hiva": load_hiva, "hill_valley": load_hill_valley, "madelon": load_madelon,
}


def load_benchmark_dataset(dataset: str, *, partition: str | None = None,
                           variant: str | None = None, data_root=None) -> RawBenchmarkDataset:
    """Dispatch to one of six loaders; split/variant choices are never inferred."""
    if dataset not in BENCHMARK_LOADERS:
        raise KeyError(f"unknown original benchmark: {dataset}")
    if dataset == "hill_valley":
        return load_hill_valley(variant="without_noise" if variant is None else variant,
                                partition=partition, data_root=data_root)
    if variant is not None:
        raise ValueError(f"{dataset} has no variant argument")
    if dataset == "madelon":
        return load_madelon(partition=partition, data_root=data_root)
    if partition not in (None, "pool"):
        raise ValueError(f"{dataset} has no supplied {partition} partition; use pool")
    return BENCHMARK_LOADERS[dataset](data_root=data_root)

