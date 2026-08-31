"""Losslessly curate the supplied benchmark uploads; never remove source files.

Run from the repository root:
    python scripts/curate_datasets.py --source-root /path/to/original/uploads

The output directory must not exist. No download, scaling, feature filtering,
row shuffling, split merging, or pickle execution is performed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickletools
import shutil
import zipfile

import numpy as np
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file


UPLOADS = (
    "BASEHOCK.mat", "colon-cancer.bz2", "gina.npz", "hiva.npz",
    "hill+valley", "madelon", "download.py",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def read_export_labels(path: Path, count: int) -> np.ndarray:
    """Read the literal string list in the supplied NumPy protocol-4 exports.

    pickletools only disassembles bytes. In particular, GLOBAL, REDUCE and BUILD
    are NEVER executed. This intentionally narrow reader rejects other layouts;
    regenerate unsupported exports with numeric labels instead of unpickling.
    """
    with zipfile.ZipFile(path) as archive, archive.open("y.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError("unsupported label NPY version")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if shape != (count,) or fortran or dtype != np.dtype(object):
            raise ValueError("expected a one-dimensional object label export")
        payload = stream.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError("label payload too large")
    operations = [(op.name, value) for op, value, _ in pickletools.genops(payload)]
    list_starts = [i for i, op in enumerate(operations) if op[0] == "EMPTY_LIST"]
    if len(list_starts) != 1 or operations[0] != ("PROTO", 4):
        raise ValueError("unsupported label pickle envelope")
    prefix = operations[:list_starts[0]]
    strings = [v for op, v in prefix if op == "SHORT_BINUNICODE"]
    if strings not in (
        ["numpy._core.multiarray", "_reconstruct", "numpy", "ndarray", "dtype", "O8", "|"],
        ["numpy.core.multiarray", "_reconstruct", "numpy", "ndarray", "dtype", "O8", "|"],
    ):
        raise ValueError("unsupported NumPy label envelope")
    suffix = [("TUPLE", None), ("MEMOIZE", None), ("BUILD", None), ("STOP", None)]
    if operations[-4:] != suffix:
        raise ValueError("unsupported label pickle suffix")
    body = operations[list_starts[0] + 1:-4]
    if not body or body[0] != ("MEMOIZE", None):
        raise ValueError("expected a memoized literal label list")
    next_memo = sum(op == "MEMOIZE" for op, _ in prefix) + 1
    memo: dict[int, str] = {}
    labels: list[int] = []
    batch = False
    literal: str | None = None
    for op, value in body[1:]:
        if op == "MARK" and not batch:
            batch = True
        elif op == "APPENDS" and batch:
            batch = False
        elif op == "SHORT_BINUNICODE" and batch and value in ("-1", "1"):
            literal = value
            labels.append(int(value))
        elif op == "MEMOIZE" and literal is not None:
            memo[next_memo] = literal
            next_memo += 1
            literal = None
        elif op == "BINGET" and batch and value in memo:
            labels.append(int(memo[value]))
        else:
            raise ValueError(f"unsupported opcode in literal label list: {op}")
    if batch or literal is not None or len(labels) != count:
        raise ValueError("incomplete label list or incorrect label count")
    return np.asarray(labels, dtype=np.int8)


def compact_features(X: np.ndarray) -> np.ndarray:
    """Use a smaller integer dtype only when every value round-trips exactly."""
    if X.dtype.kind in "iu" or np.equal(X, np.floor(X)).all():
        for dtype in (np.uint8, np.int16, np.int32):
            limits = np.iinfo(dtype)
            if X.min() >= limits.min and X.max() <= limits.max:
                candidate = X.astype(dtype)
                if np.array_equal(X, candidate):
                    return candidate
    return X


def curate(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for name in UPLOADS:
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    sources = sorted(
        p for name in UPLOADS
        for p in ((source / name).rglob("*") if (source / name).is_dir() else [source / name])
        if p.is_file()
    )
    entries = []
    destinations: dict[str, list[str]] = {}
    output.mkdir(parents=True)

    def save(name, partition, X, y, raw_paths, mapping, note):
        X, y = np.asarray(X), np.asarray(y).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError(f"invalid features for {name}/{partition}")
        if set(np.unique(y)) != {-1, 1}:
            raise ValueError(f"invalid labels for {name}/{partition}")
        stored_X, stored_y = compact_features(X), y.astype(np.int8)
        relative = f"{name}/{partition}.npz"
        target = output / relative
        target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target, X=stored_X, y=stored_y)
        with np.load(target, allow_pickle=False) as check:
            if not (np.array_equal(check["X"], X) and np.array_equal(check["y"], y)):
                raise ValueError(f"conversion changed values for {name}/{partition}")
        entries.append({
            "dataset": name, "partition": partition, "path": relative,
            "samples": len(y), "features": X.shape[1],
            "positive": int((y == 1).sum()), "negative": int((y == -1).sum()),
            "X_dtype": str(stored_X.dtype), "y_dtype": str(stored_y.dtype),
            "bytes": target.stat().st_size, "sha256": digest(target),
            "source_files": raw_paths, "label_mapping": mapping, "note": note,
        })
        for raw in raw_paths:
            destinations.setdefault(raw, []).append(relative)

    base = loadmat(source / "BASEHOCK.mat")
    if set(np.unique(base["Y"])) != {1, 2}:
        raise ValueError("unexpected BASEHOCK source labels")
    save("basehock", "pool", base["X"], np.where(base["Y"] == 1, -1, 1),
         ["BASEHOCK.mat"], {"1": -1, "2": 1}, "No official split in this MAT file.")
    X, y = load_svmlight_file(str(source / "colon-cancer.bz2"), n_features=2000,
                             zero_based=False)
    save("colon_libsvm", "pool", X.toarray(), y, ["colon-cancer.bz2"],
         {"-1": -1, "1": 1},
         "Upstream LIBSVM data are already instance-wise then feature-wise normalized; "
         "NOT raw data for strict train-only preprocessing. Not an alias for manuscript colon.csv.")
    for name in ("gina", "hiva"):
        with np.load(source / f"{name}.npz", allow_pickle=False) as raw:
            X = raw["X"]
        y = read_export_labels(source / f"{name}.npz", len(X))
        save(name, "pool", X, y, [f"{name}.npz"], {"-1": -1, "1": 1},
             "OpenML version 1 combines original training and validation samples. "
             "No original split indices were supplied; no split boundary is inferred.")
    for variant, raw_variant in (("clean", "without_noise"), ("noisy", "with_noise")):
        for partition, raw_partition in (("train", "Training"), ("test", "Testing")):
            raw = f"hill+valley/Hill_Valley_{raw_variant}_{raw_partition}.data"
            data = np.loadtxt(source / raw, delimiter=",", skiprows=1)
            if set(np.unique(data[:, -1])) != {0, 1}:
                raise ValueError("unexpected Hill-Valley labels")
            save(f"hill_valley_{variant}", partition, data[:, :-1],
                 np.where(data[:, -1] == 0, -1, 1), [raw], {"0": -1, "1": 1},
                 f"Official {raw_partition.lower()} partition, {raw_variant}; row order preserved.")
    for partition, suffix in (("train", "train"), ("validation", "valid")):
        raw = f"madelon/MADELON/madelon_{suffix}.data"
        label_path = ("madelon/MADELON/madelon_train.labels" if suffix == "train"
                      else "madelon/madelon_valid.labels")
        save("madelon", partition, np.loadtxt(source / raw),
             np.loadtxt(source / label_path), [raw, label_path], {"-1": -1, "1": 1},
             f"Official {partition} partition. Validation is not renamed as test.")
    metadata = []
    for raw, relative in (
        ("hill+valley/Hill-Valley.names", "hill_valley.names"),
        ("madelon/MADELON/madelon.param", "madelon.param"),
    ):
        target = output / relative
        shutil.copyfile(source / raw, target)
        destinations[raw] = [relative]
        metadata.append({"path": relative, "sha256": digest(target)})
    omitted = {
        "download.py": "superseded by this curation script and documented OpenML IDs",
        "madelon/MADELON/madelon_test.data": "unlabeled test matrix; not needed for supervised runs",
        "madelon/Dataset.pdf": "reference report; linked in README, original retained in backup",
        "hill+valley/Hill_Valley_sample_arff.text": "format example, not a separate benchmark partition",
        "hill+valley/Hill_Valley_visual_examples.jpg": "illustration, not model input",
    }
    inventory = []
    for path in sources:
        relative = path.relative_to(source).as_posix()
        kept = destinations.get(relative, [])
        reason = "represented losslessly in curated files" if kept else omitted.get(relative)
        if reason is None and path.name == ".DS_Store":
            reason = "operating-system metadata"
        if reason is None:
            raise ValueError(f"unreviewed source file: {relative}")
        inventory.append({
            "path": relative, "bytes": path.stat().st_size, "sha256": digest(path),
            "curated_paths": kept, "decision": reason,
        })
    manifest = {
        "schema_version": 1,
        "description": "Lossless numeric benchmark inputs; source originals retained outside git.",
        "preprocessing": "None added: row/column order and all feature values preserved. "
                         "Binary labels mapped explicitly; integer dtypes compacted losslessly.",
        "datasets": entries, "metadata": metadata, "original_inventory": inventory,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    manifest = curate(args.source_root, args.output)
    print(f"Validated lossless conversion of {len(manifest['datasets'])} labeled partitions.")
