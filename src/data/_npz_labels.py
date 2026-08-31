"""Data-only reader for the literal string labels in the uploaded NPZ files.

This is deliberately not a general pickle loader. It recognizes the NumPy
protocol-4 envelope in these exports and reads only a literal list of strings.
No GLOBAL, REDUCE, BUILD, or other pickle operation is ever executed.
"""

from pathlib import Path
import pickletools
import zipfile

import numpy as np


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
