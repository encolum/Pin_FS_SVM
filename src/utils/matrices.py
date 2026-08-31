"""Sparse-preserving validation, memory accounting and deterministic hashes."""

from hashlib import sha256
import numpy as np
from scipy import sparse


def numeric_matrix(X):
    X = sparse.csr_matrix(X, dtype=float, copy=True) if sparse.issparse(X) else np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    if sparse.issparse(X):
        X.sum_duplicates()
        X.sort_indices()
    if not np.isfinite(X.data if sparse.issparse(X) else X).all():
        raise ValueError("X contains NaN or infinite values")
    return X


def estimate_dense_bytes(X, dtype=np.float64):
    return int(X.shape[0]) * int(X.shape[1]) * np.dtype(dtype).itemsize


def guarded_dense(X, *, allow_densify=False, max_dense_bytes=None):
    if not sparse.issparse(X):
        return np.asarray(X, dtype=float)
    if allow_densify is not True or type(max_dense_bytes) is not int or max_dense_bytes <= 0:
        raise ValueError("sparse densification requires allow_densify=true and explicit max_dense_bytes")
    if estimate_dense_bytes(X) > max_dense_bytes:
        raise ValueError(f"densification exceeds max_dense_bytes: {estimate_dense_bytes(X)} > {max_dense_bytes}")
    return X.astype(float).toarray()


def matrix_metadata(X):
    is_sparse = sparse.issparse(X)
    nnz = int(X.count_nonzero()) if is_sparse else int(np.count_nonzero(X))
    sparse_bytes = int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) if is_sparse else 0
    return {"shape": list(X.shape), "dtype": str(X.dtype), "storage": "csr" if is_sparse else "dense",
            "nnz": nnz, "density": nnz / (int(X.shape[0]) * int(X.shape[1])),
            "input_sparse_bytes": sparse_bytes, "estimated_dense_bytes": estimate_dense_bytes(X),
            "matrix_bytes": sparse_bytes if is_sparse else int(X.nbytes)}


def update_array_hash(digest, X):
    if sparse.issparse(X):
        X = sparse.csr_matrix(X, copy=True)
        X.sum_duplicates()
        X.eliminate_zeros()
        X.sort_indices()
        digest.update(b"csr")
        digest.update(repr(X.shape).encode())
        for values in (X.indptr.astype(np.int64), X.indices.astype(np.int64), X.data):
            update_array_hash(digest, values)
    else:
        values = np.ascontiguousarray(X)
        digest.update(repr((values.shape, values.dtype.str)).encode())
        digest.update(values.view(np.uint8))


def data_hash(X, y):
    digest = sha256()
    update_array_hash(digest, X)
    update_array_hash(digest, y)
    return digest.hexdigest()
