"""Exact nearest-neighbour search over unit vectors with NumPy.

With every row normalised to length 1, cosine similarity reduces to a dot product, so
scoring all chunks is one matrix-vector multiply: ``scores = M @ q``. For a few hundred
chunks that takes microseconds, so an approximate index (FAISS, HNSW, pgvector) would
add a dependency and give up exactness for no measurable gain.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# float32 rounding leaves norms a little off 1.0; anything further means "not normalised".
UNIT_NORM_TOLERANCE = 1e-3


def check_unit_rows(matrix: NDArray[np.floating], *, what: str = "vectors") -> None:
    """Raise ``ValueError`` unless ``matrix`` is a non-empty, finite 2-D array of unit rows."""
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{what} must be a non-empty 2-D array, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{what} contain NaN or infinite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=UNIT_NORM_TOLERANCE):
        raise ValueError(
            f"{what} must be L2-normalised (row norms {norms.min():.4f}..{norms.max():.4f})"
        )


class VectorIndex:
    """An immutable in-memory matrix of unit vectors, one row per chunk."""

    def __init__(self, vectors: NDArray[np.floating]) -> None:
        check_unit_rows(vectors)
        matrix = np.array(vectors, dtype=np.float32)  # private copy
        matrix.setflags(write=False)
        self._matrix = matrix

    def __len__(self) -> int:
        return self._matrix.shape[0]

    @property
    def dim(self) -> int:
        """Number of dimensions in each vector."""
        return self._matrix.shape[1]

    def scores(self, query: NDArray[np.floating]) -> NDArray[np.float64]:
        """Cosine similarity of ``query`` with every row (index = chunk position), in [-1, 1]."""
        if query.shape != (self.dim,):
            raise ValueError(f"query must have shape ({self.dim},), got {query.shape}")
        check_unit_rows(query.reshape(1, -1), what="query vector")
        return (self._matrix @ query.astype(np.float32)).astype(np.float64)
