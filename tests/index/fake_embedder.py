"""A tiny deterministic ``Embedder`` so tests never download or run the real model.

Each text becomes a bag of character bigrams hashed into ``dim`` buckets, then
normalised. Texts that share bigrams point in similar directions, which is enough to
test ranking, caching and wiring. It knows nothing about meaning or languages.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class FakeEmbedder:
    """Bigram-hash embedder; records every call so tests can check what got embedded."""

    def __init__(self, dim: int = 64, model_id: str = "fake@1") -> None:
        self._dim = dim
        self._model_id = model_id
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self._dim, dtype=np.float64)
        padded = f" {text} "
        for i in range(len(padded) - 1):
            digest = hashlib.blake2b(padded[i : i + 2].encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self._dim] += 1.0
        return (vector / np.linalg.norm(vector)).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.document_calls.append(list(texts))
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.query_calls.append(text)
        return self._vector(text)
