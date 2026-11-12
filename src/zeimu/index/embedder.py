"""Turn text into dense vectors: the ``Embedder`` interface and the bge-m3 implementation.

An embedding model maps text to a point in a high-dimensional space (1024 numbers for
bge-m3) such that texts with similar meaning land close together, even across languages.
That is what lets an English question find a Japanese chunk that shares no words with it.

All vectors are L2-normalised (length 1), so cosine similarity is a plain dot product.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

BGE_M3_MODEL = "BAAI/bge-m3"
# Pinned so results are reproducible: an upstream re-upload can't silently change vectors.
# HF commit checked 2026-11-12 at https://huggingface.co/api/models/BAAI/bge-m3.
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class Embedder(Protocol):
    """Anything that embeds text into unit-length float32 vectors of a fixed size."""

    @property
    def model_id(self) -> str:
        """Model name and version, e.g. ``BAAI/bge-m3@5617a9f``; stored with cached vectors."""
        ...

    @property
    def dim(self) -> int:
        """Number of dimensions in every vector."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """One unit-length row per text, shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """One unit-length vector of shape ``(dim,)``."""
        ...


class SentenceTransformerEmbedder:
    """bge-m3 (dense output) run locally through ``sentence-transformers``.

    The model card says bge-m3 needs no instruction prefix on queries (unlike bge v1.5),
    so queries and documents are encoded the same way. The model's own config does the
    pooling (CLS token); we ask the library to normalise.

    Loading takes several seconds and the first run downloads ~2.2 GB to the Hugging Face
    cache, so the model is loaded on first use, not at construction. ``model_id`` needs no
    load, so a stale-cache check can fail fast before paying for it.

    ``device`` defaults to CPU: torch 2.15's Apple-GPU (MPS) backend crashed on macOS 14
    (2026-11-12, see WorkLog), and CPU gives the same vectors on every machine, CI included.
    """

    def __init__(
        self,
        model_name: str = BGE_M3_MODEL,
        *,
        revision: str = BGE_M3_REVISION,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._model_name = model_name
        self._revision = revision
        self._batch_size = batch_size
        self._device = device
        self._model: SentenceTransformer | None = None

    def _loaded(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name, revision=self._revision, device=self._device
            )
        return self._model

    @property
    def model_id(self) -> str:
        return f"{self._model_name}@{self._revision}"

    @property
    def dim(self) -> int:
        dim = self._loaded().get_embedding_dimension()
        if dim is None:
            raise ValueError(f"{self._model_name} does not report an embedding dimension")
        return dim

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._loaded().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > self._batch_size,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self.embed_documents([text])[0]
