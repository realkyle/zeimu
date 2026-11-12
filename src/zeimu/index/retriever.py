"""The retrieval interface shared by every search method, and its BM25 and embedding versions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from zeimu.index.bm25 import BM25Index
from zeimu.index.embedder import Embedder
from zeimu.index.tokenizer import Tokenizer
from zeimu.index.vector_cache import CacheMeta, StaleCacheError, VectorCache
from zeimu.index.vectors import VectorIndex
from zeimu.ingest.chunker import Chunk


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result. ``rank`` starts at 1; ``score`` is only comparable within a method."""

    chunk_id: str
    score: float
    rank: int


class Retriever(Protocol):
    """Anything that can search the chunks: BM25 now, embeddings and the hybrid later."""

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Return up to ``k`` hits, best first.

        Keyword search leaves out chunks that share no term with the query; embedding
        search always returns ``min(k, chunks)`` hits, because every chunk has a cosine.
        """
        ...


class BM25Retriever:
    """Keyword search: tokenizes chunks once at construction, then scores each query."""

    def __init__(self, chunks: Sequence[Chunk], tokenizer: Tokenizer) -> None:
        self._ids = [chunk.id for chunk in chunks]
        self._tokenizer = tokenizer
        self._index = BM25Index([tokenizer.terms(chunk.text) for chunk in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Top ``k`` chunks by BM25 score; ties keep corpus order so results are stable."""
        if k < 1:
            raise ValueError("k must be at least 1")
        scores = self._index.scores(self._tokenizer.terms(query))
        order = np.argsort(-scores, kind="stable")
        hits: list[Hit] = []
        for position in order[:k]:
            score = float(scores[position])
            if score <= 0:
                break
            hits.append(Hit(chunk_id=self._ids[position], score=score, rank=len(hits) + 1))
        return hits


class EmbeddingRetriever:
    """Semantic search: embeds the query and ranks chunks by cosine similarity.

    Chunk vectors are made ahead of time (``zeimu embed``) and passed in, so building a
    retriever doesn't re-embed the corpus. Row ``i`` of ``vectors`` must be ``chunks[i]``.
    """

    def __init__(
        self, chunks: Sequence[Chunk], embedder: Embedder, vectors: NDArray[np.floating]
    ) -> None:
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")
        self._ids = [chunk.id for chunk in chunks]
        self._embedder = embedder
        self._index = VectorIndex(vectors)
        if self._index.dim != embedder.dim:
            raise ValueError(f"vectors have {self._index.dim} dims, embedder makes {embedder.dim}")

    @classmethod
    def build(
        cls, chunks: Sequence[Chunk], embedder: Embedder, cache: VectorCache
    ) -> EmbeddingRetriever:
        """Embed every chunk, save the vectors to ``cache``, and return a retriever."""
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        cache.save(
            vectors, CacheMeta.for_chunks(chunks, model_id=embedder.model_id, dim=embedder.dim)
        )
        return cls(chunks, embedder, vectors)

    @classmethod
    def from_cache(
        cls, chunks: Sequence[Chunk], embedder: Embedder, cache: VectorCache
    ) -> EmbeddingRetriever:
        """Load saved vectors; raises if the cache is missing or stale (see ``VectorCache``)."""
        vectors = cache.load(chunks, model_id=embedder.model_id)
        if vectors.shape[1] != embedder.dim:
            raise StaleCacheError(
                f"cache has {vectors.shape[1]} dims, embedder makes {embedder.dim}"
            )
        return cls(chunks, embedder, vectors)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Top ``k`` chunks by cosine similarity; ties keep corpus order so results are stable."""
        if k < 1:
            raise ValueError("k must be at least 1")
        scores = self._index.scores(self._embedder.embed_query(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(chunk_id=self._ids[position], score=float(scores[position]), rank=rank)
            for rank, position in enumerate(order, start=1)
        ]
