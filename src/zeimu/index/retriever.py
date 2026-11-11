"""The retrieval interface shared by every search method, and the BM25 implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from zeimu.index.bm25 import BM25Index
from zeimu.index.tokenizer import Tokenizer
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
        """Return up to ``k`` hits, best first. Chunks that don't match at all are left out."""
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
