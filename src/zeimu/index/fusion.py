"""Hybrid search: merge several ranked lists with reciprocal rank fusion (RRF).

RRF scores a chunk by where it ranks, not by its raw score: ``Σ 1 / (k + rank)`` over
the lists it appears in. That lets us combine BM25 scores and cosine similarities,
which live on unrelated scales, without calibrating one against the other.
(Cormack, Clarke & Büttcher, SIGIR 2009; ``k = 60`` is the value from that paper.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from zeimu.index.retriever import Hit, Retriever

DEFAULT_RRF_K = 60
DEFAULT_DEPTH = 50


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]], *, k: int = DEFAULT_RRF_K
) -> list[Hit]:
    """Fuse ranked lists into one, best first, with ranks renumbered from 1.

    A chunk missing from a list adds nothing for that list, so a list with fewer hits
    (BM25 when few chunks share a term) simply contributes less. Exact ties keep the
    order chunks were first seen in: list order, then rank within the list. Each score is
    summed with ``math.fsum`` (exactly rounded), so two chunks with the same ranks in a
    different list order get bit-identical scores and really do tie.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    terms: dict[str, list[float]] = {}  # dicts keep insertion order = first-seen order
    for ranking in rankings:
        seen: set[str] = set()
        for hit in ranking:
            if hit.rank < 1:
                raise ValueError(f"chunk {hit.chunk_id!r} has rank {hit.rank}; ranks start at 1")
            if hit.chunk_id in seen:
                raise ValueError(f"chunk {hit.chunk_id!r} appears twice in one ranking")
            seen.add(hit.chunk_id)
            terms.setdefault(hit.chunk_id, []).append(1.0 / (k + hit.rank))
    scores = {chunk_id: math.fsum(parts) for chunk_id, parts in terms.items()}
    # sorted() is stable, so equal scores stay in first-seen order.
    fused = sorted(scores.items(), key=lambda item: -item[1])
    return [
        Hit(chunk_id=chunk_id, score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(fused, start=1)
    ]


class HybridRetriever:
    """Runs each retriever it is given and fuses their top ``depth`` hits with RRF.

    The retrievers are passed in (not built here), so any mix of ``Retriever``s works
    and tests can use stubs. Scores are RRF sums, comparable only within this method.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        depth: int = DEFAULT_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("hybrid search needs at least two retrievers")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self._retrievers = tuple(retrievers)
        self._depth = depth
        self._rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Top ``k`` chunks by fused score; each retriever is asked for ``max(depth, k)``."""
        if k < 1:
            raise ValueError("k must be at least 1")
        per_list = max(self._depth, k)
        rankings = [retriever.search(query, k=per_list) for retriever in self._retrievers]
        return reciprocal_rank_fusion(rankings, k=self._rrf_k)[:k]
