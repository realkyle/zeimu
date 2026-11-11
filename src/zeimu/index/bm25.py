"""Okapi BM25, the classic keyword-ranking formula, written out so every term is visible.

For a query with terms q and a document d::

    score(d) = Σ_q  idf(q) · tf(q,d) · (k1 + 1) / ( tf(q,d) + k1 · (1 - b + b · |d| / avgdl) )

    idf(q) = ln( (N - df(q) + 0.5) / (df(q) + 0.5) + 1 )

- ``tf(q,d)``: how often the term appears in the document. More is better, but with
  diminishing returns: ``k1`` controls how fast it saturates.
- ``|d| / avgdl``: document length relative to the average. Long documents mention
  everything, so ``b`` (0 = ignore, 1 = full) scales their term counts down.
- ``idf(q)``: rare terms matter more than common ones. ``N`` is the number of documents,
  ``df`` how many contain the term. The ``+ 1`` (as in Lucene) keeps idf positive even
  for a term in most documents.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Index:
    """An in-memory BM25 index over pre-tokenized documents.

    Stored as an inverted index: for each term, the documents containing it and how
    often. Scoring a query touches only the postings of its terms.
    """

    def __init__(
        self, documents: Sequence[Sequence[str]], *, k1: float = DEFAULT_K1, b: float = DEFAULT_B
    ) -> None:
        if not documents:
            raise ValueError("cannot index an empty corpus")
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        self._k1 = k1
        self._b = b
        self._doc_count = len(documents)
        self._doc_lengths = np.array([len(doc) for doc in documents], dtype=np.float64)
        average = float(self._doc_lengths.mean())
        # All-empty documents would divide by zero; their length factor is irrelevant anyway.
        self._avg_length = average if average > 0 else 1.0

        postings: dict[str, tuple[list[int], list[int]]] = {}
        for doc_id, doc in enumerate(documents):
            for term, count in Counter(doc).items():
                ids, counts = postings.setdefault(term, ([], []))
                ids.append(doc_id)
                counts.append(count)
        self._postings: dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]] = {
            term: (np.array(ids, dtype=np.int64), np.array(counts, dtype=np.float64))
            for term, (ids, counts) in postings.items()
        }

    def __len__(self) -> int:
        return self._doc_count

    def idf(self, term: str) -> float:
        """Inverse document frequency of ``term`` (0.0 if no document contains it)."""
        posting = self._postings.get(term)
        if posting is None:
            return 0.0
        df = len(posting[0])
        return math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)

    def scores(self, query_terms: Sequence[str]) -> NDArray[np.float64]:
        """BM25 score of every document for ``query_terms`` (index = document position).

        Each distinct query term counts once, so repeating a word in the query doesn't
        change the ranking.
        """
        totals = np.zeros(self._doc_count, dtype=np.float64)
        for term in dict.fromkeys(query_terms):  # distinct, in order
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_ids, tf = posting
            length_norm = 1 - self._b + self._b * self._doc_lengths[doc_ids] / self._avg_length
            totals[doc_ids] += self.idf(term) * tf * (self._k1 + 1) / (tf + self._k1 * length_norm)
        return totals
