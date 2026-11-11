"""BM25Index: matches a hand-worked example and behaves as the formula says it should."""

import math

import pytest

from zeimu.index.bm25 import BM25Index

# Hand-worked example (k1 = 1.5, b = 0.75), query ["a"]:
#   documents: d0 = [a, b] (|d|=2), d1 = [a, a, c] (|d|=3), d2 = [c] (|d|=1)
#   N = 3, df(a) = 2, avgdl = (2 + 3 + 1) / 3 = 2
#   idf(a) = ln((3 - 2 + 0.5) / (2 + 0.5) + 1) = ln(1.6) = 0.4700036
#   d0: tf=1, norm = 1 - 0.75 + 0.75 * 2/2 = 1      -> 1 * 2.5 / (1 + 1.5 * 1)     = 1.0
#   d1: tf=2, norm = 1 - 0.75 + 0.75 * 3/2 = 1.375  -> 2 * 2.5 / (2 + 1.5 * 1.375) = 1.2307692
#   d2: no "a" -> 0
DOCS = [["a", "b"], ["a", "a", "c"], ["c"]]


def test_matches_hand_worked_example() -> None:
    scores = BM25Index(DOCS).scores(["a"])

    assert scores[0] == pytest.approx(0.4700036 * 1.0)
    assert scores[1] == pytest.approx(0.4700036 * 1.2307692)
    assert scores[2] == 0


def test_idf() -> None:
    index = BM25Index(DOCS)

    assert index.idf("a") == pytest.approx(math.log(1.6))
    assert index.idf("b") > index.idf("a")  # rarer term counts more
    assert index.idf("zzz") == 0


def test_multi_term_score_is_the_sum() -> None:
    index = BM25Index(DOCS)

    combined = index.scores(["a", "b"])

    assert combined == pytest.approx(index.scores(["a"]) + index.scores(["b"]))


def test_repeated_query_terms_count_once() -> None:
    index = BM25Index(DOCS)

    assert index.scores(["a", "a"]) == pytest.approx(index.scores(["a"]))


def test_longer_document_with_same_count_scores_lower() -> None:
    scores = BM25Index([["x", "y"], ["x", "y", "z", "z", "z", "z"]]).scores(["x"])

    assert scores[0] > scores[1] > 0


def test_term_frequency_saturates() -> None:
    """Ten mentions should beat one, but by much less than ten times."""
    scores = BM25Index([["x"] + ["f"] * 9, ["x"] * 10]).scores(["x"])

    assert 1 < scores[1] / scores[0] < 3


def test_b_zero_ignores_length() -> None:
    scores = BM25Index([["x", "y"], ["x", "y", "z", "z", "z", "z"]], b=0).scores(["x"])

    assert scores[0] == pytest.approx(scores[1])


def test_unknown_and_empty_queries_score_zero() -> None:
    index = BM25Index(DOCS)

    assert not index.scores(["unknown"]).any()
    assert not index.scores([]).any()


def test_rejects_empty_corpus_and_bad_parameters() -> None:
    with pytest.raises(ValueError, match="empty"):
        BM25Index([])
    with pytest.raises(ValueError, match="k1"):
        BM25Index(DOCS, b=1.5)
