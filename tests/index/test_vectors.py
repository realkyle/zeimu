"""VectorIndex: cosine = dot product on unit rows, checked by hand, plus input validation."""

import math

import numpy as np
import pytest

from zeimu.index.vectors import VectorIndex

S = 1 / math.sqrt(2)


def test_scores_match_a_hand_worked_cosine() -> None:
    # Three unit vectors in 2-D: east, north-east, north.
    index = VectorIndex(np.array([[1.0, 0.0], [S, S], [0.0, 1.0]]))
    query = np.array([1.0, 0.0])

    # cos(0°) = 1, cos(45°) = √2/2, cos(90°) = 0
    np.testing.assert_allclose(index.scores(query), [1.0, S, 0.0], atol=1e-6)


def test_opposite_vector_scores_minus_one() -> None:
    index = VectorIndex(np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(index.scores(np.array([-1.0, 0.0])), [-1.0], atol=1e-6)


def test_index_is_a_private_read_only_copy() -> None:
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex(source)
    source[0] = [0.0, 1.0]  # mutating the caller's array must not change the index

    np.testing.assert_allclose(index.scores(np.array([1.0, 0.0])), [1.0, 0.0], atol=1e-6)
    assert len(index) == 2
    assert index.dim == 2


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.zeros((0, 3)), "non-empty 2-D"),
        (np.array([1.0, 0.0]), "non-empty 2-D"),
        (np.array([[np.nan, 1.0]]), "NaN"),
        (np.array([[np.inf, 0.0]]), "NaN or infinite"),
        (np.array([[3.0, 4.0]]), "L2-normalised"),
    ],
)
def test_rejects_invalid_matrices(matrix: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VectorIndex(matrix)


def test_rejects_query_of_wrong_dimension() -> None:
    index = VectorIndex(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="shape"):
        index.scores(np.array([1.0, 0.0, 0.0]))


def test_rejects_unnormalised_query() -> None:
    index = VectorIndex(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="L2-normalised"):
        index.scores(np.array([2.0, 0.0]))
