"""The real bge-m3 model. Skipped by default; run with `uv run pytest -m model`.

These are sanity checks that the wiring is right (dimensions, normalisation, one obvious
cross-lingual query). They are not an evaluation and their pass is not an accuracy claim.
"""

from pathlib import Path

import numpy as np
import pytest

from zeimu.index.embedder import SentenceTransformerEmbedder
from zeimu.index.retriever import EmbeddingRetriever
from zeimu.ingest.corpus import read_jsonl

pytestmark = pytest.mark.model

CHUNKS_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks.jsonl"


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def test_vectors_are_1024_dim_and_unit_length(embedder: SentenceTransformerEmbedder) -> None:
    vectors = embedder.embed_documents(["配偶者の税額軽減", "spouse tax reduction"])

    assert vectors.shape == (2, 1024)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)


def test_english_query_finds_the_japanese_page(embedder: SentenceTransformerEmbedder) -> None:
    """配偶者の税額軽減 is TA No.4158. BM25 can't match English to Japanese at all."""
    chunks = read_jsonl(CHUNKS_PATH)
    searcher = EmbeddingRetriever(
        chunks, embedder, embedder.embed_documents([c.text for c in chunks])
    )
    by_id = {c.id: c for c in chunks}

    hits = searcher.search("spouse tax reduction for inheritance tax", k=5)

    assert any(by_id[h.chunk_id].ta_no == "4158" for h in hits)
