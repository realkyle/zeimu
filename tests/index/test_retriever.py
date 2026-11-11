"""BM25Retriever: ranking, the word-pair effect, and smoke checks on the real corpus."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from zeimu.index.retriever import BM25Retriever
from zeimu.index.tokenizer import SudachiTokenizer
from zeimu.ingest.chunker import Chunk
from zeimu.ingest.corpus import read_jsonl

CHUNKS_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks.jsonl"


@pytest.fixture(scope="module")
def tokenizer() -> SudachiTokenizer:
    return SudachiTokenizer()


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        ta_no="9999",
        page_kind="main",
        title="t",
        heading_path=("h",),
        text=text,
        url="https://x",
        law_as_of=None,
        fetched_at=datetime(2026, 11, 11, tzinfo=UTC),
        tax_type=None,
        legal_basis=None,
        related=(),
    )


def test_whole_phrase_outranks_scattered_words(tokenizer: SudachiTokenizer) -> None:
    """The reason for word pairs: same words, but only one chunk has them as a phrase."""
    chunks = [
        chunk("scattered", "軽減の対象となる税額は、配偶者が取得した財産で決まります。"),
        chunk("phrase", "配偶者の税額軽減を受けるには申告が必要です。"),
        chunk("unrelated", "路線価方式による宅地の評価"),
    ]

    with_pairs = BM25Retriever(chunks, tokenizer).search("配偶者の税額軽減")
    words_only = BM25Retriever(chunks, SudachiTokenizer(pairs=False)).search("配偶者の税額軽減")

    assert [h.chunk_id for h in with_pairs] == ["phrase", "scattered"]
    gap_with_pairs = with_pairs[0].score - with_pairs[1].score
    gap_words_only = words_only[0].score - words_only[1].score
    assert gap_with_pairs > gap_words_only


def test_hits_are_ranked_limited_and_exclude_non_matches(tokenizer: SudachiTokenizer) -> None:
    chunks = [chunk(f"c{i}", "相続税" * (i + 1) + "の申告") for i in range(4)]
    chunks.append(chunk("none", "路線価方式"))

    hits = BM25Retriever(chunks, tokenizer).search("相続税", k=3)

    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert "none" not in [h.chunk_id for h in hits]


def test_no_match_returns_empty(tokenizer: SudachiTokenizer) -> None:
    assert BM25Retriever([chunk("a", "相続税")], tokenizer).search("路線価") == []


def test_k_must_be_positive(tokenizer: SudachiTokenizer) -> None:
    with pytest.raises(ValueError, match="k"):
        BM25Retriever([chunk("a", "相続税")], tokenizer).search("相続税", k=0)


@pytest.mark.parametrize(
    ("query", "expected_page"),
    [("配偶者の税額軽減", "4158"), ("相続税の税率", "4155")],
)
def test_real_corpus_smoke(tokenizer: SudachiTokenizer, query: str, expected_page: str) -> None:
    """Sanity check only. Retrieval quality is measured by the evaluation set, not here."""
    chunks = read_jsonl(CHUNKS_PATH)
    by_id = {c.id: c for c in chunks}

    top = BM25Retriever(chunks, tokenizer).search(query, k=1)[0]

    assert by_id[top.chunk_id].ta_no == expected_page
