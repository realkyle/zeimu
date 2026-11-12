"""Reciprocal rank fusion and HybridRetriever: the math, missing hits, ties, and wiring."""

from pathlib import Path

import pytest

from tests.index.fake_embedder import FakeEmbedder
from tests.index.test_retriever import chunk
from zeimu.cli import embed_corpus, main, search_corpus
from zeimu.index.fusion import HybridRetriever, reciprocal_rank_fusion
from zeimu.index.retriever import BM25Retriever, EmbeddingRetriever, Hit, Retriever
from zeimu.index.tokenizer import SudachiTokenizer
from zeimu.ingest.chunker import Chunk
from zeimu.ingest.corpus import write_jsonl


def ranking(*chunk_ids: str) -> list[Hit]:
    """A ranked list as a retriever would return it; raw scores are irrelevant to RRF."""
    return [Hit(chunk_id=c, score=99.0, rank=r) for r, c in enumerate(chunk_ids, start=1)]


class StubRetriever:
    """Returns a fixed ranking (truncated to k) and records every k it was asked for."""

    def __init__(self, *chunk_ids: str) -> None:
        self._ids = chunk_ids
        self.asked_k: list[int] = []

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.asked_k.append(k)
        return ranking(*self._ids[:k])


# --- reciprocal_rank_fusion -------------------------------------------------------------


def test_score_is_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([ranking("a", "b", "c"), ranking("x", "y", "a")])

    by_id = {hit.chunk_id: hit.score for hit in fused}
    assert by_id["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert by_id["x"] == pytest.approx(1 / 61)
    assert fused[0].chunk_id == "a"
    assert [hit.rank for hit in fused] == list(range(1, len(fused) + 1))


def test_agreement_beats_a_single_first_place() -> None:
    # "both" is #2 in each list; "solo" is #1 in only one list.
    fused = reciprocal_rank_fusion([ranking("solo", "both"), ranking("other", "both")])
    assert fused[0].chunk_id == "both"


def test_empty_list_leaves_the_other_ranking_unchanged() -> None:
    """BM25 with no matching terms returns []; the hybrid then equals the other method."""
    fused = reciprocal_rank_fusion([[], ranking("a", "b", "c")])
    assert [hit.chunk_id for hit in fused] == ["a", "b", "c"]


def test_shorter_list_only_contributes_for_its_hits() -> None:
    fused = reciprocal_rank_fusion([ranking("a"), ranking("b", "c", "a")])
    by_id = {hit.chunk_id: hit.score for hit in fused}
    assert by_id["c"] == pytest.approx(1 / 62)
    assert by_id["a"] == pytest.approx(1 / 61 + 1 / 63)


def test_exact_ties_keep_first_seen_order() -> None:
    fused = reciprocal_rank_fusion([ranking("bm25_top"), ranking("embed_top")])
    assert [hit.chunk_id for hit in fused] == ["bm25_top", "embed_top"]
    assert fused[0].score == fused[1].score


def test_ties_go_by_list_order_then_rank() -> None:
    fused = reciprocal_rank_fusion([ranking("a", "b"), ranking("c", "d")])
    assert [hit.chunk_id for hit in fused] == ["a", "c", "b", "d"]


def test_same_ranks_in_another_list_order_tie_exactly() -> None:
    """Float addition isn't associative; fsum makes ranks (1,2,3) and (2,3,1) score the same."""
    fused = reciprocal_rank_fusion(
        [ranking("p", "q", "x"), ranking("y", "p", "q"), ranking("q", "z", "p")]
    )
    by_id = {hit.chunk_id: hit.score for hit in fused}
    assert by_id["p"] == by_id["q"]
    assert [h.chunk_id for h in fused][:2] == ["p", "q"]


def test_rejects_rank_below_one() -> None:
    with pytest.raises(ValueError, match="ranks start at 1"):
        reciprocal_rank_fusion([[Hit(chunk_id="a", score=1.0, rank=0)]])


def test_no_rankings_gives_no_hits() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_custom_k_changes_scores() -> None:
    fused = reciprocal_rank_fusion([ranking("a")], k=1)
    assert fused[0].score == pytest.approx(1 / 2)


def test_rejects_k_below_one() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        reciprocal_rank_fusion([ranking("a")], k=0)


def test_rejects_a_chunk_twice_in_one_ranking() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        reciprocal_rank_fusion([ranking("a", "a")])


# --- HybridRetriever --------------------------------------------------------------------


def test_is_a_retriever() -> None:
    searcher: Retriever = HybridRetriever([StubRetriever("a"), StubRetriever("b")])
    assert searcher.search("q", k=1)


def test_asks_each_retriever_for_depth_then_cuts_to_k() -> None:
    first, second = StubRetriever("a", "b", "c"), StubRetriever("c", "d")
    hits = HybridRetriever([first, second], depth=50).search("q", k=2)

    assert first.asked_k == [50]
    assert second.asked_k == [50]
    assert [hit.chunk_id for hit in hits] == ["c", "a"]


def test_asks_for_k_when_k_is_deeper_than_depth() -> None:
    first, second = StubRetriever("a"), StubRetriever("b")
    HybridRetriever([first, second], depth=3).search("q", k=10)
    assert first.asked_k == [10]


def test_depth_limits_what_gets_fused() -> None:
    """With depth 1, a chunk ranked #2 by both is never seen."""
    hits = HybridRetriever(
        [StubRetriever("a", "both"), StubRetriever("b", "both")], depth=1
    ).search("q", k=1)
    assert [hit.chunk_id for hit in hits] == ["a"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"depth": 0}, "depth must be at least 1"),
        ({"rrf_k": 0}, "rrf_k must be at least 1"),
    ],
)
def test_rejects_bad_settings(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HybridRetriever([StubRetriever("a"), StubRetriever("b")], **kwargs)


def test_needs_at_least_two_retrievers() -> None:
    with pytest.raises(ValueError, match="at least two"):
        HybridRetriever([StubRetriever("a")])


def test_search_rejects_k_below_one() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        HybridRetriever([StubRetriever("a"), StubRetriever("b")]).search("q", k=0)


# --- real BM25 + embedding retrievers (fake embedder) and the CLI ----------------------


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        chunk("gift", "贈与税の基礎控除は110万円です。"),
        chunk("spouse", "配偶者の税額軽減を受けるには申告が必要です。"),
        chunk("land", "路線価方式による宅地の評価"),
    ]


def test_fuses_real_bm25_and_embedding_retrievers(chunks: list[Chunk]) -> None:
    embedder = FakeEmbedder()
    vectors = embedder.embed_documents([c.text for c in chunks])
    hybrid = HybridRetriever(
        [BM25Retriever(chunks, SudachiTokenizer()), EmbeddingRetriever(chunks, embedder, vectors)]
    )

    hits = hybrid.search("配偶者の税額軽減", k=3)

    assert hits[0].chunk_id == "spouse"
    assert {hit.chunk_id for hit in hits} == {"gift", "spouse", "land"}


@pytest.fixture
def chunks_file(tmp_path: Path, chunks: list[Chunk]) -> Path:
    path = tmp_path / "chunks.jsonl"
    write_jsonl(chunks, path)
    return path


def test_cli_search_hybrid(tmp_path: Path, chunks_file: Path) -> None:
    cache_dir = tmp_path / "emb"
    embed_corpus(chunks_file, cache_dir, FakeEmbedder())

    lines = search_corpus(
        chunks_file,
        "配偶者の税額軽減",
        2,
        method="hybrid",
        cache_dir=cache_dir,
        embedder=FakeEmbedder(),
    )

    assert len(lines) == 2
    assert "spouse" in lines[0]
    assert "0.0328" in lines[0]  # 1/61 + 1/61, printed with 4 decimals


def test_cli_hybrid_missing_cache_fails_before_tokenizer_loads(
    tmp_path: Path,
    chunks_file: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def tokenizer_must_not_load() -> None:
        raise AssertionError("tokenizer was built before the cache check")

    monkeypatch.setattr("zeimu.cli.SudachiTokenizer", tokenizer_must_not_load)
    code = main(
        [
            "search",
            "x",
            "--method",
            "hybrid",
            "--chunks",
            str(chunks_file),
            "--embeddings",
            str(tmp_path / "none"),
        ]
    )
    assert code == 1
    assert "Run `zeimu embed`" in capsys.readouterr().out
