"""VectorCache: round trip, and every way a cache can be stale is refused."""

from pathlib import Path

import numpy as np
import pytest

from tests.index.fake_embedder import FakeEmbedder
from tests.index.test_retriever import chunk
from zeimu.index.vector_cache import (
    META_FILE,
    VECTORS_FILE,
    CacheMeta,
    StaleCacheError,
    VectorCache,
    chunks_fingerprint,
)
from zeimu.ingest.chunker import Chunk


@pytest.fixture
def chunks() -> list[Chunk]:
    return [chunk("a", "相続税の申告"), chunk("b", "贈与税の基礎控除"), chunk("c", "路線価")]


def saved(tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder) -> VectorCache:
    cache = VectorCache(tmp_path / "emb")
    vectors = embedder.embed_documents([c.text for c in chunks])
    cache.save(vectors, CacheMeta.for_chunks(chunks, model_id=embedder.model_id, dim=embedder.dim))
    return cache


def test_round_trip_returns_identical_vectors(tmp_path: Path, chunks: list[Chunk]) -> None:
    embedder = FakeEmbedder()
    cache = saved(tmp_path, chunks, embedder)

    loaded = cache.load(chunks, model_id="fake@1")

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, embedder.embed_documents([c.text for c in chunks]))
    meta = CacheMeta.model_validate_json((tmp_path / "emb" / META_FILE).read_text())
    assert meta.chunk_ids == ("a", "b", "c")
    assert meta.model_id == "fake@1"


def test_missing_cache_raises_file_not_found(tmp_path: Path, chunks: list[Chunk]) -> None:
    with pytest.raises(FileNotFoundError):
        VectorCache(tmp_path / "nowhere").load(chunks, model_id="fake@1")


def test_different_model_is_stale(tmp_path: Path, chunks: list[Chunk]) -> None:
    cache = saved(tmp_path, chunks, FakeEmbedder())
    with pytest.raises(StaleCacheError, match="built by fake@1"):
        cache.load(chunks, model_id="fake@2")


def test_changed_chunk_text_is_stale(tmp_path: Path, chunks: list[Chunk]) -> None:
    cache = saved(tmp_path, chunks, FakeEmbedder())
    edited = [*chunks[:2], chunk("c", "路線価方式")]
    with pytest.raises(StaleCacheError, match="chunks changed"):
        cache.load(edited, model_id="fake@1")


def test_reordered_or_removed_chunks_are_stale(tmp_path: Path, chunks: list[Chunk]) -> None:
    cache = saved(tmp_path, chunks, FakeEmbedder())
    with pytest.raises(StaleCacheError):
        cache.load(list(reversed(chunks)), model_id="fake@1")
    with pytest.raises(StaleCacheError):
        cache.load(chunks[:2], model_id="fake@1")


def test_tampered_vector_file_shape_is_stale(tmp_path: Path, chunks: list[Chunk]) -> None:
    cache = saved(tmp_path, chunks, FakeEmbedder())
    np.save(tmp_path / "emb" / VECTORS_FILE, np.ones((2, 64), dtype=np.float32))
    with pytest.raises(StaleCacheError, match="expected float32"):
        cache.load(chunks, model_id="fake@1")


def test_pickled_vector_file_is_refused(tmp_path: Path, chunks: list[Chunk]) -> None:
    """An object array needs pickle to load; pickle can run code, so loading must fail."""
    cache = saved(tmp_path, chunks, FakeEmbedder())
    np.save(tmp_path / "emb" / VECTORS_FILE, np.array([{"x": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(StaleCacheError, match="pickle"):
        cache.load(chunks, model_id="fake@1")


def test_malformed_meta_is_stale(tmp_path: Path, chunks: list[Chunk]) -> None:
    cache = saved(tmp_path, chunks, FakeEmbedder())
    (tmp_path / "emb" / META_FILE).write_text("{not json")
    with pytest.raises(StaleCacheError, match="malformed"):
        cache.load(chunks, model_id="fake@1")


def test_nan_or_non_unit_vectors_are_stale(tmp_path: Path, chunks: list[Chunk]) -> None:
    cache = saved(tmp_path, chunks, FakeEmbedder())
    bad = np.full((3, 64), 0.5, dtype=np.float32)  # norm 4, not 1
    np.save(tmp_path / "emb" / VECTORS_FILE, bad)
    with pytest.raises(StaleCacheError, match="L2-normalised"):
        cache.load(chunks, model_id="fake@1")


def test_resave_removes_old_meta_before_writing_vectors(
    tmp_path: Path, chunks: list[Chunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-save must leave no cache, not new vectors under the old model's meta."""
    cache = saved(tmp_path, chunks, FakeEmbedder(model_id="old@1"))
    new = FakeEmbedder(model_id="new@1")

    def crash(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", crash)
    with pytest.raises(OSError, match="disk full"):
        cache.save(
            new.embed_documents([c.text for c in chunks]),
            CacheMeta.for_chunks(chunks, model_id=new.model_id, dim=new.dim),
        )
    with pytest.raises(FileNotFoundError):
        cache.load(chunks, model_id="old@1")


def test_save_rejects_vectors_not_matching_meta(tmp_path: Path, chunks: list[Chunk]) -> None:
    embedder = FakeEmbedder()
    meta = CacheMeta.for_chunks(chunks, model_id=embedder.model_id, dim=embedder.dim)
    too_few = embedder.embed_documents([c.text for c in chunks[:2]])
    with pytest.raises(ValueError, match="shape"):
        VectorCache(tmp_path).save(too_few, meta)


def test_fingerprint_separates_id_and_text_boundaries() -> None:
    assert chunks_fingerprint([chunk("ab", "c")]) != chunks_fingerprint([chunk("a", "bc")])
