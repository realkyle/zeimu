"""Chunk vectors saved to disk so search doesn't re-embed the corpus on every run.

Layout (gitignored, rebuilt by ``zeimu embed``)::

    data/embeddings/
        vectors.npy   float32, shape (chunks, dim), row i = chunk i of chunks.jsonl
        meta.json     which model made them and which chunks they belong to

A cache is only usable if it was made by the same model from the same chunks. Searching
stale vectors would silently return wrong chunk IDs, so ``load`` refuses instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from zeimu.index.vectors import check_unit_rows
from zeimu.ingest.chunker import Chunk

VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


class StaleCacheError(Exception):
    """The cached vectors don't match the current chunks or model; run ``zeimu embed``."""


def chunks_fingerprint(chunks: Sequence[Chunk]) -> str:
    """SHA-256 over the ordered chunk IDs and texts: changes if anything embedded changes."""
    digest = hashlib.sha256()
    for chunk in chunks:
        for part in (chunk.id, chunk.text):
            encoded = part.encode("utf-8")
            # Length prefix so ("ab", "c") and ("a", "bc") can't hash the same.
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


class CacheMeta(BaseModel):
    """What a vector file was built from. Compared field by field on load."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    dim: int
    chunks_sha256: str
    chunk_ids: tuple[str, ...]
    created_at: datetime

    @classmethod
    def for_chunks(cls, chunks: Sequence[Chunk], *, model_id: str, dim: int) -> CacheMeta:
        """Metadata describing vectors for ``chunks`` made by ``model_id`` now."""
        return cls(
            model_id=model_id,
            dim=dim,
            chunks_sha256=chunks_fingerprint(chunks),
            chunk_ids=tuple(chunk.id for chunk in chunks),
            created_at=datetime.now(UTC),
        )


class VectorCache:
    """Reads and writes one cache directory."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    @property
    def directory(self) -> Path:
        return self._dir

    def save(self, vectors: NDArray[np.floating], meta: CacheMeta) -> None:
        """Write vectors and metadata. Rejects vectors that don't match ``meta``."""
        check_unit_rows(vectors)
        if vectors.shape != (len(meta.chunk_ids), meta.dim):
            raise ValueError(
                f"vectors shape {vectors.shape} != ({len(meta.chunk_ids)}, {meta.dim}) from meta"
            )
        self._dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._dir / META_FILE
        # meta.json is the commit marker: remove it first and write it last (atomically), so a
        # crash mid-save leaves "no cache" rather than new vectors under the old model's label.
        meta_path.unlink(missing_ok=True)
        np.save(self._dir / VECTORS_FILE, vectors.astype(np.float32), allow_pickle=False)
        partial = meta_path.with_suffix(".json.tmp")
        partial.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
        partial.replace(meta_path)

    def load(self, chunks: Sequence[Chunk], *, model_id: str) -> NDArray[np.float32]:
        """Vectors for ``chunks`` made by ``model_id``.

        Raises ``FileNotFoundError`` if there is no cache and ``StaleCacheError`` if it
        was built from different chunks, by a different model, or is malformed
        (bad JSON, pickled or wrong-shape array, NaN or non-unit rows).
        """
        meta_path, vectors_path = self._dir / META_FILE, self._dir / VECTORS_FILE
        if not meta_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(f"no vector cache in {self._dir}")
        try:
            meta = CacheMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except ValidationError as error:
            raise StaleCacheError(f"{meta_path} is malformed") from error
        if meta.model_id != model_id:
            raise StaleCacheError(f"cache was built by {meta.model_id}, not {model_id}")
        if meta.chunks_sha256 != chunks_fingerprint(chunks):
            raise StaleCacheError("chunks changed since the cache was built")
        # allow_pickle=False: a .npy file can otherwise carry arbitrary Python objects.
        try:
            vectors = np.load(vectors_path, allow_pickle=False)
        except ValueError as error:  # pickled or unreadable file
            raise StaleCacheError(f"{vectors_path} can't be loaded safely: {error}") from error
        if vectors.dtype != np.float32 or vectors.shape != (len(chunks), meta.dim):
            raise StaleCacheError(
                f"vectors are {vectors.dtype} {vectors.shape}, expected float32 "
                f"({len(chunks)}, {meta.dim})"
            )
        try:
            check_unit_rows(vectors)
        except ValueError as error:
            raise StaleCacheError(f"{vectors_path}: {error}") from error
        return vectors
