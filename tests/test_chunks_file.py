"""Checks on the committed corpus file, so a bad rebuild can't be merged unnoticed."""

from pathlib import Path

import pytest

from zeimu.ingest.chunker import Chunk
from zeimu.ingest.cleaner import BOILERPLATE_SECTIONS
from zeimu.ingest.corpus import read_jsonl

CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data" / "chunks.jsonl"


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return read_jsonl(CHUNKS_PATH)


def test_ids_are_unique(chunks: list[Chunk]) -> None:
    assert len({c.id for c in chunks}) == len(chunks)


def test_every_chunk_has_a_law_as_of_date_and_text(chunks: list[Chunk]) -> None:
    assert all(c.law_as_of is not None for c in chunks)
    assert all(len(c.text.split("\n")) >= 2 for c in chunks)


def test_no_boilerplate_sections(chunks: list[Chunk]) -> None:
    assert not [c.id for c in chunks if c.heading_path[0] in BOILERPLATE_SECTIONS]


def test_covers_all_three_categories(chunks: list[Chunk]) -> None:
    urls = " ".join(c.url for c in chunks)
    for category in ("/sozoku/", "/zoyo/", "/hyoka/"):
        assert category in urls
