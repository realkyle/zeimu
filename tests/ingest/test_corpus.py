"""build_chunks and the JSONL file: Q&A inheritance, determinism, warnings, round trip."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from zeimu.ingest.cache import FetchedPage, PageMeta
from zeimu.ingest.corpus import build_chunks, read_jsonl, write_jsonl

from .sample_pages import MAIN_PAGE, QA_PAGE

TA = "https://www.nta.go.jp/taxes/shiraberu/taxanswer/sozoku"
WHEN = datetime(2026, 11, 11, tzinfo=UTC)


def fetched(url: str, body: bytes) -> FetchedPage:
    meta = PageMeta(url=url, status=200, fetched_at=WHEN, sha256=hashlib.sha256(body).hexdigest())
    return FetchedPage(meta=meta, body=body, from_cache=True)


def sample_pages() -> list[FetchedPage]:
    return [
        fetched(f"{TA}/9999_qa.htm", QA_PAGE.format(long_answer="答え").encode("cp932")),
        fetched(f"{TA}/9999.htm", MAIN_PAGE.encode()),
        fetched("https://www.nta.go.jp/taxes/shiraberu/taxanswer/code/index.htm", b"<p>index</p>"),
    ]


def test_builds_main_and_qa_chunks_and_ignores_other_pages() -> None:
    chunks, report = build_chunks(sample_pages())

    assert [(c.ta_no, c.page_kind) for c in chunks] == [("9999", "main"), ("9999", "qa")]
    assert report.pages == 2
    assert report.chunks == 2
    assert report.warnings == []


def test_qa_chunks_inherit_metadata_from_main_page() -> None:
    chunks, _ = build_chunks(sample_pages())
    qa = next(c for c in chunks if c.page_kind == "qa")

    assert qa.legal_basis == "相法15、16"
    assert qa.tax_type == "相続税"
    assert qa.fetched_at == WHEN


def test_bad_page_becomes_a_warning_not_a_crash() -> None:
    broken = fetched(f"{TA}/9998.htm", b"<html><body>no content area</body></html>")

    chunks, report = build_chunks([broken, *sample_pages()])

    assert len(chunks) == 2
    assert report.warnings == [f"{TA}/9998.htm: skipped (no #bodyArea content area)"]


def test_output_is_byte_identical_across_runs_and_round_trips(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(build_chunks(sample_pages())[0], first)
    write_jsonl(build_chunks(reversed(sample_pages()))[0], second)

    assert first.read_bytes() == second.read_bytes()
    assert "相続税" in first.read_text(encoding="utf-8")  # not \\u-escaped
    assert read_jsonl(first) == build_chunks(sample_pages())[0]
