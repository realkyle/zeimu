"""Build ``data/chunks.jsonl`` from every Tax Answer page in the raw cache."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from zeimu.ingest.cache import FetchedPage
from zeimu.ingest.chunker import DEFAULT_MAX_CHARS, Chunk, PageSource, chunk_page
from zeimu.ingest.cleaner import CleanError, CleanPage, clean_page
from zeimu.ingest.encoding import decode_html

_PAGE_PATH = re.compile(r"/taxanswer/[a-z]+/(?P<number>\d{4})(?P<qa>_qa)?\.htm$")


@dataclass
class BuildReport:
    """What happened during a build, so problems are visible instead of silent."""

    pages: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)


def build_chunks(
    pages: Iterable[FetchedPage], *, max_chars: int = DEFAULT_MAX_CHARS
) -> tuple[list[Chunk], BuildReport]:
    """Clean and chunk every Tax Answer page in ``pages``; other pages are ignored.

    Q&A pages have no 対象税目 or 根拠法令等 section of their own, so they inherit those
    from the main page with the same number. Output is sorted by URL then document
    order, so the same input always produces the same file.
    """
    report = BuildReport()
    cleaned: list[tuple[PageSource, CleanPage]] = []
    for page in sorted(pages, key=lambda p: p.url):
        match = _PAGE_PATH.search(urlsplit(page.url).path)
        if match is None:
            continue
        source = PageSource(
            url=page.url,
            ta_no=match["number"],
            page_kind="qa" if match["qa"] else "main",
            fetched_at=page.meta.fetched_at,
        )
        try:
            clean = clean_page(decode_html(page.body))
        except (CleanError, UnicodeDecodeError) as error:
            report.warnings.append(f"{page.url}: skipped ({error})")
            continue
        if clean.law_as_of is None:
            report.warnings.append(f"{page.url}: no law-as-of date")
        cleaned.append((source, clean))

    main_pages = {src.ta_no: clean for src, clean in cleaned if src.page_kind == "main"}
    chunks: list[Chunk] = []
    for source, clean in cleaned:
        if source.page_kind == "qa" and (parent := main_pages.get(source.ta_no)) is not None:
            clean = CleanPage(
                title=clean.title,
                law_as_of=clean.law_as_of,
                tax_type=clean.tax_type or parent.tax_type,
                legal_basis=clean.legal_basis or parent.legal_basis,
                related=clean.related or parent.related,
                sections=clean.sections,
            )
        page_chunks = chunk_page(clean, source, max_chars=max_chars)
        if not page_chunks:
            report.warnings.append(f"{source.url}: produced no chunks")
        chunks.extend(page_chunks)
        report.pages += 1

    duplicates = _duplicate_ids(chunks)
    if duplicates:
        raise ValueError(f"duplicate chunk IDs: {sorted(duplicates)}")
    report.chunks = len(chunks)
    return chunks, report


def write_jsonl(chunks: Iterable[Chunk], path: Path) -> None:
    """Write one chunk per line as UTF-8 JSON (Japanese stays readable, not \\u-escaped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for chunk in chunks:
            out.write(chunk.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[Chunk]:
    """Read chunks written by ``write_jsonl``."""
    with path.open(encoding="utf-8") as lines:
        return [Chunk.model_validate_json(line) for line in lines if line.strip()]


def _duplicate_ids(chunks: list[Chunk]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for chunk in chunks:
        (duplicates if chunk.id in seen else seen).add(chunk.id)
    return duplicates
