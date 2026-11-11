"""Split cleaned pages into chunks: the unit that search returns and answers cite."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from zeimu.ingest.cleaner import CleanPage, Section

#: Sections longer than this (in characters) are split; decided 2026-11-11.
DEFAULT_MAX_CHARS = 1500

_QUESTION_HEADING = re.compile(r"^Q\d*$")

PageKind = Literal["main", "qa"]


class Chunk(BaseModel):
    """One citable piece of a Tax Answer page.

    ``text`` is the source wording unchanged (so it can be quoted), preceded by one
    context line naming the page and heading path. Search-time normalisation happens
    in the index, never here.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    ta_no: str
    page_kind: PageKind
    title: str
    heading_path: tuple[str, ...]
    text: str
    url: str
    law_as_of: date | None
    fetched_at: datetime
    tax_type: str | None
    legal_basis: str | None
    related: tuple[str, ...]


class PageSource(BaseModel):
    """Where a page came from: what the chunker needs besides the cleaned content."""

    model_config = ConfigDict(frozen=True)

    url: str
    ta_no: str
    page_kind: PageKind
    fetched_at: datetime


def chunk_page(
    page: CleanPage, source: PageSource, *, max_chars: int = DEFAULT_MAX_CHARS
) -> list[Chunk]:
    """Chunk one page: one chunk per h2 section, splitting sections over ``max_chars``.

    Splitting goes h3 → h4 → paragraph groups. On Q&A pages a question and its answer
    stay together; if they must be split, every piece repeats the question.
    """
    pieces: list[tuple[tuple[str, ...], list[str]]] = []
    for section in page.sections:
        question = _question_text(section) if source.page_kind == "qa" else None
        pieces.extend(_split(section, (section.heading,), max_chars, question))

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for path, blocks in pieces:
        base_id = _chunk_id(source, path)
        seen[base_id] = seen.get(base_id, 0) + 1
        chunk_id = base_id if seen[base_id] == 1 else f"{base_id}-p{seen[base_id]}"
        header = " > ".join((page.title, *path))
        chunks.append(
            Chunk(
                id=chunk_id,
                ta_no=source.ta_no,
                page_kind=source.page_kind,
                title=page.title,
                heading_path=path,
                text="\n".join((header, *blocks)),
                url=source.url,
                law_as_of=page.law_as_of,
                fetched_at=source.fetched_at,
                tax_type=page.tax_type,
                legal_basis=page.legal_basis,
                related=page.related,
            )
        )
    return chunks


def _split(
    section: Section, path: tuple[str, ...], max_chars: int, question: str | None
) -> list[tuple[tuple[str, ...], list[str]]]:
    """Return ``(heading_path, blocks)`` pieces for ``section``, each within ``max_chars``
    where possible (a single paragraph longer than the limit is kept whole)."""
    if section.char_count() <= max_chars:
        blocks = section.all_blocks()
        return [(path, blocks)] if blocks else []

    prefix = [f"Q: {question}"] if question else []
    pieces: list[tuple[tuple[str, ...], list[str]]] = []
    if section.children:
        if section.blocks:
            pieces.append((path, prefix + section.blocks))
        for child in section.children:
            if question and _is_question(child.heading):
                continue  # the question is repeated in every piece instead
            for child_path, blocks in _split(child, (*path, child.heading), max_chars, None):
                pieces.append((child_path, prefix + blocks))
        return pieces

    group: list[str] = []
    size = 0
    for block in section.blocks:
        if group and size + len(block) > max_chars:
            pieces.append((path, prefix + group))
            group = []
            size = 0
        group.append(block)
        size += len(block)
    if group:
        pieces.append((path, prefix + group))
    return pieces


def _question_text(section: Section) -> str | None:
    for child in section.children:
        if _is_question(child.heading):
            return " ".join(child.all_blocks()) or None
    return None


def _is_question(heading: str) -> bool:
    """True for Q&A question headings: ``Q1``, and the full-width ``Ｑ1`` some pages use."""
    return _QUESTION_HEADING.match(unicodedata.normalize("NFKC", heading)) is not None


def _chunk_id(source: PageSource, path: tuple[str, ...]) -> str:
    """Stable ID from the page number and heading path, e.g. ``ta4152-1a2b3c4d``.

    Based on headings, not position, so NTA inserting a section elsewhere on the page
    doesn't change the IDs that hand-labelled evaluation cases point to.
    """
    digest = hashlib.sha1("\x1f".join(path).encode("utf-8"), usedforsecurity=False)
    kind = "qa" if source.page_kind == "qa" else ""
    return f"ta{source.ta_no}{kind}-{digest.hexdigest()[:8]}"
