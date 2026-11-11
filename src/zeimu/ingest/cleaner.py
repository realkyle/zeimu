"""Turn one NTA Tax Answer page into a clean tree of headed sections plus page metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from selectolax.lexbor import LexborHTMLParser, LexborNode

from zeimu.ingest.japanese_dates import parse_law_as_of

#: h2 sections that are navigation or contact boilerplate, not tax content.
BOILERPLATE_SECTIONS = frozenset(
    {
        "お問い合わせ先",
        "税の情報・手続・用紙",
        "サイトマップ（コンテンツ一覧）",
        "関連リンク",
        "関連コード",
        "QAリンク",
    }
)
#: h2 sections that become page metadata instead of chunks.
TAX_TYPE_SECTION = "対象税目"
LEGAL_BASIS_SECTION = "根拠法令等"
RELATED_SECTION = "関連コード"

_HEADING_LEVELS = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCK_TAGS = frozenset(
    {"p", "div", "li", "ul", "ol", "dl", "dt", "dd", "section", "article", "blockquote", "pre"}
)
_SKIP_TAGS = frozenset({"script", "style", "noscript", "nav", "form", "button", "h1"})
_WHITESPACE = re.compile(r"[ \t\r\n\u3000\u2003\u00a0]+")  # incl. ideographic, em, no-break spaces
_TA_NUMBER = re.compile(r"(\d{4})")


class CleanError(Exception):
    """The page doesn't have the structure the cleaner expects."""


@dataclass
class Section:
    """A heading and the text blocks under it, with its sub-sections as children.

    ``blocks`` holds only the text that comes before the first child heading.
    """

    heading: str
    level: int
    blocks: list[str] = field(default_factory=list)
    children: list[Section] = field(default_factory=list)

    def all_blocks(self) -> list[str]:
        """Every text block in this section, children included, with child headings as lines."""
        out = list(self.blocks)
        for child in self.children:
            out.append(f"【{child.heading}】")
            out.extend(child.all_blocks())
        return out

    def char_count(self) -> int:
        """Length of all text in this section, used to decide whether to split it."""
        return sum(len(block) for block in self.all_blocks())


@dataclass(frozen=True)
class CleanPage:
    """A cleaned page: its title, metadata and content sections (boilerplate removed)."""

    title: str
    law_as_of: date | None
    tax_type: str | None
    legal_basis: str | None
    related: tuple[str, ...]
    sections: tuple[Section, ...]


def clean_page(html: str) -> CleanPage:
    """Parse a Tax Answer page (main or Q&A) into a ``CleanPage``.

    Raises:
        CleanError: the page has no ``#bodyArea`` content area or no ``<h1>`` title.
    """
    body = LexborHTMLParser(html).css_first("#bodyArea")
    if body is None:
        raise CleanError("no #bodyArea content area")
    h1 = body.css_first("h1")
    if h1 is None:
        raise CleanError("no <h1> title")

    builder = _SectionBuilder()
    builder.walk(body)
    top_level = builder.finish()

    by_heading = {section.heading: section for section in top_level}
    content = tuple(
        section
        for section in top_level
        if section.heading not in BOILERPLATE_SECTIONS
        and section.heading not in (TAX_TYPE_SECTION, LEGAL_BASIS_SECTION)
    )
    return CleanPage(
        title=_squash(h1.text()),
        law_as_of=parse_law_as_of(body.text()),
        tax_type=_joined(by_heading.get(TAX_TYPE_SECTION)),
        legal_basis=_joined(by_heading.get(LEGAL_BASIS_SECTION)),
        related=_related_numbers(by_heading.get(RELATED_SECTION)),
        sections=content,
    )


class _SectionBuilder:
    """Walks the DOM in document order, collecting text blocks under the current heading.

    Inline content (text, links, images) accumulates in a buffer; block elements and
    headings flush it. Text before the first h2 (the law-as-of line) is discarded.
    """

    def __init__(self) -> None:
        self._roots: list[Section] = []
        self._stack: list[Section] = []
        self._buffer: list[str] = []

    def walk(self, node: LexborNode) -> None:
        for child in node.iter(include_text=True):
            tag = child.tag
            if tag == "-text":
                self._buffer.append(child.text(deep=False))
            elif tag is None or tag in _SKIP_TAGS or child.is_comment_node:
                continue
            elif tag in _HEADING_LEVELS:
                self._open_section(_squash(child.text()), _HEADING_LEVELS[tag])
            elif tag == "table":
                self._flush()
                for row in _table_rows(child):
                    self._emit(row)
            elif tag == "img":
                alt = _squash(child.attributes.get("alt") or "")
                if alt:
                    self._buffer.append(f"[図: {alt}]")
            elif tag == "br":
                self._flush()
            elif tag in _BLOCK_TAGS:
                self._flush()
                self.walk(child)
                self._flush()
            else:
                self.walk(child)

    def finish(self) -> list[Section]:
        self._flush()
        return self._roots

    def _open_section(self, heading: str, level: int) -> None:
        self._flush()
        section = Section(heading=heading, level=level)
        while self._stack and self._stack[-1].level >= level:
            self._stack.pop()
        if self._stack:
            self._stack[-1].children.append(section)
        else:
            self._roots.append(section)
        self._stack.append(section)

    def _flush(self) -> None:
        text = _squash("".join(self._buffer))
        self._buffer.clear()
        self._emit(text)

    def _emit(self, text: str) -> None:
        if text and self._stack:
            self._stack[-1].blocks.append(text)


def _table_rows(table: LexborNode) -> list[str]:
    """Flatten a table to one line per row.

    If the first row is all ``<th>``, later rows are written ``header: value | ...`` so a
    row still makes sense on its own. Merged cells (rowspan/colspan) are not expanded.
    """
    rows: list[list[tuple[str, bool]]] = []
    for tr in table.css("tr"):
        cells = [(_squash(cell.text()), cell.tag == "th") for cell in tr.css("th, td")]
        if cells:
            rows.append(cells)
    lines: list[str] = []
    caption = table.css_first("caption")
    if caption is not None and (caption_text := _squash(caption.text())):
        lines.append(caption_text)
    if not rows:
        return lines
    header = [text for text, _ in rows[0]] if all(is_th for _, is_th in rows[0]) else None
    body = rows[1:] if header else rows
    if header:
        lines.append(" | ".join(header))
    for row in body:
        texts = [text for text, _ in row]
        if header and len(header) == len(texts):
            lines.append(" | ".join(f"{h}: {t}" for h, t in zip(header, texts, strict=True)))
        else:
            lines.append(" | ".join(texts))
    return [line for line in lines if line]


def _squash(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _joined(section: Section | None) -> str | None:
    if section is None:
        return None
    return " ".join(section.all_blocks()) or None


def _related_numbers(section: Section | None) -> tuple[str, ...]:
    if section is None:
        return ()
    numbers = (_TA_NUMBER.match(block) for block in section.all_blocks())
    return tuple(sorted({match.group(1) for match in numbers if match}))
