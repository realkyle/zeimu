"""chunk_page: section-based chunks, size-based splitting, Q&A grouping and stable IDs."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from zeimu.ingest.chunker import PageSource, chunk_page
from zeimu.ingest.cleaner import CleanPage, Section, clean_page

from .sample_pages import MAIN_PAGE, QA_PAGE

WHEN = datetime(2026, 11, 11, tzinfo=UTC)
MAIN = PageSource(url="https://x/9999.htm", ta_no="9999", page_kind="main", fetched_at=WHEN)
QA = PageSource(url="https://x/9999_qa.htm", ta_no="9999", page_kind="qa", fetched_at=WHEN)


def page_with(*sections: Section) -> CleanPage:
    return CleanPage(
        title="No.9999 テスト",
        law_as_of=None,
        tax_type=None,
        legal_basis=None,
        related=(),
        sections=sections,
    )


def test_small_section_is_one_chunk_with_context_header() -> None:
    chunks = chunk_page(clean_page(MAIN_PAGE), MAIN)

    assert len(chunks) == 1
    lines = chunks[0].text.split("\n")
    assert lines[0] == "No.9999 テストの計算 > 概要"
    assert "【税率】" in lines
    assert chunks[0].heading_path == ("概要",)
    assert chunks[0].legal_basis == "相法15、16"


def test_long_section_splits_at_subheadings() -> None:
    intro, part_a, part_b = "前書き", "あ" * 60, "い" * 60
    section = Section(
        "概要",
        2,
        blocks=[intro],
        children=[Section("A", 3, blocks=[part_a]), Section("B", 3, blocks=[part_b])],
    )

    chunks = chunk_page(page_with(section), MAIN, max_chars=100)

    assert [c.heading_path for c in chunks] == [("概要",), ("概要", "A"), ("概要", "B")]
    assert chunks[1].text == f"No.9999 テスト > 概要 > A\n{part_a}"


def test_long_leaf_section_splits_between_paragraphs() -> None:
    paragraphs: list[str] = ["あ" * 40, "い" * 40, "う" * 40]

    chunks = chunk_page(page_with(Section("概要", 2, blocks=paragraphs)), MAIN, max_chars=100)

    assert [c.text.split("\n")[1:] for c in chunks] == [paragraphs[:2], paragraphs[2:]]
    assert chunks[1].id == f"{chunks[0].id}-p2"


def test_paragraph_longer_than_limit_is_kept_whole() -> None:
    huge = "あ" * 300

    chunks = chunk_page(page_with(Section("概要", 2, blocks=[huge])), MAIN, max_chars=100)

    assert [c.text.split("\n")[1] for c in chunks] == [huge]


def test_split_qa_repeats_the_question_in_every_piece() -> None:
    page = clean_page(QA_PAGE.format(long_answer="答" * 80))

    chunks = chunk_page(page, QA, max_chars=100)

    assert [c.heading_path[-1] for c in chunks] == ["Ａ1", "1 課税価格"]
    for chunk in chunks:
        assert chunk.text.split("\n")[1] == "Q: 各相続人の課税価格はいくらですか。①の場合。"


def test_short_qa_keeps_question_and_answer_together() -> None:
    page = clean_page(QA_PAGE.format(long_answer="答え"))

    (chunk,) = chunk_page(page, QA)

    assert "【Ｑ1】" in chunk.text
    assert "【Ａ1】" in chunk.text
    assert chunk.id.startswith("ta9999qa-")


def test_ids_depend_on_headings_not_position() -> None:
    a = Section("A", 2, blocks=["a"])
    b = Section("B", 2, blocks=["b"])

    def ids(*sections: Section) -> dict[tuple[str, ...], str]:
        return {c.heading_path: c.id for c in chunk_page(page_with(*sections), MAIN)}

    before = ids(a, b)
    after = ids(Section("NEW", 2, blocks=["n"]), a, b)  # NTA inserts a section on top

    assert after[("A",)] == before[("A",)]
    assert after[("B",)] == before[("B",)]
    assert before[("A",)] != before[("B",)]
    assert before[("A",)].startswith("ta9999-")


def test_chunks_are_immutable() -> None:
    (chunk,) = chunk_page(clean_page(MAIN_PAGE), MAIN)

    with pytest.raises(ValidationError):
        chunk.text = "changed"  # pyrefly: ignore[read-only]
