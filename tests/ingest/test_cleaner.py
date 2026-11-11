"""clean_page: metadata, kept content (image alt text, tables) and dropped boilerplate."""

from datetime import date

import pytest

from zeimu.ingest.cleaner import CleanError, CleanPage, clean_page

from .sample_pages import MAIN_PAGE


@pytest.fixture
def page() -> CleanPage:
    return clean_page(MAIN_PAGE)


def test_title_has_normalised_spacing(page: CleanPage) -> None:
    assert page.title == "No.9999 テストの計算"


def test_page_metadata(page: CleanPage) -> None:
    assert page.law_as_of == date(2026, 4, 1)
    assert page.tax_type == "相続税"
    assert page.legal_basis == "相法15、16"
    assert page.related == ("4102", "4155")


def test_only_content_sections_remain(page: CleanPage) -> None:
    assert [s.heading for s in page.sections] == ["概要"]
    assert [c.heading for c in page.sections[0].children] == ["総額の計算", "税率"]


def test_inline_link_text_kept_in_one_block(page: CleanPage) -> None:
    assert page.sections[0].blocks == ["計算は次の順序で行います。"]


def test_image_alt_text_kept_and_decorative_images_skipped(page: CleanPage) -> None:
    total = page.sections[0].children[0]

    assert total.blocks == [
        "基礎控除額（3,000万円 ＋ 600万円 × 法定相続人の数）",  # source text unchanged
        "[図: 課税価格 － 基礎控除額 ＝ 課税遺産総額]",
    ]


def test_table_rows_carry_their_headers(page: CleanPage) -> None:
    rates = page.sections[0].children[1]

    assert rates.blocks == [
        "速算表",
        "取得金額 | 税率 | 控除額",
        "取得金額: 1,000万円以下 | 税率: 10％ | 控除額: －",
        "取得金額: 3,000万円以下 | 税率: 15％ | 控除額: 50万円",
    ]


def test_no_boilerplate_script_or_comments_anywhere(page: CleanPage) -> None:
    text = " ".join(block for s in page.sections for block in s.all_blocks())

    for unwanted in ("相談窓口", "var x", "comment", "ホーム", "現在法令等"):
        assert unwanted not in text


def test_page_without_body_area_is_rejected() -> None:
    with pytest.raises(CleanError, match="bodyArea"):
        clean_page("<html><body><h1>x</h1></body></html>")
