"""decode_html picks the declared charset; parse_law_as_of reads Japanese era dates."""

from datetime import date

import pytest

from zeimu.ingest.encoding import decode_html
from zeimu.ingest.japanese_dates import parse_law_as_of


def test_defaults_to_utf8() -> None:
    assert decode_html("<p>相続税</p>".encode()) == "<p>相続税</p>"


def test_shift_jis_is_decoded_as_cp932() -> None:
    """① exists in cp932 (what Shift_JIS pages really use) but not in strict Shift_JIS."""
    html = '<meta charset="Shift_JIS"><p>①相続税</p>'

    assert decode_html(html.encode("cp932")) == html


def test_wrong_declared_charset_raises_instead_of_garbling() -> None:
    with pytest.raises(UnicodeDecodeError):
        decode_html('<meta charset="utf-8"><p>相続税</p>'.encode("cp932"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[令和8年4月1日現在法令等]", date(2026, 4, 1)),
        ("[令和８年７月１日現在法令等]", date(2026, 7, 1)),  # full-width digits
        ("令和元年5月1日現在法令等", date(2019, 5, 1)),  # 元年 = first year of the era
        ("平成31年4月30日現在法令等", date(2019, 4, 30)),
    ],
)
def test_parses_law_as_of(text: str, expected: date) -> None:
    assert parse_law_as_of(f"前文 {text} 後文") == expected


def test_no_law_as_of_line() -> None:
    assert parse_law_as_of("令和8年4月1日に死亡しました。") is None
