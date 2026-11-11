"""RawPageCache: round-trips pages and refuses URLs that could escape the cache folder."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from zeimu.ingest.cache import RawPageCache

URL = "https://www.nta.go.jp/taxes/shiraberu/taxanswer/sozoku/4152.htm"
WHEN = datetime(2026, 11, 11, 3, 0, tzinfo=UTC)


def test_miss_returns_none(tmp_path: Path) -> None:
    assert RawPageCache(tmp_path).get(URL) is None


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = RawPageCache(tmp_path)
    stored = cache.put(URL, status=200, fetched_at=WHEN, body="相続税".encode())

    loaded = cache.get(URL)

    assert loaded is not None
    assert loaded.body == "相続税".encode()
    assert loaded.meta == stored.meta
    assert loaded.meta.fetched_at == WHEN
    assert loaded.from_cache is True
    assert stored.from_cache is False


def test_layout_mirrors_url_path(tmp_path: Path) -> None:
    RawPageCache(tmp_path).put(URL, status=200, fetched_at=WHEN, body=b"x")

    page = tmp_path / "www.nta.go.jp/taxes/shiraberu/taxanswer/sozoku/4152.htm"
    assert page.read_bytes() == b"x"
    assert page.with_name("4152.htm.meta.json").is_file()


def test_body_without_meta_is_a_miss(tmp_path: Path) -> None:
    """A crash between writing body and meta must not yield a page with no fetch date."""
    page = tmp_path / "www.nta.go.jp/taxes/shiraberu/taxanswer/sozoku/4152.htm"
    page.parent.mkdir(parents=True)
    page.write_bytes(b"partial")

    assert RawPageCache(tmp_path).get(URL) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://www.nta.go.jp/taxes/../../etc/passwd",
        "https://www.nta.go.jp/a/./b.htm",
        "https://www.nta.go.jp/page.htm?x=1",
        "file:///etc/passwd",
        "not a url",
    ],
)
def test_rejects_unsafe_urls(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - message varies by reason
        RawPageCache(tmp_path).put(url, status=200, fetched_at=WHEN, body=b"x")
