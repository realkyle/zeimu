"""PoliteFetcher: robots.txt, rate limiting, caching, host allowlist, User-Agent, errors."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from zeimu.ingest.cache import RawPageCache
from zeimu.ingest.fetcher import FetchError, PoliteFetcher, RobotsDisallowedError

from .fakes import FakeClock, FakeServer

HOST = "www.nta.go.jp"
PAGE = f"https://{HOST}/taxes/shiraberu/taxanswer/sozoku/4152.htm"
OTHER = f"https://{HOST}/taxes/shiraberu/taxanswer/sozoku/4155.htm"
UA = "zeimu-test/0.0"
WHEN = datetime(2026, 11, 11, 3, 0, tzinfo=UTC)


def make_fetcher(
    server: FakeServer, tmp_path: Path, clock: FakeClock, min_interval: float = 1.5
) -> PoliteFetcher:
    return PoliteFetcher(
        server.client(),
        RawPageCache(tmp_path),
        user_agent=UA,
        allowed_hosts=frozenset({HOST}),
        min_interval=min_interval,
        clock=clock.time,
        sleep=clock.sleep,
        now=lambda: WHEN,
    )


@pytest.fixture
def server() -> FakeServer:
    return FakeServer(
        pages={
            PAGE: httpx.Response(200, content="相続税の計算".encode()),
            OTHER: httpx.Response(200, content=b"rates"),
        }
    )


def test_fetches_and_records_metadata(server: FakeServer, tmp_path: Path) -> None:
    page = make_fetcher(server, tmp_path, FakeClock()).fetch(PAGE)

    assert page.body == "相続税の計算".encode()
    assert page.meta.status == 200
    assert page.meta.fetched_at == WHEN
    assert page.from_cache is False


def test_reads_robots_once_then_page(server: FakeServer, tmp_path: Path) -> None:
    fetcher = make_fetcher(server, tmp_path, FakeClock())
    fetcher.fetch(PAGE)
    fetcher.fetch(OTHER)

    assert server.urls() == [f"https://{HOST}/robots.txt", PAGE, OTHER]


def test_sends_user_agent_on_every_request(server: FakeServer, tmp_path: Path) -> None:
    make_fetcher(server, tmp_path, FakeClock()).fetch(PAGE)

    assert [r.headers["User-Agent"] for r in server.requests] == [UA, UA]


def test_waits_min_interval_between_requests(server: FakeServer, tmp_path: Path) -> None:
    clock = FakeClock()
    fetcher = make_fetcher(server, tmp_path, clock, min_interval=2.0)
    fetcher.fetch(PAGE)  # robots.txt, then PAGE after a 2 s wait
    fetcher.fetch(OTHER)  # another 2 s wait

    assert clock.sleeps == [2.0, 2.0]


def test_only_waits_the_remaining_time(server: FakeServer, tmp_path: Path) -> None:
    clock = FakeClock()
    fetcher = make_fetcher(server, tmp_path, clock, min_interval=2.0)
    fetcher.fetch(PAGE)
    clock.advance(1.5)  # e.g. time spent parsing the page
    fetcher.fetch(OTHER)

    assert clock.sleeps == [2.0, pytest.approx(0.5)]


def test_obeys_crawl_delay_when_longer(server: FakeServer, tmp_path: Path) -> None:
    server.robots_txt = "User-agent: *\nCrawl-delay: 5\n"
    clock = FakeClock()
    make_fetcher(server, tmp_path, clock, min_interval=1.5).fetch(PAGE)

    assert clock.sleeps == [5.0]


def test_cached_page_skips_network(server: FakeServer, tmp_path: Path) -> None:
    make_fetcher(server, tmp_path, FakeClock()).fetch(PAGE)
    server.requests.clear()

    again = make_fetcher(server, tmp_path, FakeClock()).fetch(PAGE)

    assert again.from_cache is True
    assert again.meta.fetched_at == WHEN
    assert server.requests == []


def test_refresh_refetches(server: FakeServer, tmp_path: Path) -> None:
    make_fetcher(server, tmp_path, FakeClock()).fetch(PAGE)
    server.requests.clear()

    page = make_fetcher(server, tmp_path, FakeClock()).fetch(PAGE, refresh=True)

    assert page.from_cache is False
    assert PAGE in server.urls()


def test_robots_disallow_is_refused(server: FakeServer, tmp_path: Path) -> None:
    blocked = f"https://{HOST}/private/secret.htm"
    server.pages[blocked] = httpx.Response(200, content=b"x")

    with pytest.raises(RobotsDisallowedError):
        make_fetcher(server, tmp_path, FakeClock()).fetch(blocked)
    assert blocked not in server.urls()


def test_missing_robots_txt_allows_all(server: FakeServer, tmp_path: Path) -> None:
    server.robots_txt = None

    assert make_fetcher(server, tmp_path, FakeClock()).fetch(PAGE).body


def test_other_hosts_are_refused_without_a_request(server: FakeServer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="host not allowed"):
        make_fetcher(server, tmp_path, FakeClock()).fetch("https://example.com/x.htm")
    assert server.requests == []


def test_redirect_is_an_error_and_not_cached(server: FakeServer, tmp_path: Path) -> None:
    """NTA redirects missing pages to its 404 page; that must never be cached as content."""
    server.pages[PAGE] = httpx.Response(302, headers={"Location": "/error/404.htm"})
    fetcher = make_fetcher(server, tmp_path, FakeClock())

    with pytest.raises(FetchError) as error:
        fetcher.fetch(PAGE)
    assert error.value.status == 302
    assert RawPageCache(tmp_path).get(PAGE) is None


def test_rejects_interval_below_one_second(server: FakeServer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_interval"):
        make_fetcher(server, tmp_path, FakeClock(), min_interval=0.5)
