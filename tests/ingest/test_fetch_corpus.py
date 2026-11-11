"""fetch_corpus: index → main pages → linked Q&A pages, with failures skipped and reported."""

from pathlib import Path

import httpx

from zeimu.cli import fetch_corpus
from zeimu.ingest.cache import RawPageCache
from zeimu.ingest.discover import CODE_INDEX_URL
from zeimu.ingest.fetcher import PoliteFetcher

from .fakes import FakeClock, FakeServer

TA = "https://www.nta.go.jp/taxes/shiraberu/taxanswer"


def make_fetcher(server: FakeServer, tmp_path: Path) -> PoliteFetcher:
    clock = FakeClock()
    return PoliteFetcher(
        server.client(),
        RawPageCache(tmp_path),
        user_agent="zeimu-test/0.0",
        allowed_hosts=frozenset({"www.nta.go.jp"}),
        clock=clock.time,
        sleep=clock.sleep,
    )


def html(body: str) -> httpx.Response:
    return httpx.Response(200, content=f"<html><body>{body}</body></html>".encode())


def test_fetches_index_pages_and_qa_and_reports_failures(tmp_path: Path) -> None:
    server = FakeServer(
        pages={
            CODE_INDEX_URL: html(
                '<a href="../sozoku/4152.htm">a</a>'
                '<a href="../zoyo/4402.htm">b</a>'
                '<a href="../sozoku/4999.htm">missing</a>'
            ),
            f"{TA}/sozoku/4152.htm": html('<a href="4152_qa.htm#q1">Q1</a>'),
            f"{TA}/zoyo/4402.htm": html("no links"),
            f"{TA}/sozoku/4152_qa.htm": html("qa"),
        }
    )

    ok, failed = fetch_corpus(make_fetcher(server, tmp_path), refresh=False)

    assert ok == 3  # 4152, 4402, 4152_qa
    assert failed == [f"{TA}/sozoku/4999.htm"]
    assert RawPageCache(tmp_path).get(f"{TA}/sozoku/4152_qa.htm") is not None


def test_second_run_uses_cache_only(tmp_path: Path) -> None:
    server = FakeServer(
        pages={
            CODE_INDEX_URL: html('<a href="../sozoku/4152.htm">a</a>'),
            f"{TA}/sozoku/4152.htm": html("page"),
        }
    )
    fetch_corpus(make_fetcher(server, tmp_path), refresh=False)
    server.requests.clear()

    second = make_fetcher(server, tmp_path)
    fetch_corpus(second, refresh=False)

    assert server.requests == []
    assert second.network_requests == 0
