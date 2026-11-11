"""Command-line entry point: ``uv run zeimu <command>``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import httpx

from zeimu.ingest.cache import RawPageCache
from zeimu.ingest.discover import (
    CODE_INDEX_URL,
    CORPUS_CATEGORIES,
    NTA_HOST,
    find_tax_answer_links,
)
from zeimu.ingest.fetcher import FetchError, PoliteFetcher

USER_AGENT = "zeimu/0.1 (+https://github.com/realkyle/zeimu)"
DEFAULT_RAW_DIR = Path("data/raw")

logger = logging.getLogger("zeimu")


def build_fetcher(client: httpx.Client, raw_dir: Path, min_interval: float) -> PoliteFetcher:
    """Wire up the production fetcher for the NTA site."""
    return PoliteFetcher(
        client,
        RawPageCache(raw_dir),
        user_agent=USER_AGENT,
        allowed_hosts=frozenset({NTA_HOST}),
        min_interval=min_interval,
    )


def fetch_corpus(fetcher: PoliteFetcher, *, refresh: bool) -> tuple[int, list[str]]:
    """Fetch the code index, every corpus page and the Q&A pages they link to.

    Returns the number of pages now cached and the URLs that failed. A failed page is
    logged and skipped so one bad link doesn't lose the rest of the run.
    """
    index = fetcher.fetch(CODE_INDEX_URL, refresh=refresh)
    main_urls = find_tax_answer_links(
        index.body.decode("utf-8"), CODE_INDEX_URL, categories=CORPUS_CATEGORIES, kind="main"
    )
    failed: list[str] = []
    qa_urls: set[str] = set()
    ok = 0
    for url in main_urls:
        try:
            page = fetcher.fetch(url, refresh=refresh)
        except FetchError as error:
            logger.warning("skipped: %s", error)
            failed.append(url)
            continue
        ok += 1
        qa_urls.update(
            find_tax_answer_links(
                page.body.decode("utf-8"), url, categories=CORPUS_CATEGORIES, kind="qa"
            )
        )
    for url in sorted(qa_urls):
        try:
            fetcher.fetch(url, refresh=refresh)
            ok += 1
        except FetchError as error:
            logger.warning("skipped: %s", error)
            failed.append(url)
    logger.info(
        "%d main pages, %d Q&A pages, %d cached OK, %d failed, %d network requests",
        len(main_urls),
        len(qa_urls),
        ok,
        len(failed),
        fetcher.network_requests,
    )
    return ok, failed


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the chosen command. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog="zeimu")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="fetch NTA Tax Answer pages into the raw cache")
    fetch.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    fetch.add_argument("--min-interval", type=float, default=1.5, help="seconds between requests")
    fetch.add_argument("--refresh", action="store_true", help="re-fetch pages already cached")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # our fetcher already logs each GET
    with httpx.Client(timeout=30.0) as client:
        fetcher = build_fetcher(client, args.raw_dir, args.min_interval)
        _, failed = fetch_corpus(fetcher, refresh=args.refresh)
    return 1 if failed else 0
