"""A polite HTTP fetcher: robots.txt-aware, rate-limited, host-restricted and cached."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from zeimu.ingest.cache import FetchedPage, RawPageCache

logger = logging.getLogger(__name__)

#: PLAN.md §2 requires at least one second between requests.
MIN_ALLOWED_INTERVAL = 1.0


class FetchError(Exception):
    """The server answered with something other than 200 OK (redirects included)."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"GET {url} returned HTTP {status}")
        self.url = url
        self.status = status


class RobotsDisallowedError(Exception):
    """robots.txt forbids fetching this URL."""


class PoliteFetcher:
    """Fetches pages the way PLAN.md §2 requires.

    - Only hosts in ``allowed_hosts`` can be fetched.
    - robots.txt is read once per host and obeyed (including any ``Crawl-delay``).
    - Requests are at least ``min_interval`` seconds apart.
    - Every request carries ``user_agent``.
    - Pages are cached; a cached page is returned without touching the network
      unless ``refresh=True``.
    - Redirects are not followed: NTA answers a missing page with a 302 to its 404
      page, which must not be cached as if it were content.

    The HTTP client, clocks and sleep are injected so tests run instantly and offline.
    """

    def __init__(
        self,
        client: httpx.Client,
        cache: RawPageCache,
        *,
        user_agent: str,
        allowed_hosts: frozenset[str],
        min_interval: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if min_interval < MIN_ALLOWED_INTERVAL:
            raise ValueError(f"min_interval must be >= {MIN_ALLOWED_INTERVAL}s")
        self._client = client
        self._cache = cache
        self._user_agent = user_agent
        self._allowed_hosts = allowed_hosts
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._now = now
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request_at: float | None = None
        self.network_requests = 0

    def fetch(self, url: str, *, refresh: bool = False) -> FetchedPage:
        """Return the page at ``url``, from the cache when possible.

        Raises:
            ValueError: the host is not allowed.
            RobotsDisallowedError: robots.txt forbids the URL.
            FetchError: the server did not answer 200 OK.
        """
        host = self._check_host(url)
        if not refresh and (cached := self._cache.get(url)) is not None:
            return cached
        robots = self._robots_for(host, url)
        if not robots.can_fetch(self._user_agent, url):
            raise RobotsDisallowedError(url)
        response = self._get(url, interval=self._interval_for(robots))
        if response.status_code != httpx.codes.OK:
            raise FetchError(url, response.status_code)
        return self._cache.put(
            url, status=response.status_code, fetched_at=self._now(), body=response.content
        )

    def _check_host(self, url: str) -> str:
        host = urlsplit(url).hostname
        if host is None or host not in self._allowed_hosts:
            raise ValueError(f"host not allowed: {url!r}")
        return host

    def _robots_for(self, host: str, url: str) -> RobotFileParser:
        if host not in self._robots:
            scheme = urlsplit(url).scheme
            response = self._get(f"{scheme}://{host}/robots.txt", interval=self._min_interval)
            rules: list[str]
            if response.status_code == httpx.codes.OK:
                rules = response.text.splitlines()
            elif response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
                # Same convention as RobotFileParser.read(): an auth error means "keep out".
                rules = ["User-agent: *", "Disallow: /"]
            else:
                rules = []  # no robots.txt means no restrictions
            parser = RobotFileParser()
            parser.parse(rules)
            self._robots[host] = parser
        return self._robots[host]

    def _interval_for(self, robots: RobotFileParser) -> float:
        delay = robots.crawl_delay(self._user_agent)
        return max(self._min_interval, float(delay)) if delay is not None else self._min_interval

    def _get(self, url: str, *, interval: float) -> httpx.Response:
        if self._last_request_at is not None:
            wait = interval - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        logger.info("GET %s", url)
        try:
            return self._client.get(
                url, headers={"User-Agent": self._user_agent}, follow_redirects=False
            )
        finally:
            # Counted even on failure: a failed request still hit the server.
            self._last_request_at = self._clock()
            self.network_requests += 1
