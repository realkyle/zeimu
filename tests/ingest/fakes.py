"""Test doubles: a fake clock and a fake NTA server, so fetcher tests run offline and instantly."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass
class FakeClock:
    """Monotonic clock that only moves when ``sleep`` is called (or ``advance``)."""

    now: float = 1000.0
    sleeps: list[float] = field(default_factory=list)

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeServer:
    """Serves canned responses by URL and records every request it receives."""

    pages: dict[str, httpx.Response] = field(default_factory=dict)
    robots_txt: str | None = "User-agent: *\nDisallow: /private/\n"
    requests: list[httpx.Request] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if request.url.path == "/robots.txt":
            if self.robots_txt is None:
                return httpx.Response(404)
            return httpx.Response(200, text=self.robots_txt)
        return self.pages.get(url, httpx.Response(404))

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]
