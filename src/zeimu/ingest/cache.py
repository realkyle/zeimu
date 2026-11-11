"""On-disk cache of fetched pages, so the NTA site is never re-fetched on every run."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict


class PageMeta(BaseModel):
    """Metadata saved next to each cached page.

    ``fetched_at`` matters because tax rules change: every chunk built from this page
    inherits it, so an answer can say how fresh its source was.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    status: int
    fetched_at: datetime
    sha256: str


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """A page body plus its metadata, and whether it came from the cache."""

    meta: PageMeta
    body: bytes
    from_cache: bool

    @property
    def url(self) -> str:
        """The URL the page was fetched from."""
        return self.meta.url


class RawPageCache:
    """Stores raw HTML at ``<root>/<host>/<path>`` with a ``.meta.json`` file beside it.

    Mirroring the URL path keeps the cache easy to browse by hand. URLs are validated
    first so a crafted URL (``..`` segments, query strings) can't write outside ``root``.
    """

    META_SUFFIX = ".meta.json"

    def __init__(self, root: Path) -> None:
        self._root = root

    def get(self, url: str) -> FetchedPage | None:
        """Return the cached page for ``url``, or ``None`` if it was never stored."""
        body_path = self._body_path(url)
        meta_path = self._meta_path(body_path)
        if not (body_path.is_file() and meta_path.is_file()):
            return None
        meta = PageMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        return FetchedPage(meta=meta, body=body_path.read_bytes(), from_cache=True)

    def iter_pages(self) -> Iterator[FetchedPage]:
        """Yield every cached page that has metadata, in a stable (sorted path) order."""
        for meta_path in sorted(self._root.rglob(f"*{self.META_SUFFIX}")):
            body_path = meta_path.with_name(meta_path.name.removesuffix(self.META_SUFFIX))
            if body_path.is_file():
                meta = PageMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
                yield FetchedPage(meta=meta, body=body_path.read_bytes(), from_cache=True)

    def put(self, url: str, *, status: int, fetched_at: datetime, body: bytes) -> FetchedPage:
        """Store ``body`` for ``url`` and return it as a (non-cached) ``FetchedPage``."""
        meta = PageMeta(
            url=url,
            status=status,
            fetched_at=fetched_at,
            sha256=hashlib.sha256(body).hexdigest(),
        )
        body_path = self._body_path(url)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        body_path.write_bytes(body)
        # Meta is written last: a crash mid-write leaves no meta, so get() treats it as a miss.
        self._meta_path(body_path).write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        return FetchedPage(meta=meta, body=body, from_cache=False)

    def _body_path(self, url: str) -> Path:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"not an http(s) URL: {url!r}")
        if parts.query:
            raise ValueError(f"URLs with a query string are not cacheable: {url!r}")
        # Check the raw segments: PurePosixPath would silently drop "." segments.
        if any(segment in ("..", ".") for segment in parts.path.split("/")):
            raise ValueError(f"URL path must not contain '.' or '..' segments: {url!r}")
        path = PurePosixPath(parts.path or "/")
        relative = path.relative_to("/") if path.is_absolute() else path
        if parts.path.endswith("/") or str(relative) == ".":
            relative = relative / "index.html"
        return self._root / parts.hostname / Path(*relative.parts)

    def _meta_path(self, body_path: Path) -> Path:
        return body_path.with_name(body_path.name + self.META_SUFFIX)
