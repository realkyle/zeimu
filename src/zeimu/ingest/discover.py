"""Find Tax Answer page URLs in NTA HTML (the code index and each page's Q&A links)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

from selectolax.lexbor import LexborHTMLParser

NTA_HOST = "www.nta.go.jp"
CODE_INDEX_URL = "https://www.nta.go.jp/taxes/shiraberu/taxanswer/code/index.htm"

#: Inheritance tax, gift tax and property valuation (PLAN.md §2, decided 2026-11-11).
CORPUS_CATEGORIES: tuple[str, ...] = ("sozoku", "zoyo", "hyoka")

_TAX_ANSWER_PATH = re.compile(
    r"^/taxes/shiraberu/taxanswer/(?P<category>[a-z]+)/(?P<number>\d{4})(?P<qa>_qa)?\.htm$"
)

PageKind = Literal["main", "qa"]


def find_tax_answer_links(
    html: str,
    base_url: str,
    *,
    categories: Iterable[str],
    kind: PageKind,
) -> list[str]:
    """Return sorted, de-duplicated absolute Tax Answer URLs linked from ``html``.

    Relative links are resolved against ``base_url``; fragments (``#q1``) are dropped so
    one Q&A page with several anchors is fetched once. Only NTA links in ``categories``
    of the requested ``kind`` (main page ``4152.htm`` or Q&A page ``4152_qa.htm``) count.
    """
    wanted = frozenset(categories)
    found: set[str] = set()
    for anchor in LexborHTMLParser(html).css("a[href]"):
        href = anchor.attributes.get("href")
        if not href:
            continue
        parts = urlsplit(urljoin(base_url, href))
        if parts.hostname != NTA_HOST:
            continue
        match = _TAX_ANSWER_PATH.match(parts.path)
        if match is None or match["category"] not in wanted:
            continue
        if (match["qa"] is not None) != (kind == "qa"):
            continue
        found.add(urlunsplit(("https", NTA_HOST, parts.path, "", "")))
    return sorted(found)
