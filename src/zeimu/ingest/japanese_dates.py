"""Parse the "law as of" line on NTA pages, e.g. ``[令和8年4月1日現在法令等]``."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

#: First Gregorian year of each era minus 1, so era year N is ``offset + N``.
_ERA_OFFSETS = {"令和": 2018, "平成": 1988}

_LAW_AS_OF = re.compile(r"(令和|平成)(元|\d+)年(\d+)月(\d+)日現在法令等")


def parse_law_as_of(text: str) -> date | None:
    """Return the date the page's law is current as of, or ``None`` if there is no such line.

    Handles 元年 (the first year of an era) and full-width digits (令和８年).
    """
    match = _LAW_AS_OF.search(unicodedata.normalize("NFKC", text))
    if match is None:
        return None
    era, year, month, day = match.groups()
    era_year = 1 if year == "元" else int(year)
    return date(_ERA_OFFSETS[era] + era_year, int(month), int(day))
