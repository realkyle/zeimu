"""Decode raw NTA HTML: main pages are UTF-8, Q&A pages are Shift_JIS."""

from __future__ import annotations

import re

_META_CHARSET = re.compile(rb"""charset\s*=\s*["']?([A-Za-z0-9_.:-]+)""", re.IGNORECASE)

#: Shift_JIS as labelled on the web is really Microsoft's cp932 in practice: it adds
#: characters such as ① and ㈱ that strict Shift_JIS decoding rejects.
_ALIASES = {"shift_jis": "cp932", "shift-jis": "cp932", "sjis": "cp932", "x-sjis": "cp932"}


def decode_html(body: bytes) -> str:
    """Decode ``body`` using the charset declared in its ``<meta>`` tag (default UTF-8).

    Decoding is strict: a wrongly declared charset raises ``UnicodeDecodeError`` instead
    of silently producing garbled text that would end up in the corpus.
    """
    match = _META_CHARSET.search(body[:4096])
    declared = match.group(1).decode("ascii").lower() if match else "utf-8"
    return body.decode(_ALIASES.get(declared, declared))
