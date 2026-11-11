"""Turn Japanese text into search terms: Sudachi words plus neighbouring-word pairs."""

from __future__ import annotations

import itertools
import unicodedata
from typing import Protocol

from sudachipy import Dictionary, SplitMode

#: Parts of speech dropped entirely: particles (の, は), auxiliary verbs (です, ます).
_DROP_POS = frozenset({"助詞", "助動詞"})
#: Parts of speech that also break a phrase: punctuation (、。「」) and whitespace.
_BREAK_POS = frozenset({"補助記号", "空白"})
#: Joins the two words of a pair term; Sudachi never emits a space inside a word.
PAIR_SEPARATOR = " "


class Tokenizer(Protocol):
    """Anything that turns text into a list of search terms."""

    def terms(self, text: str) -> list[str]:
        """Return the search terms for ``text``, in order, with repeats."""
        ...


class SudachiTokenizer:
    """Japanese tokenizer for BM25 (decided 2026-11-11).

    1. NFKC-normalise (full-width digits and letters become ASCII), then split each line
       with Sudachi in mode C (longest units: 基礎控除額 stays one word).
    2. Use Sudachi's normalised form, so 1,000万 and its full-width form both become 10000000.
    3. Drop particles and auxiliary verbs; punctuation and line breaks end a phrase.
    4. Emit every word, plus every pair of neighbouring words within a phrase
       (配偶者の税額軽減 → 配偶者, 税額, 軽減, "配偶者 税額", "税額 軽減"). The pairs make a
       chunk that contains the whole term outrank one with the words scattered, without
       a hand-maintained dictionary of terms.
    """

    def __init__(self, *, pairs: bool = True) -> None:
        # Loading the dictionary takes ~0.5 s, so build one tokenizer and reuse it.
        self._sudachi = Dictionary(dict="core").tokenizer(SplitMode.C)
        self._pairs = pairs

    def terms(self, text: str) -> list[str]:
        """Return words then word pairs for ``text`` (see class docstring)."""
        words: list[str] = []
        pairs: list[str] = []
        for line in unicodedata.normalize("NFKC", text).splitlines():
            phrase: list[str] = []
            for morpheme in self._sudachi.tokenize(line):
                pos = morpheme.part_of_speech()[0]
                if pos in _BREAK_POS:
                    self._add_pairs(phrase, pairs)
                    phrase = []
                    continue
                if pos in _DROP_POS:
                    continue
                word = morpheme.normalized_form().lower()
                words.append(word)
                phrase.append(word)
            self._add_pairs(phrase, pairs)
        return words + pairs

    def _add_pairs(self, phrase: list[str], out: list[str]) -> None:
        if self._pairs:
            out.extend(f"{a}{PAIR_SEPARATOR}{b}" for a, b in itertools.pairwise(phrase))
