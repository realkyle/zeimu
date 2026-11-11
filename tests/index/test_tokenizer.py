"""SudachiTokenizer: words, word pairs, normalization and phrase boundaries."""

import pytest

from zeimu.index.tokenizer import SudachiTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> SudachiTokenizer:
    return SudachiTokenizer()  # loading the dictionary is slow; share it


def test_particles_dropped_and_pairs_bridge_them(tokenizer: SudachiTokenizer) -> None:
    assert tokenizer.terms("配偶者の税額軽減") == [
        "配偶者",
        "税額",
        "軽減",
        "配偶者 税額",
        "税額 軽減",
    ]


def test_longest_units_keep_some_compounds_whole(tokenizer: SudachiTokenizer) -> None:
    assert "基礎控除額" in tokenizer.terms("基礎控除額")


def test_compound_term_pairs_are_kept(tokenizer: SudachiTokenizer) -> None:
    """Sudachi splits 相続時精算課税; the pairs preserve its word order for matching."""
    terms = tokenizer.terms("相続時精算課税")

    assert {"相続 時", "時 精算", "精算 課税"} <= set(terms)


def test_amounts_normalise_across_widths_and_formats(tokenizer: SudachiTokenizer) -> None:
    assert "30000000" in tokenizer.terms("３,０００万円")
    assert "30000000" in tokenizer.terms("3,000万円")


def test_punctuation_and_line_breaks_end_a_phrase(tokenizer: SudachiTokenizer) -> None:
    terms = tokenizer.terms("贈与税、相続税\n申告期限")

    assert "贈与 税" in terms
    assert not [t for t in terms if t.startswith("税 ")]  # no pair across 、 or newline


def test_pairs_can_be_switched_off(tokenizer: SudachiTokenizer) -> None:
    assert SudachiTokenizer(pairs=False).terms("配偶者の税額軽減") == ["配偶者", "税額", "軽減"]


def test_query_and_document_use_the_same_terms(tokenizer: SudachiTokenizer) -> None:
    document = "配偶者の税額の軽減を受けるための手続き"

    assert set(tokenizer.terms("税額の軽減")) <= set(tokenizer.terms(document))
