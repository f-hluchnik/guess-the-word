"""Input normalisation shared by the API and the vector build script."""

import unicodedata

MAX_WORD_LENGTH = 40

CZECH_LETTERS = frozenset("aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž")


def normalize(raw: str) -> str:
    """NFC-normalise, trim and lowercase.

    NFC matters: macOS and some mobile keyboards emit decomposed diacritics
    ("c" + combining caron), which would otherwise never match the vocabulary.
    """
    return unicodedata.normalize("NFC", raw).strip().lower()


def is_word(text: str) -> bool:
    """True for a single, already-normalised Czech word made of letters only."""
    return 0 < len(text) <= MAX_WORD_LENGTH and CZECH_LETTERS.issuperset(text)


def strip_diacritics(word: str) -> str:
    """ "židle" -> "zidle"."""
    decomposed = unicodedata.normalize("NFD", word)
    return "".join(c for c in decomposed if not unicodedata.combining(c))
