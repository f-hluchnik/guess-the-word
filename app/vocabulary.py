"""Word vectors and lemmatisation behind a small, game-shaped interface."""

import logging
from functools import lru_cache
from pathlib import Path

import simplemma
from gensim.models import KeyedVectors

from app.text import is_word

log = logging.getLogger(__name__)

# Players see how similar the 1st, 10th and 1000th closest words are to the
# secret, as a scale for their own guesses. The words themselves stay hidden.
REFERENCE_RANKS = (1, 10, 1000)
# Extra neighbours to fetch, since inflections of the secret are skipped.
_INFLECTION_SLACK = 300
# Czech noun case endings. simplemma doesn't know every inflected form (it has
# "kominík" but not "kominíka"), so these catch the forms it misses.
_NOUN_ENDINGS = frozenset(
    {"a", "u", "e", "ě", "y", "i", "í", "o", "ou", "ů", "em", "ům", "ám", "ím"}
    | {"ech", "ích", "ách", "ami", "emi", "mi", "ovi", "ové"}
)


class SemanticModel:
    def __init__(self, vectors: KeyedVectors) -> None:
        self._vectors = vectors
        # Precompute unit norms now so the first request doesn't pay for it.
        vectors.fill_norms()
        self.reference = lru_cache(maxsize=32)(self._reference)

    @classmethod
    def load(cls, path: Path) -> "SemanticModel":
        log.info("Loading word vectors from %s", path)
        # mmap keeps the matrix in the page cache, shared between workers.
        model = cls(KeyedVectors.load(str(path), mmap="r"))
        log.info("Loaded %d word vectors", len(model))
        return model

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, word: str) -> bool:
        return word in self._vectors

    def rank(self, word: str) -> int:
        """Frequency rank, 0 = most common (fastText vectors are frequency-sorted)."""
        return self._vectors.key_to_index[word]

    @staticmethod
    def lemma(word: str) -> str:
        return simplemma.lemmatize(word, lang="cs")

    def resolve(self, word: str) -> str | None:
        """The vocabulary entry to measure a guess by, or None if unknown.

        The exact form wins over the lemma: fastText has vectors for inflected
        forms, and simplemma occasionally maps a base form to a different word
        (e.g. "rána" -> "ráno"). The lemma is only a fallback for rare forms.
        """
        if word in self._vectors:
            return word
        lemma = self.lemma(word)
        return lemma if lemma in self._vectors else None

    def similarity(self, a: str, b: str) -> float:
        return float(self._vectors.similarity(a, b))

    def is_form_of(self, word: str, secret: str) -> bool:
        """True if `word` looks like an inflected form of `secret`.

        Used only to keep inflections out of the reference scale; solving a
        puzzle uses the stricter lemma check in Game, since this heuristic
        accepts homographs ("kosa" as a form of "kos").
        """
        if word == secret or self.lemma(word) == secret:
            return True
        return any(
            word.startswith(stem) and word[len(stem) :] in _NOUN_ENDINGS
            for stem in (secret, secret[:-1])
        )

    def _reference(self, secret: str) -> dict[int, float]:
        """Similarity of the n-th closest *different* word, for each reference rank.

        Inflections ("jizvy", "jizvou" for "jizva") are skipped: they are always
        the nearest vectors and would make every scale meaningless. Ranks the
        vocabulary can't fill are left out.
        """
        topn = min(len(self._vectors), max(REFERENCE_RANKS) + _INFLECTION_SLACK)
        scores = [
            float(score)
            for word, score in self._vectors.most_similar(secret, topn=topn)
            if is_word(word) and not self.is_form_of(word, secret)
        ]
        return {rank: scores[rank - 1] for rank in REFERENCE_RANKS if rank <= len(scores)}

    def ceiling(self, secret: str) -> float | None:
        """Similarity of the closest different word."""
        return self.reference(secret).get(1)
