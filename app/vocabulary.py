"""Word vectors and lemmatisation behind a small, game-shaped interface."""

import json
import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
import simplemma
from gensim.models import KeyedVectors

from app.text import is_word
from app.vectorfiles import aliases_file, norms_file, references_file

log = logging.getLogger(__name__)

# Players see how similar the 1st, 10th and 1000th closest words are to the
# secret, as a scale for their own guesses. The words themselves stay hidden.
REFERENCE_RANKS = (1, 10, 1000)
# Rows per chunk when computing vector norms; see compute_norms().
_NORM_CHUNK = 20_000
# Extra neighbours to fetch, since inflections of the secret are skipped.
_INFLECTION_SLACK = 300
# Czech noun case endings. simplemma doesn't know every inflected form (it has
# "kominík" but not "kominíka"), so these catch the forms it misses.
_NOUN_ENDINGS = frozenset(
    {"a", "u", "e", "ě", "y", "i", "í", "o", "ou", "ů", "em", "ům", "ám", "ím"}
    | {"ech", "ích", "ách", "ami", "emi", "mi", "ovi", "ové"}
)


def compute_norms(matrix: np.ndarray) -> np.ndarray:
    """Row lengths, in chunks: gensim's fill_norms() squares the whole matrix
    at once, a temporary as large as the vectors (~420 MB)."""
    return np.concatenate(
        [
            np.linalg.norm(matrix[i : i + _NORM_CHUNK], axis=1)
            for i in range(0, len(matrix), _NORM_CHUNK)
        ]
    )


class SemanticModel:
    def __init__(
        self,
        vectors: KeyedVectors,
        aliases: Mapping[str, str] | None = None,
        norms: np.ndarray | None = None,
        references: Mapping[str, Mapping[int, float]] | None = None,
    ) -> None:
        self._vectors = vectors
        self._precomputed = dict(references or {})
        # Spellings without diacritics ("zidle") -> the dictionary word ("židle").
        self._aliases = dict(aliases or {})
        vectors.norms = compute_norms(vectors.vectors) if norms is None else norms
        self.reference = lru_cache(maxsize=32)(self._reference)

    @classmethod
    def load(cls, path: Path) -> "SemanticModel":
        log.info("Loading word vectors from %s", path)
        aliases_path, norms_path = aliases_file(path), norms_file(path)
        aliases = (
            json.loads(aliases_path.read_text(encoding="utf-8")) if aliases_path.exists() else {}
        )
        norms = np.load(norms_path) if norms_path.exists() else None
        references_path = references_file(path)
        references = (
            {
                word: {int(rank): score for rank, score in ranks.items()}
                for word, ranks in json.loads(references_path.read_text(encoding="utf-8")).items()
            }
            if references_path.exists()
            else {}
        )
        # mmap keeps the matrix in the page cache, shared between workers.
        model = cls(KeyedVectors.load(str(path), mmap="r"), aliases, norms, references)
        log.info("Loaded %d word vectors and %d spelling aliases", len(model), len(aliases))
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

        In order of preference:
        1. The word itself. fastText has vectors for inflected forms, and
           simplemma occasionally maps a base form to a different word
           (e.g. "rána" -> "ráno"), so the exact form wins.
        2. The dictionary word it is a diacritic-free spelling of ("zidle" ->
           "židle"). Those spellings have vectors of their own, learnt from
           text typed without diacritics, and would score unrelated to the
           word the player means.
        3. Its lemma, for rare forms.
        """
        if word in self._vectors:
            return word
        if word in self._aliases:
            return self._aliases[word]
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
        if secret in self._precomputed:
            return dict(self._precomputed[secret])
        return self.compute_reference(secret)

    def compute_reference(self, secret: str) -> dict[int, float]:
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
