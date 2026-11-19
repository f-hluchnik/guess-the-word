"""Game rules, independent of HTTP."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.puzzle import Puzzle, PuzzleCalendar

if TYPE_CHECKING:  # importing gensim takes seconds; keep it off the startup path
    from app.vocabulary import SemanticModel


class PuzzleNotAvailable(LookupError):
    """The puzzle number is before the first or after today's puzzle."""


@dataclass(frozen=True, slots=True)
class GuessResult:
    word: str
    # Vocabulary form the guess was measured as, when it differs from `word`.
    lemma: str | None
    similarity: float | None
    solved: bool

    @property
    def known(self) -> bool:
        return self.similarity is not None


class Game:
    def __init__(self, model: "SemanticModel", calendar: PuzzleCalendar) -> None:
        missing = [w for w in calendar.words if w not in model]
        if missing:
            raise ValueError(f"Secret words missing from the vectors: {', '.join(missing)}")
        self._model = model
        self._calendar = calendar

    def today(self) -> Puzzle:
        return self._calendar.get(self._calendar.current_number())

    def puzzle(self, number: int) -> Puzzle:
        """A published puzzle: today's or any earlier one, never a future one."""
        if not 1 <= number <= self._calendar.current_number():
            raise PuzzleNotAvailable(number)
        return self._calendar.get(number)

    @property
    def vocabulary_size(self) -> int:
        return len(self._model)

    def reference(self, puzzle: Puzzle) -> dict[int, float]:
        """Similarity of the 1st, 10th and 1000th closest words to the secret."""
        return self._model.reference(puzzle.secret)

    def warm_up(self) -> None:
        """Compute today's and tomorrow's reference values ahead of time.

        Each is a full pass over the vectors, several seconds on a small
        machine, which would otherwise land on a player's request.
        """
        today = self._calendar.current_number()
        for number in (today, today + 1):
            self._model.reference(self._calendar.get(number).secret)

    def public_id(self, puzzle: Puzzle) -> str:
        """Opaque id that changes whenever the puzzle's word could change."""
        return self._calendar.public_id(puzzle.number)

    def is_preview(self) -> bool:
        """True before the launch day: puzzle 1 is shown early."""
        return self._calendar.is_preview()

    def guess(self, puzzle: Puzzle, word: str) -> GuessResult:
        """Score an already normalised and validated word."""
        secret = puzzle.secret
        term = self._model.resolve(word)
        forms = {word, self._model.lemma(word)}
        if term is not None:
            forms |= {term, self._model.lemma(term)}  # "zidli" -> "židli" -> "židle"
        if secret in forms:
            return GuessResult(word, None if word == secret else secret, 1.0, solved=True)

        if term is None:
            return GuessResult(word, None, None, solved=False)
        return GuessResult(
            word,
            None if term == word else term,
            self._model.similarity(term, secret),
            solved=False,
        )
