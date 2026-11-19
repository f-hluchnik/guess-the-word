"""Game rules, independent of HTTP."""

from dataclasses import dataclass

from app.puzzle import Puzzle, PuzzleCalendar
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
    def __init__(self, model: SemanticModel, calendar: PuzzleCalendar) -> None:
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
