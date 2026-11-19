"""Which word is secret on which day.

The schedule lives in data/words.toml: an epoch (the day of puzzle no. 1), a
time zone, and batches of words. Batches are played in file order. Within a
batch, the order is a shuffle keyed by PUZZLE_SECRET, so the public word list
doesn't reveal which word falls on which day.

Appending a batch never changes an earlier puzzle. When every batch has been
played, the whole list is reshuffled and played again.
"""

import hashlib
import hmac
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.text import is_word, normalize


class WordListError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Puzzle:
    number: int
    date: date
    secret: str


@dataclass(frozen=True)
class WordList:
    epoch: date
    timezone: ZoneInfo
    batches: tuple[tuple[str, ...], ...]

    @classmethod
    def load(cls, path: Path) -> "WordList":
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise WordListError(f"{path}: {error}") from None
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict, source: str = "word list") -> "WordList":
        epoch = data.get("epoch")
        if type(epoch) is not date:
            raise WordListError(f"{source}: 'epoch' must be a date like 2026-11-16.")
        try:
            timezone = ZoneInfo(data.get("timezone", ""))
        except (ZoneInfoNotFoundError, ValueError):
            raise WordListError(f"{source}: 'timezone' must be an IANA name.") from None

        batches = []
        seen: dict[str, int] = {}
        for number, batch in enumerate(data.get("batch", []), start=1):
            words = batch.get("words", [])
            if not words:
                raise WordListError(f"{source}: batch {number} has no words.")
            for word in words:
                if not isinstance(word, str) or normalize(word) != word or not is_word(word):
                    raise WordListError(
                        f"{source}: batch {number}: {word!r} must be one lowercase Czech word."
                    )
                if word in seen:
                    raise WordListError(
                        f"{source}: {word!r} is in batch {seen[word]} and batch {number}."
                    )
                seen[word] = number
            batches.append(tuple(words))
        if not batches:
            raise WordListError(f"{source}: add at least one [[batch]].")
        return cls(epoch, timezone, tuple(batches))

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(word for batch in self.batches for word in batch)

    @property
    def scheduled(self) -> int:
        """Number of puzzles before the list starts repeating."""
        return sum(map(len, self.batches))

    def today(self, now: datetime | None = None) -> date:
        return (now or datetime.now(self.timezone)).astimezone(self.timezone).date()

    def number_on(self, day: date) -> int:
        return (day - self.epoch).days + 1

    def date_of(self, number: int) -> date:
        return self.epoch + timedelta(days=number - 1)

    def first_number_of(self, batch: int) -> int:
        """Puzzle number on which the batch (0-based) starts."""
        return 1 + sum(len(b) for b in self.batches[:batch])


class PuzzleCalendar:
    def __init__(self, words: WordList, key: bytes) -> None:
        self._words = words
        self._key = key
        self._batch_orders = tuple(self._shuffle(batch, b"") for batch in words.batches)
        self._cycle_order = cache(lambda cycle: self._shuffle(words.words, f"{cycle}".encode()))

    @property
    def words(self) -> tuple[str, ...]:
        return self._words.words

    def current_number(self, now: datetime | None = None) -> int:
        # Before the epoch, puzzle no. 1 is shown, which is handy for testing a launch.
        return max(1, self._words.number_on(self._words.today(now)))

    def get(self, number: int) -> Puzzle:
        if number < 1:
            raise ValueError(f"Puzzle numbers start at 1, got {number}.")
        return Puzzle(number, self._words.date_of(number), self._secret(number - 1))

    def _secret(self, index: int) -> str:
        for order in self._batch_orders:
            if index < len(order):
                return order[index]
            index -= len(order)
        cycle, index = divmod(index, self._words.scheduled)
        return self._cycle_order(cycle + 1)[index]

    def _shuffle(self, words: Sequence[str], salt: bytes) -> tuple[str, ...]:
        """A keyed order that is stable across processes and Python versions."""
        return tuple(
            sorted(
                words,
                key=lambda w: hmac.digest(self._key, salt + b":" + w.encode(), hashlib.sha256),
            )
        )
