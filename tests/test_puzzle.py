from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.puzzle import PuzzleCalendar, WordList, WordListError

PRAGUE = ZoneInfo("Europe/Prague")
EPOCH = date(2026, 1, 1)
FIRST = ("hrad", "auto", "kniha", "slunce")
SECOND = ("moře", "chléb", "kytara")
KEY = b"k" * 16


def word_list(*batches: tuple[str, ...]) -> WordList:
    return WordList(EPOCH, PRAGUE, batches or (FIRST, SECOND))


def secrets(calendar: PuzzleCalendar, numbers: range) -> list[str]:
    return [calendar.get(n).secret for n in numbers]


def test_batches_are_played_in_order_each_word_once() -> None:
    played = secrets(PuzzleCalendar(word_list(), KEY), range(1, 8))
    assert sorted(played[:4]) == sorted(FIRST)
    assert sorted(played[4:]) == sorted(SECOND)


def test_appending_a_batch_keeps_earlier_puzzles() -> None:
    before = secrets(PuzzleCalendar(word_list(FIRST), KEY), range(1, 5))
    after = secrets(PuzzleCalendar(word_list(FIRST, SECOND), KEY), range(1, 5))
    assert before == after


def test_order_within_a_batch_is_keyed_not_file_order() -> None:
    reordered = PuzzleCalendar(word_list(tuple(reversed(FIRST))), KEY)
    original = PuzzleCalendar(word_list(FIRST), KEY)
    other_key = PuzzleCalendar(word_list(FIRST), b"another-key-0000")
    assert secrets(reordered, range(1, 5)) == secrets(original, range(1, 5))
    assert secrets(other_key, range(1, 5)) != secrets(original, range(1, 5))


def test_exhausted_list_is_replayed_reshuffled() -> None:
    calendar = PuzzleCalendar(word_list(), KEY)
    replay = secrets(calendar, range(8, 15))
    assert sorted(replay) == sorted(FIRST + SECOND)
    assert secrets(calendar, range(15, 22)) != replay


def test_puzzle_dates_count_from_epoch() -> None:
    calendar = PuzzleCalendar(word_list(), KEY)
    assert calendar.get(1).date == EPOCH
    assert calendar.get(32).date == date(2026, 2, 1)
    with pytest.raises(ValueError):
        calendar.get(0)


def test_day_boundary_follows_the_list_timezone() -> None:
    # 23:30 UTC on 15 Nov is already 00:30 on 16 Nov in Prague.
    late_utc = datetime(2026, 11, 15, 23, 30, tzinfo=UTC)
    calendar = PuzzleCalendar(word_list(), KEY)
    assert calendar.current_number(late_utc) == (date(2026, 11, 16) - EPOCH).days + 1


def test_before_the_epoch_puzzle_one_is_current() -> None:
    calendar = PuzzleCalendar(word_list(), KEY)
    assert calendar.current_number(datetime(2025, 6, 1, tzinfo=UTC)) == 1


def test_load(tmp_path: Path) -> None:
    path = tmp_path / "words.toml"
    path.write_text(
        'epoch = 2026-01-01\ntimezone = "Europe/Prague"\n'
        '[[batch]]\nwords = ["hrad", "kniha"]\n[[batch]]\nwords = ["moře"]\n',
        encoding="utf-8",
    )
    words = WordList.load(path)
    assert words.batches == (("hrad", "kniha"), ("moře",))
    assert words.first_number_of(1) == 3
    assert words.scheduled == 3


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('timezone = "Europe/Prague"\n[[batch]]\nwords = ["hrad"]', "epoch"),
        ('epoch = 2026-01-01\ntimezone = "Mars/Base"\n[[batch]]\nwords = ["hrad"]', "timezone"),
        ('epoch = 2026-01-01\ntimezone = "Europe/Prague"', "at least one"),
        ('epoch = 2026-01-01\ntimezone = "UTC"\n[[batch]]\nwords = ["Hrad"]', "lowercase"),
        ('epoch = 2026-01-01\ntimezone = "UTC"\n[[batch]]\nwords = ["dvě slova"]', "one"),
        (
            'epoch = 2026-01-01\ntimezone = "UTC"\n[[batch]]\nwords = ["hrad"]\n'
            '[[batch]]\nwords = ["hrad"]',
            "batch 1 and batch 2",
        ),
        ("epoch = ", "words.toml"),
    ],
)
def test_load_rejects_invalid_lists(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "words.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(WordListError, match=message):
        WordList.load(path)
