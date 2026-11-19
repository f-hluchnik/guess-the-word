from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from app.puzzle import WordList
from scripts.words import frozen_violations, started_batches

PRAGUE = ZoneInfo("Europe/Prague")
EPOCH = date(2026, 1, 1)
ONE = ("hrad", "auto", "kniha")
TWO = ("moře", "chléb")
THREE = ("kytara", "slunce")


def at(day: int) -> datetime:
    """Noon in Prague on puzzle `day`."""
    return datetime(2026, 1, day, 11, 0, tzinfo=UTC)


def words(*batches: tuple[str, ...]) -> WordList:
    return WordList(EPOCH, PRAGUE, batches)


def test_started_batches_include_tomorrow() -> None:
    assert started_batches(words(ONE, TWO), at(2)) == 1
    assert started_batches(words(ONE, TWO), at(3)) == 2  # batch 2 starts tomorrow


def test_appending_and_editing_future_batches_is_allowed() -> None:
    old = words(ONE, TWO)
    assert frozen_violations(old, words(ONE, TWO, THREE), at(2)) == []
    assert frozen_violations(old, words(ONE, ("moře", "sýr")), at(2)) == []
    assert frozen_violations(old, words(tuple(reversed(ONE)), TWO), at(2)) == []


def test_editing_a_started_batch_is_refused() -> None:
    old = words(ONE, TWO)
    assert frozen_violations(old, words(("hrad", "auto", "sýr"), TWO), at(2))
    assert frozen_violations(old, words(ONE), at(4))


def test_epoch_is_frozen_after_launch_only() -> None:
    moved = WordList(date(2026, 1, 5), PRAGUE, (ONE,))
    assert frozen_violations(words(ONE), moved, at(2))
    before_launch = datetime(2025, 12, 1, tzinfo=UTC)
    assert frozen_violations(words(ONE), moved, before_launch) == []


def test_status_before_launch(tmp_path, capsys, monkeypatch) -> None:
    import argparse

    from scripts.words import status
    from tests.conftest import write_words

    path = write_words(tmp_path / "words.toml", ["hrad", "auto"], epoch=date(2999, 1, 1))
    assert status(argparse.Namespace(path=path, min_runway=None)) == 0
    out = capsys.readouterr().out
    assert "Not launched yet: puzzle 1 is on 2999-01-01" in out
    assert "Today is puzzle 1." in out
