"""Maintain the secret word list (data/words.toml).

python -m scripts.words check                      # format and duplicates
python -m scripts.words check --vectors PATH       # + vocabulary and quality report
python -m scripts.words check --against OLD.toml   # started batches unchanged
python -m scripts.words status [--min-runway 30]   # how many days are left
PUZZLE_SECRET=... python -m scripts.words peek     # upcoming answers (spoilers!)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from app.puzzle import PuzzleCalendar, WordList, WordListError

DEFAULT_PATH = Path("data/words.toml")

# Quality thresholds for the report. Warnings only; they never fail a build.
RARE_RANK = 50_000  # beyond this frequency rank, most players won't think of the word
LOW_CEILING = 0.35  # nearest different word below this: the puzzle will feel hopeless


def started_batches(words: WordList, now: datetime | None = None) -> int:
    """How many batches have begun, counting tomorrow's puzzle as already begun."""
    upcoming = words.number_on(words.today(now)) + 1
    return sum(1 for i in range(len(words.batches)) if words.first_number_of(i) <= upcoming)


def frozen_violations(old: WordList, new: WordList, now: datetime | None = None) -> list[str]:
    """Changes in `new` that would rewrite puzzles already played under `old`."""
    if old.number_on(old.today(now)) < 1:
        return []  # not launched yet: anything goes
    problems = []
    if (old.epoch, old.timezone) != (new.epoch, new.timezone):
        problems.append("epoch and timezone can't change after launch.")
    for i in range(started_batches(old, now)):
        if i >= len(new.batches) or set(old.batches[i]) != set(new.batches[i]):
            problems.append(f"batch {i + 1} has started and must not change.")
    return problems


def check(args: argparse.Namespace) -> int:
    words = WordList.load(args.path)
    errors: list[str] = []

    if args.against:
        try:
            old = WordList.load(args.against)
        except WordListError as error:
            print(f"Skipping the frozen-batch check, old list is invalid: {error}")
        else:
            errors += frozen_violations(old, words)
            if old.number_on(old.today()) > old.scheduled and words.batches != old.batches:
                print("warning: the old list had run out; recent recycled days will change.")

    if args.vectors:
        errors += quality_report(words, args.vectors, quiet=args.quiet)

    for error in errors:
        print(f"error: {error}")
    print(f"{len(words.words)} words in {len(words.batches)} batches: ", end="")
    print("problems found." if errors else "OK.")
    return 1 if errors else 0


def quality_report(words: WordList, vectors: Path, quiet: bool) -> list[str]:
    from app.vocabulary import SemanticModel  # gensim is slow to import; only load it here

    model = SemanticModel.load(vectors)
    errors = []
    print(f"{'word':<16}{'rank':>9}{'nearest':>9}  notes")
    for word in words.words:
        if word not in model:
            errors.append(f"{word!r} is not in the vectors.")
            continue
        rank, ceiling, lemma = model.rank(word), model.ceiling(word), model.lemma(word)
        notes = []
        if rank > RARE_RANK:
            notes.append("rare word")
        if ceiling is None or ceiling < LOW_CEILING:
            notes.append("few close words, hard puzzle")
        if lemma != word:
            notes.append(f"lemmatiser reads it as {lemma!r}")
        if notes or not quiet:
            shown = "-" if ceiling is None else f"{ceiling:.0%}"
            print(f"{word:<16}{rank:>9}{shown:>9}  {', '.join(notes)}")
    return errors


def status(args: argparse.Namespace) -> int:
    words = WordList.load(args.path)
    today = words.number_on(words.today())
    runway = words.scheduled - today
    print(f"Today is puzzle {today}. The list covers puzzles 1-{words.scheduled}", end="")
    print(f" (until {words.date_of(words.scheduled)}).")
    if runway < 0:
        print("The list has run out and is being replayed. Append a batch.")
    else:
        print(f"{runway} days left after today.")
    if args.min_runway is not None and runway < args.min_runway:
        print(f"Fewer than {args.min_runway} days left. Append a new batch.")
        return 1
    return 0


def peek(args: argparse.Namespace) -> int:
    secret = os.environ.get("PUZZLE_SECRET")
    if not secret:
        print("Set PUZZLE_SECRET to the production value to see the real schedule.")
        return 2
    words = WordList.load(args.path)
    calendar = PuzzleCalendar(words, key=secret.encode())
    start = max(1, words.number_on(words.today()) + args.offset)
    for number in range(start, start + args.days):
        puzzle = calendar.get(number)
        print(f"{puzzle.date}  #{puzzle.number:<5} {puzzle.secret}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Maintain the secret word list.")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    commands = parser.add_subparsers(required=True)

    p = commands.add_parser("check", help="validate the list")
    p.add_argument("--vectors", type=Path, help="also check every word against the vectors")
    p.add_argument("--against", type=Path, help="previous version, to protect started batches")
    p.add_argument("--quiet", action="store_true", help="only list words with notes")
    p.set_defaults(run=check)

    p = commands.add_parser("status", help="show how long the list lasts")
    p.add_argument("--min-runway", type=int, help="exit 1 if fewer days are left")
    p.set_defaults(run=status)

    p = commands.add_parser("peek", help="print upcoming answers (needs PUZZLE_SECRET)")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--offset", type=int, default=0, help="start N days from today")
    p.set_defaults(run=peek)

    args = parser.parse_args()
    try:
        sys.exit(args.run(args))
    except WordListError as error:
        print(f"error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
