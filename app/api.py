from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import AfterValidator, BaseModel

from app.game import Game, PuzzleNotAvailable
from app.puzzle import Puzzle
from app.text import is_word, normalize


def _valid_word(raw: str) -> str:
    word = normalize(raw)
    if not is_word(word):
        raise ValueError("Expected a single word made of Czech letters only.")
    return word


Word = Annotated[str, AfterValidator(_valid_word)]


class Neighbour(BaseModel):
    rank: int
    similarity: float


class PuzzleOut(BaseModel):
    number: int
    date: date
    vocabulary_size: int
    # How similar the n-th closest words are to the secret, as a scale.
    neighbours: list[Neighbour]


class GuessIn(BaseModel):
    word: Word


class GuessOut(BaseModel):
    word: str
    lemma: str | None
    known: bool
    similarity: float | None
    solved: bool


def get_game(request: Request) -> Game:
    return request.app.state.game


def guess_rate_limit(request: Request) -> None:
    request.app.state.guess_rate_limit(request)


GameDep = Annotated[Game, Depends(get_game)]

router = APIRouter(prefix="/api")


def _puzzle_out(game: Game, puzzle: Puzzle) -> PuzzleOut:
    return PuzzleOut(
        number=puzzle.number,
        date=puzzle.date,
        vocabulary_size=game.vocabulary_size,
        neighbours=[
            Neighbour(rank=rank, similarity=round(similarity, 4))
            for rank, similarity in game.reference(puzzle).items()
        ],
    )


@router.get("/puzzles/today")
def todays_puzzle(game: GameDep) -> PuzzleOut:
    return _puzzle_out(game, game.today())


@router.post("/puzzles/{number}/guesses", dependencies=[Depends(guess_rate_limit)])
def submit_guess(number: int, body: GuessIn, game: GameDep) -> GuessOut:
    try:
        puzzle = game.puzzle(number)
    except PuzzleNotAvailable:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such puzzle.") from None

    result = game.guess(puzzle, body.word)
    return GuessOut(
        word=result.word,
        lemma=result.lemma,
        known=result.known,
        similarity=None if result.similarity is None else round(result.similarity, 4),
        solved=result.solved,
    )
