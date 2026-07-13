"""
Semantic word-guessing game — FastAPI scaffolding.

Design principles:
- The secret word NEVER leaves the server. The client only ever holds an
  opaque session token.
- Similarity is reported exactly (raw cosine + percentage), not clipped or
  bucketed, so players get precise feedback.
- A one-time "ceiling" score (the closest any other word in the vocabulary
  gets to the secret) is computed per session, giving players a calibration
  point without revealing which word that is.
- All user input is validated (length, charset) before touching the model.
- Guess submissions are rate-limited per session AND per IP to blunt
  brute-force / scripted guessing.
- Sessions expire after inactivity so memory doesn't grow unbounded.

Run with:
    uvicorn main:app --reload
"""

import datetime
import os
import re
import secrets
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Optional

import simplemma
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from gensim.models import KeyedVectors
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VECTORS_PATH = os.environ.get("VECTORS_PATH", "cc.cs.300.kv")  # native gensim format
SESSION_TTL_SECONDS = 60 * 30       # sessions expire after 30 min idle
MAX_GUESSES_PER_MINUTE = 20         # per-session rate limit (belt...)
IP_GUESS_RATE = "30/minute"         # per-IP rate limit via slowapi (...and suspenders)
IP_NEW_GAME_RATE = "10/minute"      # stop one IP from spawning endless sessions
MAX_WORD_LENGTH = 40
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# For big vocabularies (the full cc.cs.300 file has ~2M words), a full
# most_similar() scan can take a noticeable moment. restrict_vocab caps the
# search to the N most frequent words (vectors are frequency-sorted), which
# keeps the one-time "ceiling" lookup below fast without materially changing
# the result for common nouns. Set to None to search the full vocabulary.
RESTRICT_VOCAB_FOR_CEILING = 200_000

# Placeholder candidate word pool — this is deliberately NOT theme-specific.
# Replace with your curated noun list later (any part of speech you like,
# though nouns tend to make the cleanest secret words). Keeping this list
# separate from "any word gensim knows" is what stops the game from ever
# picking a typo, slur, or foreign-language token as the secret word.
# CANDIDATE_WORDS = ["hrad", "auto", "kniha", "slunce", "počítač", "moře", "chléb", "kytara"]
CANDIDATE_WORDS = ["jizva"]

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------

print("Loading Czech semantic database...")
word_vectors: KeyedVectors = KeyedVectors.load(VECTORS_PATH, mmap="r")
print(f"Loaded {len(word_vectors)} vectors.")


def normalize_input(raw: str) -> str:
    """Lowercase, strip, and Unicode-normalize so diacritics compare consistently."""
    cleaned = raw.strip().lower()
    cleaned = unicodedata.normalize("NFC", cleaned)
    return cleaned


def lemmatize(word: str) -> str:
    # 1. Check if the raw word is already a valid vector in your model
    if word in word_vectors:
        return word
        
    # 2. Only fallback to lemmatization if the exact word isn't found
    try:
        return simplemma.lemmatize(word, lang="cs")
    except Exception:
        # If simplemma chokes on something unexpected, fall back to the raw form
        # rather than 500-ing the whole request.
        return word


def get_similarity(guess_word: str, target_word: str) -> Optional[float]:
    """Raw cosine similarity in [-1, 1], or None if either word is out of vocabulary."""
    if guess_word not in word_vectors or target_word not in word_vectors:
        return None
    return float(word_vectors.similarity(guess_word, target_word))


def compute_ceiling_percentage(target_word: str) -> Optional[float]:
    """
    The similarity score of the single closest word in the vocabulary to the
    target — NOT the word itself. This gives players a calibration point:
    e.g. if the ceiling is 71%, they know 71% is essentially "as close as
    language gets" for this particular secret word, so a 60% guess is
    actually excellent even though it looks unremarkable on a naive 0-100
    scale. Computed once per session, not per guess.
    """
    try:
        nearest = word_vectors.most_similar(target_word, topn=1, restrict_vocab=RESTRICT_VOCAB_FOR_CEILING)
    except KeyError:
        return None
    if not nearest:
        return None
    return round(nearest[0][1] * 100, 2)


# ---------------------------------------------------------------------------
# Session store (swap for Redis in a multi-worker deployment)
# ---------------------------------------------------------------------------

@dataclass
class GameSession:
    secret_word: str
    ceiling_percentage: Optional[float]
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    guess_timestamps: list = field(default_factory=list)
    guess_count: int = 0
    solved: bool = False


SESSIONS: Dict[str, GameSession] = {}


def create_session() -> str:
    token = secrets.token_urlsafe(24)
    # random pick
    # secret = secrets.choice(CANDIDATE_WORDS)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    day_of_year = today.toordinal()
    index = day_of_year % len(CANDIDATE_WORDS)
    secret = CANDIDATE_WORDS[index]
    ceiling = compute_ceiling_percentage(secret)
    SESSIONS[token] = GameSession(secret_word=secret, ceiling_percentage=ceiling)
    return token


def get_session(token: str) -> GameSession:
    session = SESSIONS.get(token)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    if time.time() - session.last_seen > SESSION_TTL_SECONDS:
        del SESSIONS[token]
        raise HTTPException(status_code=410, detail="Session expired.")
    session.last_seen = time.time()
    return session


def check_rate_limit(session: GameSession) -> None:
    now = time.time()
    session.guess_timestamps = [t for t in session.guess_timestamps if now - t < 60]
    if len(session.guess_timestamps) >= MAX_GUESSES_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many guesses — slow down a bit.")
    session.guess_timestamps.append(now)


def cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [tok for tok, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SECONDS]
    for tok in expired:
        del SESSIONS[tok]


# ---------------------------------------------------------------------------
# Request/response models with validation
# ---------------------------------------------------------------------------

# Czech letters + basic Latin; adjust if you want to support other alphabets.
WORD_PATTERN = re.compile(r"^[a-zá-žA-Zá-žéěíóúůýřčšžťďň]+$", re.IGNORECASE)


class GuessRequest(BaseModel):
    session_token: str
    guess: str

    @field_validator("guess")
    @classmethod
    def validate_guess(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Guess cannot be empty.")
        if len(v) > MAX_WORD_LENGTH:
            raise ValueError("Guess is too long.")
        if not WORD_PATTERN.match(v):
            raise ValueError("Guess must contain only letters.")
        return v


class GuessResponse(BaseModel):
    guess: str
    lemma: str
    found: bool                       # False if the lemma isn't in the vector vocabulary
    similarity: Optional[float]       # raw cosine similarity, -1.0 to 1.0
    percentage: Optional[float]       # similarity * 100, for display
    solved: bool


class NewGameResponse(BaseModel):
    session_token: str
    ceiling_percentage: Optional[float]  # best-possible similarity score for this round


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# IP-based limiter. This is the OUTER guard: it stops a single IP from
# hammering the API across many sessions/tokens at once. The per-session
# check_rate_limit() above is the INNER guard: it stops one session token
# from being hammered even if guesses somehow came from rotating IPs
# (proxies, botnets). Together they cover both axes; either alone leaves a
# gap.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Czech Semantic Word Game")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)

@app.get("/")
async def read_index():
    return FileResponse("index.html")


@app.post("/new-game", response_model=NewGameResponse)
@limiter.limit(IP_NEW_GAME_RATE)
def new_game(request: Request):
    cleanup_expired_sessions()
    token = create_session()
    session = SESSIONS[token]
    return NewGameResponse(session_token=token, ceiling_percentage=session.ceiling_percentage)


@app.post("/guess", response_model=GuessResponse)
@limiter.limit(IP_GUESS_RATE)
def guess(payload: GuessRequest, request: Request):
    session = get_session(payload.session_token)
    check_rate_limit(session)

    normalized = normalize_input(payload.guess)
    lemma = lemmatize(normalized)
    session.guess_count += 1

    similarity = get_similarity(lemma, session.secret_word)
    if similarity is None:
        return GuessResponse(
            guess=payload.guess, lemma=lemma, found=False,
            similarity=None, percentage=None, solved=False,
        )

    solved = lemma == session.secret_word
    if solved:
        session.solved = True

    return GuessResponse(
        guess=payload.guess, lemma=lemma, found=True,
        similarity=round(similarity, 4), percentage=round(similarity * 100, 2),
        solved=solved,
    )


@app.get("/health")
def health():
    return {"status": "ok", "vectors_loaded": len(word_vectors), "active_sessions": len(SESSIONS)}