import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import Settings
from app.game import Game
from app.puzzle import PuzzleCalendar, WordList
from app.ratelimit import RateLimit

log = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Content-Type-Options": "nosniff",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    # Browsers ignore this over plain HTTP, so it's safe to send everywhere.
    "Strict-Transport-Security": "max-age=31536000",
}


def load_game(settings: Settings) -> Game:
    from app.vocabulary import SemanticModel  # imports gensim, which takes seconds

    model = SemanticModel.load(settings.vectors_path)
    calendar = PuzzleCalendar(
        WordList.load(settings.words_path),
        key=settings.puzzle_secret.get_secret_value().encode(),
    )
    game = Game(model, calendar)
    game.warm_up()
    return game


def _load_in_background(app: FastAPI, settings: Settings) -> None:
    try:
        app.state.game = load_game(settings)
        log.info("Ready")
    except Exception:
        # A broken word list or vector file: crash so it shows up in fly logs
        # and the machine restarts, instead of answering 503 forever.
        log.exception("Loading the game failed")
        os._exit(1)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.game = None
        if settings.load_in_background:
            threading.Thread(
                target=_load_in_background, args=(app, settings), name="load-game", daemon=True
            ).start()
        else:
            app.state.game = load_game(settings)
        yield

    app = FastAPI(
        title="Kosinus",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.guess_rate_limit = RateLimit(
        "guess",
        settings.guess_rate_limit,
        settings.rate_limit_storage_uri,
        client_ip_header=settings.client_ip_header,
    )

    @app.middleware("http")
    async def response_headers(request: Request, call_next) -> Response:
        response = await call_next(request)
        path = request.url.path
        if not path.startswith("/docs"):  # Swagger UI loads from a CDN
            response.headers.update(SECURITY_HEADERS)
        # Answers must never be cached. Static files may be, but are revalidated
        # on every load (cheap 304s via ETag), so a deploy is picked up at once.
        response.headers.setdefault(
            "Cache-Control", "no-store" if path.startswith("/api/") else "no-cache"
        )
        return response

    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.get("/healthz", include_in_schema=False)
    def healthz(request: Request) -> dict[str, str | bool]:
        # Liveness only: the process is up. "ready" says whether /api works yet.
        return {"status": "ok", "ready": request.app.state.game is not None}

    app.include_router(router)
    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")
    return app
