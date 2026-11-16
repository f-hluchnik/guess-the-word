from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from gensim.models import KeyedVectors

from app.config import Settings
from app.main import create_app

DIM = 32


def _toy_vectors() -> KeyedVectors:
    """A tiny vocabulary with known geometry around the secret word "jizva"."""
    rng = np.random.default_rng(7)
    noise = lambda: rng.normal(size=DIM)  # noqa: E731
    jizva = noise()
    words = {
        "jizva": jizva,
        # Inflections sit almost on top of the secret.
        "jizvy": jizva + 0.05 * noise(),
        "jizvou": jizva + 0.05 * noise(),
        # A genuinely different but related word.
        "šrám": jizva + 1.0 * noise(),
        "rána": jizva + 2.0 * noise(),
        "hrad": noise(),
        "kniha": noise(),
        "auto": noise(),
    }
    vectors = KeyedVectors(DIM, dtype=np.float32)
    vectors.add_vectors(list(words), np.array(list(words.values()), dtype=np.float32))
    return vectors


@pytest.fixture(scope="session")
def vectors_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("vectors") / "toy.kv"
    _toy_vectors().save(str(path))
    return path


def write_words(path: Path, *batches: list[str], epoch: date | None = None) -> Path:
    epoch = epoch or date.today() - timedelta(days=10)
    lines = [f"epoch = {epoch.isoformat()}", 'timezone = "Europe/Prague"']
    for batch in batches:
        lines += ["[[batch]]", f"words = {batch!r}".replace("'", '"')]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def settings(vectors_path: Path, tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        puzzle_secret="test-secret-test-secret",
        vectors_path=vectors_path,
        words_path=write_words(tmp_path / "words.toml", ["jizva"]),
        guess_rate_limit="1000/minute",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as client:
        yield client
