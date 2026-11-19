from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vectors_path: Path = ROOT_DIR / "vectors" / "cc.cs.300.kv"
    words_path: Path = ROOT_DIR / "data" / "words.toml"
    static_dir: Path = ROOT_DIR / "static"

    # Keys the order in which secret words are drawn. Anyone who knows it can
    # compute every future answer, so it must stay out of the repository.
    puzzle_secret: SecretStr = Field(min_length=16)

    guess_rate_limit: str = "30/minute"
    # Header carrying the real client IP, set by a trusted proxy in front of
    # the app (e.g. "Fly-Client-IP" on Fly.io). Unset: the TCP peer address.
    client_ip_header: str | None = None
    # "memory://" is per process. Use "redis://host:6379" with several workers
    # or replicas so they share one budget.
    rate_limit_storage_uri: str = "memory://"

    # Load the vectors after the server starts listening, so the page and
    # /healthz answer at once on a cold start while /api returns 503 until
    # ready. Tests turn it off to load synchronously and see errors directly.
    load_in_background: bool = True

    docs_enabled: bool = False
    log_level: str = "INFO"
