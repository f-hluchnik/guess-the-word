import unicodedata

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import write_words


def today(client: TestClient) -> dict:
    response = client.get("/api/puzzles/today")
    assert response.status_code == 200
    return response.json()


def guess(client: TestClient, word: str, number: int | None = None):
    if number is None:
        number = today(client)["number"]
    return client.post(f"/api/puzzles/{number}/guesses", json={"word": word})


def test_todays_puzzle(client: TestClient, settings: Settings) -> None:
    puzzle = today(client)
    assert puzzle["number"] == 11  # epoch is ten days ago
    assert puzzle["vocabulary_size"] == 8
    assert "secret" not in puzzle


def test_neighbours_skip_inflections_of_the_secret(client: TestClient) -> None:
    neighbours = today(client)["neighbours"]
    # 8 words minus the secret and its two inflections leaves 5: rank 10 and
    # 1000 can't be filled, rank 1 is the closest genuinely different word.
    assert [n["rank"] for n in neighbours] == [1]
    assert neighbours[0]["similarity"] == guess(client, "šrám").json()["similarity"]
    assert guess(client, "jizvou").json()["solved"]


def test_exact_guess_solves(client: TestClient) -> None:
    body = guess(client, "jizva").json()
    assert body == {
        "word": "jizva",
        "lemma": None,
        "known": True,
        "similarity": 1.0,
        "solved": True,
    }


def test_inflected_guess_solves(client: TestClient) -> None:
    body = guess(client, "jizvy").json()
    assert body["solved"] and body["lemma"] == "jizva"


def test_related_guess_reports_similarity(client: TestClient) -> None:
    body = guess(client, "šrám").json()
    assert body["known"] and not body["solved"]
    assert 0 < body["similarity"] < 1


def test_unknown_word_is_not_an_error(client: TestClient) -> None:
    response = guess(client, "žluťoučký")
    assert response.status_code == 200
    assert response.json() == {
        "word": "žluťoučký",
        "lemma": None,
        "known": False,
        "similarity": None,
        "solved": False,
    }


def test_input_is_normalised(client: TestClient) -> None:
    decomposed_upper = unicodedata.normalize("NFD", "  ŠRÁM ")
    assert guess(client, decomposed_upper).json()["word"] == "šrám"


@pytest.mark.parametrize("word", ["", "   ", "dvě slova", "abc1", "x" * 41, "<script>"])
def test_invalid_words_are_rejected(client: TestClient, word: str) -> None:
    assert guess(client, word).status_code == 422


def test_past_puzzles_are_playable(client: TestClient) -> None:
    assert guess(client, "hrad", number=1).status_code == 200


@pytest.mark.parametrize("offset", [1, 365])
def test_future_puzzles_are_hidden(client: TestClient, offset: int) -> None:
    number = today(client)["number"] + offset
    assert guess(client, "hrad", number=number).status_code == 404


def test_puzzle_zero_does_not_exist(client: TestClient) -> None:
    assert guess(client, "hrad", number=0).status_code == 404


def test_guesses_are_rate_limited(settings: Settings) -> None:
    limited = settings.model_copy(update={"guess_rate_limit": "2/minute"})
    with TestClient(create_app(limited)) as client:
        assert guess(client, "hrad").status_code == 200
        assert guess(client, "kniha").status_code == 200
        response = guess(client, "auto")
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1


def test_frontend_is_served_with_security_headers(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Kosinus</title>" in response.text
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-cache"


def test_api_responses_are_not_cached(client: TestClient) -> None:
    assert client.get("/api/puzzles/today").headers["Cache-Control"] == "no-store"


def test_unknown_pages_get_the_404_page(client: TestClient) -> None:
    response = client.get("/neexistuje")
    assert response.status_code == 404
    assert "Tahle stránka neexistuje" in response.text


def test_method_page_credits_fasttext(client: TestClient) -> None:
    assert "CC BY-SA 3.0" in client.get("/metoda.html").text


def test_docs_are_off_by_default(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_rate_limit_uses_trusted_proxy_header(settings: Settings) -> None:
    limited = settings.model_copy(
        update={"guess_rate_limit": "1/minute", "client_ip_header": "Fly-Client-IP"}
    )
    with TestClient(create_app(limited)) as client:
        number = today(client)["number"]
        url = f"/api/puzzles/{number}/guesses"
        first = client.post(url, json={"word": "hrad"}, headers={"Fly-Client-IP": "1.1.1.1"})
        other = client.post(url, json={"word": "hrad"}, headers={"Fly-Client-IP": "2.2.2.2"})
        again = client.post(url, json={"word": "hrad"}, headers={"Fly-Client-IP": "1.1.1.1"})
    assert (first.status_code, other.status_code, again.status_code) == (200, 200, 429)


def test_startup_fails_when_a_secret_word_has_no_vector(settings: Settings) -> None:
    write_words(settings.words_path, ["jizva", "neexistuje"])
    with pytest.raises(ValueError, match="neexistuje"), TestClient(create_app(settings)):
        pass


@pytest.mark.parametrize(
    ("word", "secret", "expected"),
    [
        ("jizvy", "jizva", True),  # simplemma knows this form
        ("kominíka", "kominík", True),  # it doesn't know this one
        ("kominíků", "kominík", True),
        ("jizvou", "jizva", True),
        ("jizvení", "jizva", False),  # derived word, not an inflection
        ("zjizvená", "jizva", False),
        ("baskytara", "kytara", False),
    ],
)
def test_inflections_are_recognised(word: str, secret: str, expected: bool) -> None:
    from app.vocabulary import SemanticModel

    assert SemanticModel.is_form_of(SemanticModel.__new__(SemanticModel), word, secret) is expected
