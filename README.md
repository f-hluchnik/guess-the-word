# Kosinus

A daily Czech word-guessing game. Every day there is one hidden noun. Each
guess is scored by the cosine similarity of its fastText vector to the hidden
word's vector, shown as a percentage. For scale, the game also shows how
similar the 1st, 10th and 1000th closest words are. The method is explained to
players on [`/metoda.html`](static/metoda.html).

## How it works

```
browser ──GET /api/puzzles/today──────────────▶ FastAPI ──▶ PuzzleCalendar (which word today)
        ──POST /api/puzzles/{n}/guesses {word}─▶         ──▶ SemanticModel (fastText vectors, simplemma)
        ◀── static/ (HTML, CSS, JS, no build step)
```

- **Deterministic.** A score is cos θ = a·b / (‖a‖‖b‖) of two fixed vectors.
  The same word always gets the same number.
- **Stateless server.** The secret for puzzle *n* is derived from
  `data/words.toml` and `PUZZLE_SECRET`. Nothing is stored server-side, so
  any number of machines can serve the game. The secret never leaves the server.
- **Inflections handled.** `jizvy` solves a puzzle whose answer is `jizva`,
  and inflections of the secret are kept out of the reference values.
- **Private by default.** No cookies, accounts or analytics. Progress lives in
  the player's `localStorage`; the server keeps IPs only in the rate limiter.

| Path | Purpose |
| --- | --- |
| `app/main.py` | App factory, startup, headers, static files |
| `app/api.py` | HTTP routes and request/response schemas |
| `app/game.py` | Game rules, independent of HTTP |
| `app/puzzle.py` | Word list format and the day-to-word schedule |
| `app/vocabulary.py` | Vectors, lemmatisation, reference values |
| `app/text.py` | Input normalisation and validation |
| `app/ratelimit.py` | Per-IP rate limit (in-memory or Redis) |
| `data/words.toml` | The secret words |
| `scripts/words.py` | Word list checks, runway, spoiler-free maintenance |
| `scripts/build_vectors.py` | Builds the vocabulary from fastText |
| `static/` | Frontend |

## Vocabulary

fastText's Czech file has 2,000,000 tokens; 710,085 of them are lowercase
words in the Czech alphabet. The game keeps the **300,000 most frequent** of
those plus every rarer one that is in **simplemma's Czech dictionary**:
351,654 words, 422 MB. Measured on the full file:

| Frequency band | Share of tokens that are dictionary words |
| --- | --- |
| 0–50k | 81 % |
| 200k–300k | 24 % |
| 300k–400k | 18 % |
| 500k–710k | 10 % |

- Past 300k the list is mostly typos, glued words (`tématůmkominík`) and Slovak
  forms (`počítača`). With 500k or more words they start appearing among the
  nearest neighbours of the secret and inflate the reference values.
- A sample of 94 nouns ranging from `stůl` to `škvor` and `chroust`: all
  are within the top 300k (the rarest, `škvor`, ranks 268,104).
- Of word forms the dictionary knows, 93 % are within the top 300k directly or
  through their base form; the dictionary pass adds the remaining real words.

Change `FREQUENT_WORDS` in `fly.toml` to tune it.

## Running locally

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env                # then set PUZZLE_SECRET
python -m scripts.build_vectors     # one-off: downloads ~1.3 GB, writes vectors/

uvicorn app.main:create_app --factory --reload
# open http://localhost:8000
```

Tests use a tiny synthetic vocabulary and don't need the real vectors:

```sh
pytest
ruff check . && ruff format --check .
```

`docker compose up --build` runs the production image locally.

## Maintaining the word list

`data/words.toml` holds the launch date (`epoch`), the time zone in which
days change, and the words in **batches**. Batches are played in order; the
order *within* a batch is shuffled with `PUZZLE_SECRET`, so the public file
doesn't reveal which word falls on which day.

```toml
[[batch]]
words = [
  "jizva",
  "hrad",
]
```

### Routine (about once a month)

1. Pick 30–60 new nouns by hand and append them as a new `[[batch]]` at the
   end of the file.
2. Check them against the real vectors:

   ```sh
   python -m scripts.words check --vectors vectors/cc.cs.300.kv
   ```

   Replace words noted as *rare word* (players won't think of it), *few close
   words* (the puzzle will feel hopeless) or *lemmatiser reads it as …*
   (ambiguous forms, e.g. `rána` → `ráno`).
3. Open a pull request. CI checks the format, duplicates, and that no batch
   that has already started was changed. To rewrite started batches on
   purpose (only before launch), add the `schedule-reset` label to the pull
   request.
4. Merge. The deploy re-checks every word against the vectors while building
   the image and fails before anything goes live if one is missing.

`python -m scripts.words status` shows how many days are left. A scheduled
workflow runs it every Monday and fails (GitHub emails you) when fewer than
14 days remain: that email is the cue to add the next batch. If the list runs out, it is replayed in a new shuffled
order, so the game never breaks. Add a batch before that happens.

### Rules

- **Only append.** Never edit or reorder a batch that has started: that
  rewrites past puzzles. Unstarted batches can be edited freely.
- **Don't change `epoch`, `timezone` or `PUZZLE_SECRET` after launch.** Each
  one reshuffles every puzzle, past ones included.
- **Choosing words:** common, concrete nouns in the singular nominative. Avoid
  proper nouns, brands, slurs and topics that could hurt someone (illness,
  death). Mix easy and hard words, and avoid near-synonyms in one batch.
- **Curate your own list.** Don't take words from other games, including
  answers they reveal after the fact.

To see the real upcoming answers, for example to proofread them, run
`fly ssh console -C "python -m scripts.words peek --days 14"`. It prints
spoilers, so use it sparingly.

## Deploying to Fly.io

Region `ams`, one `shared-cpu-1x` machine with 512 MB RAM that Fly suspends
when nobody is playing and resumes on the next request in a few hundred
milliseconds. While suspended, only its root filesystem is billed, so a quiet
game costs well under a dollar a month. The process needs about 180 MB; the
rest is page cache for the memory-mapped vectors.

The vectors are built into the image, so machines need no volume. Fly's
builder caches the vectors layer; it only rebuilds when
`scripts/build_vectors.py`, `requirements.txt` or `FREQUENT_WORDS` changes.

The machine count isn't part of `fly.toml`. Keep it at one with
`fly scale count 1`; Fly creates two on the first deploy.

### First deploy

```sh
brew install flyctl
fly auth login
fly apps create kosinus          # names are global; if taken, pick another and update fly.toml

# Generate the secret, keep a copy in your password manager, then store it on Fly.
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
fly secrets set --stage PUZZLE_SECRET=<the value>

fly deploy                       # first build downloads the vectors: about 10 minutes
fly scale count 1                # Fly starts with two machines; one is enough
fly open                         # opens https://kosinus.fly.dev
```

### Your own domain

```sh
fly certs add kosinus.cz         # prints the DNS records to create
fly certs show kosinus.cz        # repeat until the certificate is issued
```

### Deploy on every push to main

```sh
fly tokens create deploy -x 8760h
```

Save the output as the repository secret `FLY_API_TOKEN` (GitHub → Settings →
Secrets and variables → Actions). From then on, every push to `main` that
passes CI is deployed. The token expires after a year.

### Day to day

| Task | Command |
| --- | --- |
| Logs | `fly logs` |
| Status and health checks | `fly status`, `fly checks list` |
| Roll back | `fly releases --image`, then `fly deploy --image <previous image>` |
| More capacity | `fly scale memory 1024`, then `fly scale count 2` |
| IP addresses | `fly ips list`: keep the shared IPv4 (free); a dedicated IPv4 costs extra |

The rate limiter is in memory, so each machine has its own budget. That is
fine for one or two machines. For more, point `RATE_LIMIT_STORAGE_URI` at
Redis (for example Upstash through `fly redis create`).

### Dependency updates

Dependabot (`.github/dependabot.yml`) opens a pull request each month when a
Python package, a GitHub Action or the Docker base image has a new version.
CI runs on each one; merge it if it passes. New Python versions (3.13 and up)
are ignored on purpose until gensim, numpy and scipy publish packages for them.

## Configuration

Environment variables, or `.env` locally. See `.env.example`.

| Variable | Default | Notes |
| --- | --- | --- |
| `PUZZLE_SECRET` | none, required | At least 16 characters. See the warnings above. |
| `VECTORS_PATH` | `vectors/cc.cs.300.kv` | |
| `WORDS_PATH` | `data/words.toml` | |
| `GUESS_RATE_LIMIT` | `30/minute` | Per client IP |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | `redis://…` to share limits between machines |
| `CLIENT_IP_HEADER` | unset | Header with the real client IP. `Fly-Client-IP` on Fly. |
| `DOCS_ENABLED` | `false` | Serves the API docs at `/docs` |
| `LOG_LEVEL` | `INFO` | |

## Credits

Word vectors: [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) by
Facebook AI Research, CC BY-SA 3.0 (Grave et al., *Learning Word Vectors for
157 Languages*, LREC 2018). Lemmatisation and dictionary:
[simplemma](https://github.com/adbar/simplemma), MIT.
