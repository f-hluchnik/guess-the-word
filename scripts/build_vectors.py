"""Build the game's vector file from Facebook's fastText Czech crawl vectors.

    python -m scripts.build_vectors                 # download + build
    python -m scripts.build_vectors --source cc.cs.300.vec.gz --frequent 200000

Only lowercase tokens made of Czech letters are kept, which drops capitalised
duplicates, numbers, URLs and other crawl debris. Of those, the vocabulary is

- the `--frequent` most frequent tokens, whatever they are (slang, rare
  inflections and neologisms live here), plus
- every less frequent token that is a word in simplemma's Czech dictionary.

Past roughly 300,000 tokens, more than 80 % of the list is typos, glued words
and Slovak forms, and they crowd the nearest neighbours of the secret words.
The dictionary pass keeps the rare real words without that noise.

Writes <output> plus <output>.vectors.npy; the server memory-maps both.
"""

import argparse
import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np
import simplemma
from gensim.models import KeyedVectors

from app.text import is_word, normalize

SOURCE_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz"

log = logging.getLogger("build_vectors")


def build(source: Path, frequent: int) -> KeyedVectors:
    words: list[str] = []
    rows: list[np.ndarray] = []
    seen: set[str] = set()

    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as lines:
        _, dim = map(int, next(lines).split())
        for line in lines:
            token, _, values = line.rstrip().partition(" ")
            word = normalize(token)
            if word != token or not is_word(word) or word in seen:
                continue
            seen.add(word)  # the first, most frequent spelling wins
            if len(seen) > frequent and not simplemma.is_known(word, lang="cs"):
                continue
            rows.append(np.array(values.split(), dtype=np.float32))
            words.append(word)
            if len(words) % 50_000 == 0:
                log.info("%d words", len(words))

    log.info("%d frequent + %d dictionary words", min(frequent, len(words)), len(words) - frequent)
    vectors = KeyedVectors(dim, dtype=np.float32)
    vectors.add_vectors(words, np.vstack(rows))
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=Path("vectors/cc.cs.300.vec.gz"))
    parser.add_argument("--output", type=Path, default=Path("vectors/cc.cs.300.kv"))
    parser.add_argument(
        "--frequent", type=int, default=300_000, help="most frequent tokens to keep as they are"
    )
    parser.add_argument(
        "--delete-source", action="store_true", help="remove the source file when done"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s (about 1.3 GB)", SOURCE_URL)
        urllib.request.urlretrieve(SOURCE_URL, args.source)

    vectors = build(args.source, args.frequent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vectors.save(str(args.output))
    log.info("Saved %d words to %s", len(vectors), args.output)
    if args.delete_source:
        args.source.unlink()


if __name__ == "__main__":
    main()
