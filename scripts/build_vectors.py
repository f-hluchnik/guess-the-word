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

Spellings without diacritics ("zidle") of a more frequent dictionary word
("židle") are not kept as words. Their vectors come from text typed without
diacritics and sit among other such words, only about 50 % similar to the
word they stand for. They are written to an alias file instead, so a player
typing "zidle" is measured as "židle".

Writes <output>, <output>.vectors.npy, <output>.aliases.json and
<output>.norms.npy (vector lengths, so the server needn't read every vector
at startup).
"""

import argparse
import gzip
import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import simplemma
from gensim.models import KeyedVectors

from app.text import is_word, normalize, strip_diacritics
from app.vectorfiles import aliases_file, norms_file

SOURCE_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz"

log = logging.getLogger("build_vectors")


def build(source: Path, frequent: int) -> tuple[KeyedVectors, dict[str, str]]:
    words: list[str] = []
    rows: list[np.ndarray] = []
    seen: set[str] = set()
    # Diacritic-free spelling -> most frequent dictionary word spelled that way.
    spelled: dict[str, str] = {}
    aliases: dict[str, str] = {}

    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as lines:
        _, dim = map(int, next(lines).split())
        for line in lines:
            token, _, values = line.rstrip().partition(" ")
            word = normalize(token)
            if word != token or not is_word(word) or word in seen:
                continue
            seen.add(word)  # the first, most frequent spelling wins
            known = simplemma.is_known(word, lang="cs")
            plain = strip_diacritics(word)
            if known and plain != word:
                spelled.setdefault(plain, word)
            elif not known and plain == word and plain in spelled:
                aliases[word] = spelled[plain]  # "zidle" -> "židle", seen earlier
                continue
            if len(seen) > frequent and not known:
                continue
            rows.append(np.array(values.split(), dtype=np.float32))
            words.append(word)
            if len(words) % 50_000 == 0:
                log.info("%d words", len(words))

    log.info("%d words, %d spelling aliases", len(words), len(aliases))
    vectors = KeyedVectors(dim, dtype=np.float32)
    vectors.add_vectors(words, np.vstack(rows))
    return vectors, aliases


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

    vectors, aliases = build(args.source, args.frequent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vectors.save(str(args.output))
    aliases_file(args.output).write_text(
        json.dumps(aliases, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    np.save(norms_file(args.output), np.linalg.norm(vectors.vectors, axis=1).astype(np.float32))
    log.info("Saved %d words and %d aliases next to %s", len(vectors), len(aliases), args.output)
    if args.delete_source:
        args.source.unlink()


if __name__ == "__main__":
    main()
