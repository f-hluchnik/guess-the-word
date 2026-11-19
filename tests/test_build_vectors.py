import gzip
from pathlib import Path

import numpy as np

from scripts.build_vectors import build


def write_vec(path: Path, tokens: list[str], dim: int = 4) -> Path:
    rng = np.random.default_rng(0)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(f"{len(tokens)} {dim}\n")
        for token in tokens:
            f.write(token + " " + " ".join(f"{x:.4f}" for x in rng.normal(size=dim)) + "\n")
    return path


def test_build_keeps_frequent_and_dictionary_words_and_aliases_plain_spellings(
    tmp_path: Path,
) -> None:
    source = write_vec(
        tmp_path / "toy.vec.gz",
        [
            "židle",  # dictionary word
            "Židle",  # capitalised duplicate: dropped
            "laptop",  # frequent, not in the dictionary, no diacritic twin: kept
            "zidle",  # plain spelling of a more frequent dictionary word: alias
            "2024",  # not a word: dropped
            "kočka",  # beyond --frequent but a dictionary word: kept
            "kocka",  # beyond --frequent, plain spelling: alias
            "xqzwv",  # beyond --frequent, not a dictionary word: dropped
        ],
    )
    vectors, aliases = build(source, frequent=3)
    assert vectors.index_to_key == ["židle", "laptop", "kočka"]
    assert aliases == {"zidle": "židle", "kocka": "kočka"}
