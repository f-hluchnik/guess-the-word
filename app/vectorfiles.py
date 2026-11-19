"""Names of the files the build script writes next to the vectors."""

from pathlib import Path


def aliases_file(vectors_path: Path) -> Path:
    """Spelling aliases ("zidle" -> "židle"), JSON."""
    return vectors_path.with_name(vectors_path.name + ".aliases.json")


def references_file(vectors_path: Path) -> Path:
    """Reference values (1st/10th/1000th neighbour) of every secret word, JSON.

    Written when the image is built, from data/words.toml, so a cold start
    doesn't need a pass over the whole matrix for today's puzzle.
    """
    return vectors_path.with_name(vectors_path.name + ".references.json")


def norms_file(vectors_path: Path) -> Path:
    """Precomputed vector lengths, so startup doesn't have to read every vector."""
    return vectors_path.with_name(vectors_path.name + ".norms.npy")
