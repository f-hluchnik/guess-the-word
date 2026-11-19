"""Names of the files the build script writes next to the vectors."""

from pathlib import Path


def aliases_file(vectors_path: Path) -> Path:
    """Spelling aliases ("zidle" -> "židle"), JSON."""
    return vectors_path.with_name(vectors_path.name + ".aliases.json")


def norms_file(vectors_path: Path) -> Path:
    """Precomputed vector lengths, so startup doesn't have to read every vector."""
    return vectors_path.with_name(vectors_path.name + ".norms.npy")
