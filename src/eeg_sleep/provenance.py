"""File-based provenance helpers used by scientific experiment artifacts."""
from __future__ import annotations
from pathlib import Path
import hashlib


def sha256_file(path: str | Path) -> str:
    """Compute a deterministic dataset digest with bounded memory."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()
