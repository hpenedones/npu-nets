"""Artifact helpers for the curated residual-MLP NPU path."""

from hashlib import sha1
from pathlib import Path


def source_fingerprint(*paths: Path) -> str:
    digest = sha1()
    for path in paths:
        resolved = Path(path).resolve()
        digest.update(str(resolved).encode('utf-8'))
        digest.update(resolved.read_bytes())
    return digest.hexdigest()[:10]
