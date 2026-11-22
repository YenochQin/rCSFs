"""Seal rewritten test payloads so semantic tests reach row validation."""

import hashlib
from pathlib import Path


def seal(header: Path, parquet: Path, body: bytes | None = None) -> bytes:
    if body is None:
        body = header.read_bytes().split(b"\n", 1)[1]
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    bound = f'descriptor_sha256 = "{digest}"\n'.encode() + body
    header.write_bytes(bound)
    return bound
