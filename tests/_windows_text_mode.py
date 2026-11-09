"""Opt-in pytest plugin exercising Windows text-write defaults on any host.

Load with ``-p _windows_text_mode`` and put tests/ on PYTHONPATH. Explicit
newline choices remain unchanged, so canonical CSF fixtures must request LF.
"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def windows_default_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.write_text

    def write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        return original(
            self,
            data,
            encoding=encoding,
            errors=errors,
            newline="\r\n" if newline is None else newline,
        )

    monkeypatch.setattr(Path, "write_text", write_text)
