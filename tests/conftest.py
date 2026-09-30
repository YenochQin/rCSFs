from pathlib import Path

import pytest


@pytest.fixture
def cli_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run CLI checks from their own isolated invocation directory."""
    monkeypatch.chdir(tmp_path)
