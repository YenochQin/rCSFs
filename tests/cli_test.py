from pathlib import Path

import _native_cli as cli
import pytest


@pytest.mark.parametrize(
    ("fixture", "expected_hash"),
    [
        (
            "e1_cc1as1",
            "5c24c1db3a6955317dfd5f2a5fd2370b72d68bb4bd3a80d99b1130232bd22011",
        ),
        (
            "o1_cc1as1",
            "a6d55d2a104c8beb68f399ec07c2e6187525b95f575e32a7c1c927f55cc34bb4",
        ),
    ],
)
def test_interactive_generation_matches_registered_hash(
    tmp_path: Path,
    fixture: str,
    expected_hash: str,
) -> None:
    import hashlib
    import subprocess

    transcript = (
        Path(__file__).parent / "fixtures" / f"{fixture}.rcsfgenerate"
    ).read_text()
    answers = (
        "\n".join(line.split("!")[0].strip() for line in transcript.splitlines()[1:-1])
        + "\n"
    )
    for threads in (1, 2, 4):
        work = tmp_path / str(threads)
        work.mkdir()
        result = subprocess.run(
            [
                str(cli.executable()),
                "csfsgenerate",
                "--threads",
                str(threads),
            ],
            input=answers,
            text=True,
            capture_output=True,
            cwd=work,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with (work / "rcsf.out").open("rb") as output:
            assert hashlib.file_digest(output, "sha256").hexdigest() == expected_hash


pytestmark = pytest.mark.usefixtures("cli_cwd")
