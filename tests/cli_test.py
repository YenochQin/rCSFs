from pathlib import Path

import pytest


def test_gen_descriptors_reads_header_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    calls: dict[str, object] = {}

    def fake_read_peel_subshells(header_path: Path) -> list[str]:
        calls["header_path"] = header_path
        return ["5s", "4d-", "4d"]

    def fake_generate_descriptors_from_parquet(
        input_parquet: Path,
        output_parquet: Path,
        peel_subshells: list[str],
        num_workers: int | None = None,
        normalize: bool = False,
    ) -> dict[str, object]:
        calls["input_parquet"] = input_parquet
        calls["output_parquet"] = output_parquet
        calls["peel_subshells"] = peel_subshells
        calls["num_workers"] = num_workers
        calls["normalize"] = normalize
        return {
            "success": True,
            "input_file": str(input_parquet),
            "output_file": str(output_parquet),
            "descriptor_count": 12,
        }

    monkeypatch.setattr(cli, "read_peel_subshells", fake_read_peel_subshells)
    monkeypatch.setattr(
        cli,
        "generate_descriptors_from_parquet",
        fake_generate_descriptors_from_parquet,
    )

    exit_code = cli.main(
        [
            "gen-descriptors",
            "csf.parquet",
            "descriptors.parquet",
            "--header",
            "csf_header.toml",
            "--num-workers",
            "2",
            "--normalize",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "header_path": Path("csf_header.toml"),
        "input_parquet": Path("csf.parquet"),
        "output_parquet": Path("descriptors.parquet"),
        "peel_subshells": ["5s", "4d-", "4d"],
        "num_workers": 2,
        "normalize": True,
    }

    captured = capsys.readouterr()
    assert captured.out == (
        "Generated normalized descriptors: descriptors.parquet\n"
        "descriptor_count: 12\n"
    )
    assert captured.err == ""


def test_gen_descriptors_returns_failure_exit_code_and_prints_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    monkeypatch.setattr(cli, "read_peel_subshells", lambda header_path: ["5s"])
    monkeypatch.setattr(
        cli,
        "generate_descriptors_from_parquet",
        lambda *args, **kwargs: {
            "success": False,
            "input_file": "csf.parquet",
            "output_file": "descriptors.parquet",
            "error": "failed",
        },
    )

    exit_code = cli.main(
        [
            "gen-descriptors",
            "csf.parquet",
            "descriptors.parquet",
            "--header",
            "csf_header.toml",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Descriptor generation failed: failed\n"


def test_gen_descriptors_can_print_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from rcsfs import cli

    monkeypatch.setattr(cli, "read_peel_subshells", lambda header_path: ["5s"])
    monkeypatch.setattr(
        cli,
        "generate_descriptors_from_parquet",
        lambda *args, **kwargs: {
            "success": True,
            "input_file": "csf.parquet",
            "output_file": "descriptors.parquet",
            "descriptor_count": 12,
        },
    )

    exit_code = cli.main(
        [
            "gen-descriptors",
            "csf.parquet",
            "descriptors.parquet",
            "--header",
            "csf_header.toml",
            "--json",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "success": True,
        "input_file": "csf.parquet",
        "output_file": "descriptors.parquet",
        "descriptor_count": 12,
    }
