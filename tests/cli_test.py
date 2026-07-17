from pathlib import Path

import pytest

import shutil


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
        compression: str | None = None,
    ) -> dict[str, object]:
        calls["input_parquet"] = input_parquet
        calls["output_parquet"] = output_parquet
        calls["peel_subshells"] = peel_subshells
        calls["num_workers"] = num_workers
        calls["normalize"] = normalize
        calls["compression"] = compression
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
        "compression": None,
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


def test_zero_first_orchestrates_convert_and_partition(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    convert_calls: list[dict[str, object]] = []
    partition_calls: list[dict[str, object]] = []

    def fake_convert(
        input_path: Path,
        output_path: Path,
        max_line_len: int | None = None,
        num_workers: int | None = None,
    ) -> dict[str, object]:
        convert_calls.append(
            {
                "input": Path(input_path),
                "output": Path(output_path),
                "max_line_len": max_line_len,
                "num_workers": num_workers,
            }
        )
        return {
            "success": True,
            "header_file": str(Path(output_path).parent / "fake_header.toml"),
        }

    def fake_partition(
        zero_parquet: Path,
        zero_header: str,
        full_parquet: Path,
        full_header: str,
        output_csf: Path,
    ) -> dict[str, object]:
        partition_calls.append(
            {
                "zero_parquet": Path(zero_parquet),
                "zero_header": zero_header,
                "full_parquet": Path(full_parquet),
                "full_header": full_header,
                "output_csf": Path(output_csf),
            }
        )
        return {
            "success": True,
            "output_file": str(output_csf),
            "first_order_count": 5,
            "block_count": 2,
        }

    monkeypatch.setattr(cli, "convert_csfs", fake_convert)
    monkeypatch.setattr(cli, "partition_csfs", fake_partition)

    exit_code = cli.main(
        ["zero-first", "zero.csf", "full.csf", "out.csf", "--num-workers", "4"]
    )

    assert exit_code == 0
    assert len(convert_calls) == 2
    assert convert_calls[0]["input"] == Path("zero.csf")
    assert convert_calls[1]["input"] == Path("full.csf")
    assert all(c["num_workers"] == 4 for c in convert_calls)
    # outputs isolated into zero/ and full/ subdirs (no stem collision)
    assert convert_calls[0]["output"].parent.name == "zero"
    assert convert_calls[1]["output"].parent.name == "full"

    assert len(partition_calls) == 1
    assert partition_calls[0]["output_csf"] == Path("out.csf")

    captured = capsys.readouterr()
    assert "Partitioned CSFs: out.csf" in captured.out
    assert "first_order_count: 5" in captured.out
    assert "block_count: 2" in captured.out


def test_zero_first_default_output_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    partition_outputs: list[Path] = []

    monkeypatch.setattr(
        cli,
        "convert_csfs",
        lambda input_path, output_path, max_line_len=None, num_workers=None: {
            "success": True,
            "header_file": str(Path(output_path).parent / "fake_header.toml"),
        },
    )

    def fake_partition(
        zero_parquet: Path,
        zero_header: str,
        full_parquet: Path,
        full_header: str,
        output_csf: Path,
    ) -> dict[str, object]:
        partition_outputs.append(Path(output_csf))
        return {"success": True, "output_file": str(output_csf)}

    monkeypatch.setattr(cli, "partition_csfs", fake_partition)

    exit_code = cli.main(["zero-first", "zero.csf", "full.csf"])

    assert exit_code == 0
    expected = Path("full.csf").with_name("full_zf.csf")
    assert partition_outputs[0] == expected


def test_zero_first_keep_parquet_keeps_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    monkeypatch.setattr(
        cli,
        "convert_csfs",
        lambda input_path, output_path, max_line_len=None, num_workers=None: {
            "success": True,
            "header_file": str(Path(output_path).parent / "fake_header.toml"),
        },
    )
    monkeypatch.setattr(
        cli,
        "partition_csfs",
        lambda zero_parquet, zero_header, full_parquet, full_header, output_csf: {
            "success": True,
            "output_file": str(output_csf),
        },
    )

    exit_code = cli.main(
        ["zero-first", "zero.csf", "full.csf", "out.csf", "--keep-parquet"]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Intermediate Parquet kept under:" in captured.out
    kept_line = next(
        line for line in captured.out.splitlines() if "kept under:" in line
    )
    kept_path = Path(kept_line.split("kept under:", 1)[1].strip())
    assert kept_path.exists(), "kept temp dir should still exist after run"
    shutil.rmtree(kept_path, ignore_errors=True)


def test_zero_first_propagates_partition_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    monkeypatch.setattr(
        cli,
        "convert_csfs",
        lambda input_path, output_path, max_line_len=None, num_workers=None: {
            "success": True,
            "header_file": str(Path(output_path).parent / "fake_header.toml"),
        },
    )
    monkeypatch.setattr(
        cli,
        "partition_csfs",
        lambda zero_parquet, zero_header, full_parquet, full_header, output_csf: {
            "success": False,
            "error": "boom",
        },
    )

    exit_code = cli.main(["zero-first", "zero.csf", "full.csf", "out.csf"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Partition failed: boom" in captured.err


def test_zero_first_propagates_convert_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    monkeypatch.setattr(
        cli,
        "convert_csfs",
        lambda input_path, output_path, max_line_len=None, num_workers=None: {
            "success": False,
            "error": "bad input",
        },
    )
    monkeypatch.setattr(
        cli,
        "partition_csfs",
        lambda *a, **k: {"success": True, "output_file": "out.csf"},
    )

    exit_code = cli.main(["zero-first", "zero.csf", "full.csf", "out.csf"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "conversion failed: bad input" in captured.err
