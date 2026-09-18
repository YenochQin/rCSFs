import shutil
from pathlib import Path

import pytest


def test_gen_descriptors_reads_header_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    monkeypatch.chdir(tmp_path)
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
        descriptor_version: int = 1,
        header_path: Path | None = None,
        compression: str | None = None,
    ) -> dict[str, object]:
        calls["input_parquet"] = input_parquet
        calls["output_parquet"] = output_parquet
        calls["peel_subshells"] = peel_subshells
        calls["num_workers"] = num_workers
        calls["normalize"] = normalize
        calls["descriptor_version"] = descriptor_version
        calls["header_path"] = header_path
        calls["compression"] = compression
        return {
            "success": True,
            "input_file": str(input_parquet),
            "output_file": str(output_parquet),
            "descriptor_count": 12,
            "descriptor_version": descriptor_version,
            "channels_per_subshell": 3,
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
            "--descriptor-version",
            "1",
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
        "descriptor_version": 1,
        "compression": None,
    }

    captured = capsys.readouterr()
    assert captured.out == (
        "Generated normalized descriptors: descriptors.parquet\ndescriptor_count: 12\n"
    )
    assert captured.err == ""

    sidecar = tmp_path / "descriptors.toml"
    assert sidecar.exists()
    import tomllib

    assert tomllib.loads(sidecar.read_text()) == {
        "format_version": 1,
        "encoding": "parquet",
        "normalized": True,
        "record_count": 12,
        "subshells": ["5s", "4d-", "4d"],
    }


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from rcsfs import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "read_peel_subshells", lambda header_path: ["5s"])
    monkeypatch.setattr(
        cli,
        "generate_descriptors_from_parquet",
        lambda *args, **kwargs: {
            "success": True,
            "input_file": "csf.parquet",
            "output_file": "descriptors.parquet",
            "descriptor_count": 12,
            "descriptor_version": 1,
            "channels_per_subshell": 3,
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
        "descriptor_version": 1,
        "channels_per_subshell": 3,
    }


def test_gen_descriptors_passes_descriptor_version_and_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rcsfs import cli

    monkeypatch.chdir(tmp_path)
    calls: dict[str, object] = {}

    monkeypatch.setattr(cli, "read_peel_subshells", lambda header_path: ["5s"])
    monkeypatch.setattr(
        cli,
        "generate_descriptors_from_parquet",
        lambda *a, **kw: (
            calls.update(kw),
            {
                "success": True,
                "descriptor_count": 1,
                "descriptor_version": kw["descriptor_version"],
                "channels_per_subshell": 4,
            },
        )[1],
    )

    exit_code = cli.main(
        [
            "gen-descriptors",
            "csf.parquet",
            "descriptors.parquet",
            "--header",
            "csf_header.toml",
            "--descriptor-version",
            "2",
        ]
    )

    assert exit_code == 0
    assert calls["descriptor_version"] == 2
    assert calls["header_path"] == Path("csf_header.toml")

    sidecar = tmp_path / "descriptors.toml"
    import tomllib

    assert tomllib.loads(sidecar.read_text())["format_version"] == 2


def test_gen_descriptors_rejects_sidecar_that_aliases_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    header = tmp_path / "calculation_header.toml"
    original = "source header metadata\n"
    header.write_text(original)
    called = False

    def unexpected_generate(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"success": True}

    monkeypatch.setattr(cli, "generate_descriptors_from_parquet", unexpected_generate)
    exit_code = cli.main(
        [
            "gen-descriptors",
            str(tmp_path / "calculation.parquet"),
            str(tmp_path / "calculation_header.parquet"),
            "--header",
            str(header),
            "--descriptor-version",
            "2",
        ]
    )

    assert exit_code == 1
    assert called is False
    assert header.read_text() == original
    assert "descriptor sidecar aliases header" in capsys.readouterr().err


def test_restore_csfs_invokes_restore_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    calls: dict[str, object] = {}

    def fake_restore(descriptor_parquet, header_path, output, indices=None):
        calls["descriptor_parquet"] = descriptor_parquet
        calls["header_path"] = header_path
        calls["output"] = output
        calls["indices"] = indices
        return {
            "success": True,
            "output_file": str(output),
            "record_count": 5,
            "output_bytes": 123,
        }

    monkeypatch.setattr(cli, "restore_csfs_from_descriptors", fake_restore)

    exit_code = cli.main(
        [
            "restore-csfs",
            "--descriptors",
            "desc.parquet",
            "--header",
            "desc_header.toml",
            "--output",
            "restored.c",
            "--indices",
            "0",
            "2",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "descriptor_parquet": Path("desc.parquet"),
        "header_path": Path("desc_header.toml"),
        "output": Path("restored.c"),
        "indices": [0, 2],
    }
    captured = capsys.readouterr()
    assert captured.out == "Restored CSFs: restored.c\nrecord_count: 5\n"
    assert captured.err == ""


def test_restore_csfs_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    def fake_restore(*args: object, **kwargs: object) -> None:
        raise ValueError("header hash mismatch")

    monkeypatch.setattr(cli, "restore_csfs_from_descriptors", fake_restore)

    exit_code = cli.main(
        [
            "restore-csfs",
            "--descriptors",
            "desc.parquet",
            "--header",
            "desc_header.toml",
            "--output",
            "restored.c",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.err == "CSF restoration failed: header hash mismatch\n"


def test_interacting_defaults_to_rcsf_out_and_eight_threads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    calls: dict[str, object] = {}
    successful_stats: dict[str, object] = {
        "exact": False,
        "output_file": "rcsf.out",
        "reference_count": 3,
        "candidate_count": 20,
        "selected_count": 8,
    }

    def fake_select_interacting_csfs(
        reference: Path,
        candidates: Path,
        output: Path,
        *,
        hamiltonian: str,
        method: str,
        num_workers: int | None,
        overwrite: bool,
    ) -> dict[str, object]:
        calls.update(
            reference=reference,
            candidates=candidates,
            output=output,
            hamiltonian=hamiltonian,
            method=method,
            num_workers=num_workers,
            overwrite=overwrite,
        )
        return successful_stats

    monkeypatch.setattr(cli, "select_interacting_csfs", fake_select_interacting_csfs)

    exit_code = cli.main(
        [
            "interacting",
            "rcsfsmr.inp",
            "rcsf.inp",
            "--hamiltonian",
            "dc",
        ]
    )

    assert exit_code == 0
    assert successful_stats["exact"] is False
    assert "success" not in successful_stats
    assert calls == {
        "reference": Path("rcsfsmr.inp"),
        "candidates": Path("rcsf.inp"),
        "output": Path("rcsf.out"),
        "hamiltonian": "dirac_coulomb",
        "method": "structural_upper_bound",
        "num_workers": 8,
        "overwrite": False,
    }
    captured = capsys.readouterr()
    assert "Selected interacting CSFs (structural upper bound): rcsf.out" in (
        captured.out
    )
    assert "selected_count: 8" in captured.out
    assert "STRUCTURAL UPPER BOUND ONLY" in captured.err
    assert "NOT an exact rcsfinteract90" in captured.err


def test_interacting_maps_output_thread_and_overwrite_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rcsfs import cli

    calls: dict[str, object] = {}

    def fake_select_interacting_csfs(
        reference: Path,
        candidates: Path,
        output: Path,
        *,
        hamiltonian: str,
        method: str,
        num_workers: int | None,
        overwrite: bool,
    ) -> dict[str, object]:
        calls.update(
            output=output,
            hamiltonian=hamiltonian,
            method=method,
            num_workers=num_workers,
            overwrite=overwrite,
        )
        return {"exact": False, "output_file": str(output)}

    monkeypatch.setattr(cli, "select_interacting_csfs", fake_select_interacting_csfs)

    exit_code = cli.main(
        [
            "interacting",
            "rcsfsmr.inp",
            "rcsf.inp",
            "--hamiltonian",
            "dcb",
            "--threads",
            "4",
            "--output",
            "selected.csf",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "output": Path("selected.csf"),
        "hamiltonian": "dirac_coulomb_breit",
        "method": "structural_upper_bound",
        "num_workers": 4,
        "overwrite": True,
    }


def test_interacting_rejects_a_positional_output_path() -> None:
    from rcsfs import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "interacting",
                "rcsfsmr.inp",
                "rcsf.inp",
                "selected.csf",
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize("threads", ["0", "-1"])
def test_interacting_rejects_non_positive_thread_counts(threads: str) -> None:
    from rcsfs import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "interacting",
                "rcsfsmr.inp",
                "rcsf.inp",
                "--threads",
                threads,
            ]
        )

    assert exc_info.value.code == 2


def test_interacting_returns_failure_exit_code_and_prints_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    def fail_selection(*args: object, **kwargs: object) -> object:
        raise ValueError("candidate header does not match reference header")

    monkeypatch.setattr(cli, "select_interacting_csfs", fail_selection)

    exit_code = cli.main(["interacting", "rcsfsmr.inp", "rcsf.inp"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "STRUCTURAL UPPER BOUND ONLY" in captured.err
    assert (
        "Interaction selection failed: candidate header does not match reference header"
        in captured.err
    )


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


def test_csfsgenerate_builds_transcript_and_reports_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    answers = iter(
        [
            "*",  # orbital order
            "3",  # core
            "3d(10,i)4s(2,*)4p(6,*)4d(6,*)",  # reference configuration 1
            "3d(10,*)4s(2,i)4p(6,i)4d(6,*)",  # reference configuration 2
            "",  # end reference list
            "5s,5p,5d,4f",  # active orbitals
            "0,12",  # 2J range
            "2",  # excitations
            "n",  # no more lists
        ]
    )
    monkeypatch.setattr(cli, "_prompt", lambda message: next(answers))

    calls: dict[str, object] = {}

    def fake_generate_csfs_from_transcript(
        transcript: str,
        output_path: Path,
        normalize: bool = False,
        threads: int | None = None,
    ) -> dict[str, object]:
        calls["transcript"] = transcript
        calls["output_path"] = output_path
        calls["normalize"] = normalize
        calls["threads"] = threads
        return {
            "success": True,
            "output_file": str(output_path),
            "record_count": 452373,
            "block_count": 7,
        }

    monkeypatch.setattr(
        cli, "generate_csfs_from_transcript", fake_generate_csfs_from_transcript
    )

    exit_code = cli.main(["csfsgenerate", "out.c"])

    assert exit_code == 0
    assert calls["output_path"] == Path("out.c")
    assert calls["normalize"] is False
    assert calls["threads"] is None

    transcript = calls["transcript"]
    assert isinstance(transcript, str)
    lines = transcript.splitlines()
    assert lines == [
        "* ! Orbital order",
        "3",
        "3d(10,i)4s(2,*)4p(6,*)4d(6,*)",
        "3d(10,*)4s(2,i)4p(6,i)4d(6,*)",
        "",
        "5s,5p,5d,4f",
        "0,12",
        "2",
        "n",
    ]

    captured = capsys.readouterr()
    assert "Generated CSFs: out.c" in captured.out
    assert "record_count: 452373" in captured.out
    assert "block_count: 7" in captured.out


def test_csfsgenerate_aborts_on_unsupported_multiple_lists(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rcsfs import cli

    answers = iter(
        [
            "*",
            "3",
            "3d(10,i)4s(2,*)4p(6,*)4d(6,*)",
            "",
            "5s,5p,5d,4f",
            "0,12",
            "2",
            "y",  # requests another list, which is not supported
        ]
    )
    monkeypatch.setattr(cli, "_prompt", lambda message: next(answers))

    called = False

    def fake_generate_csfs_from_transcript(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("must not call generation when continuation is requested")

    monkeypatch.setattr(
        cli, "generate_csfs_from_transcript", fake_generate_csfs_from_transcript
    )

    exit_code = cli.main(["csfsgenerate", "out.c"])

    assert exit_code == 1
    assert called is False
    captured = capsys.readouterr()
    assert "not supported" in captured.err.lower()


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
    import sys

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
                sys.executable,
                "-m",
                "rcsfs.cli",
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
