"""Batch logs preserve generation/split results, failures, and previous runs."""

import json
import sys
from pathlib import Path

import pytest

from rcsfs import cli
from rcsfs._cli_log import record_stats

pytestmark = pytest.mark.usefixtures("cli_cwd")


def _generation(level: int, *, reference: str = "1s(2,*)") -> str:
    return (
        f"[[csfsgenerate]]\nas = {level}\ninactive_core = 0\n"
        f'reference_configuration = ["{reference}"]\nactive_space = "2s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 2\ngenerate_descriptors = false\n"
    )


def test_batch_log_saves_config_generation_and_split_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config = inputs / "calculation.toml"
    contents = (
        'conf = "e1_vv"\n'
        + _generation(0)
        + _generation(6)
        + '[csfs-split.active_spaces]\nAS1 = "1s"\nAS6 = "2s"\n'
    )
    config.write_text(contents, encoding="utf-8")

    assert cli.main(["-c", str(config)]) == 0
    console = capsys.readouterr()
    log = (tmp_path / "rcsfs_e1_vv.log").read_text(encoding="utf-8")
    assert "Run log: rcsfs_e1_vv.log" in console.err
    assert contents.rstrip() in log
    assert str(config) in log
    assert f"working_directory: {tmp_path}" in log
    assert "rcsfs_version:" in log
    assert "started_at:" in log and "finished_at:" in log
    assert log.index("--- [csfsgenerate #1]") < log.index("--- [csfsgenerate #2]")
    assert log.index("--- [csfsgenerate #2]") < log.index("--- [csfs-split]")
    assert log.count("result_statistics:") == 3
    for field in (
        "record_count",
        "block_count",
        "stage_stats",
        "resource_stats",
        "csf_count",
        "block_lengths",
    ):
        assert f'"{field}":' in log
    assert '"generation_storage": "disk"' in log
    assert '"generate_parquet": true' in log
    assert '"output_file": "e1_vv_as6raw.c"' in log
    assert "e1_vv_as1raw.c:" in console.out
    assert all(line in log for line in console.out.splitlines())
    assert "status: success\nexit_code: 0\n" in log
    assert "elapsed_seconds:" in log
    assert not (inputs / "rcsfs_e1_vv.log").exists()
    assert config.read_text(encoding="utf-8") == contents

    assert cli.main(["-c", str(config)]) == 0
    appended = (tmp_path / "rcsfs_e1_vv.log").read_text(encoding="utf-8")
    assert appended.startswith(log)
    assert appended.count("=== rCSFs run ===") == 2
    assert appended.count("=== end run ===") == 2


@pytest.mark.parametrize(
    "conf, filename", [(None, "rcsfs.log"), ("calc_", "rcsfs_calc_.log")]
)
def test_batch_log_default_name_and_json_console(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], conf: str | None, filename: str
) -> None:
    contents = (f'conf = "{conf}"\n' if conf else "") + (
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\njson = true\n'
    )
    (tmp_path / "rcsfs.toml").write_text(contents, encoding="utf-8")
    assert cli.main(["-c", "rcsfs.toml"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["success"] is True
    assert result["record_count"] == 1
    assert '"record_count": 1' in (tmp_path / filename).read_text(encoding="utf-8")


def test_batch_log_retains_failure_and_stops_before_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        'conf = "calc"\n'
        + _generation(0)
        + _generation(6, reference="1s(3,*)")
        + '[csfs-split.active_spaces]\nAS1 = "1s"\n',
        encoding="utf-8",
    )
    assert cli.main(["-c", "rcsfs.toml"]) == 1
    error = capsys.readouterr().err
    log = (tmp_path / "rcsfs_calc.log").read_text(encoding="utf-8")
    assert "CSF generation failed:" in error
    assert error.split("CSF generation failed:", 1)[1].strip() in log
    assert '"success": false' in log
    assert "step_status: failed\nstep_exit_code: 1\n" in log
    assert "status: failed\nexit_code: 1\n" in log
    assert "--- [csfs-split]" not in log
    assert (tmp_path / "calc_as0raw.c").exists()
    assert not (tmp_path / "calc_as6raw.c").exists()


@pytest.mark.parametrize(
    "exception, exit_code",
    [(RuntimeError("unexpected failure"), 1), (KeyboardInterrupt(), 130)],
)
def test_batch_log_records_exceptions_and_restores_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: BaseException,
    exit_code: int,
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        'conf = "calc"\n' + _generation(0) + _generation(6), encoding="utf-8"
    )
    calls = 0

    def run(args: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise exception
        return 0

    monkeypatch.setattr(cli, "_run_parsed_command", run)
    stdout, stderr = sys.stdout, sys.stderr
    with pytest.raises(type(exception)):
        cli.main(["-c", "rcsfs.toml"])
    assert sys.stdout is stdout and sys.stderr is stderr
    log_path = tmp_path / "rcsfs_calc.log"
    contents = log_path.read_text(encoding="utf-8")
    assert f"exception: {type(exception).__name__}:" in contents
    assert f"exit_code: {exit_code}\n" in contents
    status = "interrupted" if exit_code == 130 else "failed"
    assert f"status: {status}\nexit_code: {exit_code}\n" in contents
    record_stats({"outside_run": True})
    assert log_path.read_text(encoding="utf-8") == contents


def test_unwritable_batch_log_fails_before_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        'conf = "calc"\n' + _generation(0), encoding="utf-8"
    )
    (tmp_path / "rcsfs_calc.log").mkdir()

    def run(args: object) -> int:
        pytest.fail("processing must not start without a writable run log")

    monkeypatch.setattr(cli, "_run_parsed_command", run)
    assert cli.main(["-c", "rcsfs.toml"]) == 1
    assert "Cannot write run log" in capsys.readouterr().err
    assert not (tmp_path / "calc_as0raw.c").exists()


@pytest.mark.parametrize("role", ["config", "input", "output", "hardlink", "symlink"])
def test_log_cannot_modify_input_or_output_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], role: str
) -> None:
    log_path = tmp_path / "rcsfs_calc.log"
    config = log_path if role == "config" else tmp_path / "rcsfs.toml"
    input_path = log_path if role == "input" else tmp_path / "input.c"
    if role != "config":
        input_path.write_text("original input\n", encoding="utf-8")
    if role in {"hardlink", "symlink"}:
        if role == "hardlink":
            log_path.hardlink_to(input_path)
        else:
            log_path.symlink_to(input_path)
    output = log_path.name if role == "output" else "selected.c"
    config.write_text(
        f'conf = "calc"\n[interacting]\nreference = "{input_path.name}"\n'
        f'candidates = "input.c"\noutput = "{output}"\n',
        encoding="utf-8",
    )
    originals = {p: p.read_bytes() for p in {input_path, config} if p.exists()}
    with pytest.raises(SystemExit) as exc:
        cli.main(["-c", str(config)])
    assert exc.value.code == 2
    assert "run log" in capsys.readouterr().err
    for path, contents in originals.items():
        assert path.read_bytes() == contents
    assert not (tmp_path / "selected.c").exists()


def test_single_command_does_not_create_batch_log(tmp_path: Path) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        'conf = "calc"\n'
        + _generation(0).replace("[[csfsgenerate]]", "[csfsgenerate]"),
        encoding="utf-8",
    )
    assert cli.main(["csfsgenerate", "-c"]) == 0
    assert not list(tmp_path.glob("*.log"))
