"""Native batch logs preserve configuration, results, failures and append history."""

import json
from pathlib import Path

import _native_cli as cli
import pytest

pytestmark = pytest.mark.usefixtures("cli_cwd")


def _generation(level: int) -> str:
    return (
        f"[[csfsgenerate]]\nas = {level}\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "2s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 2\n"
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_batch_log_saves_config_statistics_and_append_history(
    tmp_path: Path, newline: str
) -> None:
    config = tmp_path / "batch.toml"
    contents = 'conf = "calc"\n' + _generation(0) + _generation(6)
    contents += '[csfs-split.active_spaces]\nAS1 = "1s"\nAS6 = "2s"\n'
    config.write_text(contents, encoding="utf-8", newline=newline)
    assert cli.main(["-c", str(config)]) == 0
    log_path = tmp_path / "rcsfs_calc.log"
    before = log_path.read_bytes()
    events = _events(log_path)
    assert events[0]["event"] == "start"
    assert events[0]["config_contents"] == config.read_bytes().decode("utf-8")
    assert events[0]["working_directory"] == str(tmp_path)
    assert events[0]["implementation"] == "native Rust"
    starts = [event for event in events if event["event"] == "step_start"]
    assert [event["command"] for event in starts] == [
        "csfsgenerate",
        "csfsgenerate",
        "csfs-split",
    ]
    assert starts[0]["resolved_parameters"]["reference_configuration"] == ["1s(2,*)"]
    results = [event for event in events if event["event"] == "result"]
    assert len(results) == 3
    assert results[0]["stats"]["record_count"] > 0
    assert events[-1]["success"] is True and events[-1]["exit_code"] == 0
    assert events[-1]["completed_steps"] == 3
    assert events[-1]["finished_at_unix_ms"] >= events[0]["started_at_unix_ms"]
    assert events[-1]["elapsed_seconds"] >= 0
    assert cli.main(["-c", str(config)]) == 0
    assert log_path.read_bytes().startswith(before)
    assert sum(event["event"] == "start" for event in _events(log_path)) == 2
    assert config.read_text() == contents


@pytest.mark.parametrize(
    "conf,filename", [(None, "rcsfs.log"), ("calc_", "rcsfs_calc_.log")]
)
def test_batch_log_names_and_json_console(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], conf: str | None, filename: str
) -> None:
    contents = (f'conf = "{conf}"\n' if conf else "") + (
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\njson = true\n'
    )
    (tmp_path / "rcsfs.toml").write_text(contents)
    assert cli.main(["-c", "rcsfs.toml"]) == 0
    assert json.loads(capsys.readouterr().out)["record_count"] == 1
    assert _events(tmp_path / filename)[-1]["success"] is True


def test_runtime_failure_is_logged_and_stops_later_jobs(tmp_path: Path) -> None:
    (tmp_path / "batch.toml").write_text(
        'conf = "calc"\n'
        + _generation(0)
        + '[info]\ninput_parquet = "missing.parquet"\n'
        + '[jsplit]\ninput_csf = "calc_as0raw.c"\nprefix = "later"\n'
    )
    assert cli.main(["-c", "batch.toml"]) == 1
    events = _events(tmp_path / "rcsfs_calc.log")
    assert events[-1]["event"] == "complete"
    assert events[-1]["success"] is False and events[-1]["exit_code"] == 1
    assert events[-1]["completed_steps"] == 1
    assert [event["command"] for event in events if event["event"] == "error"] == [
        "info"
    ]
    assert not any(event.get("command") == "jsplit" for event in events)
    assert (tmp_path / "calc_as0raw.c").is_file()
    assert not list(tmp_path.glob("later*"))


def test_unwritable_log_fails_before_processing(tmp_path: Path) -> None:
    (tmp_path / "batch.toml").write_text('conf = "calc"\n' + _generation(0))
    (tmp_path / "rcsfs_calc.log").mkdir()
    assert cli.main(["-c", "batch.toml"]) == 1
    assert not (tmp_path / "calc_as0raw.c").exists()


@pytest.mark.parametrize("role", ["config", "input", "output", "hardlink", "symlink"])
@pytest.mark.parametrize("batch", [True, False])
def test_log_cannot_modify_input_or_output(tmp_path: Path, role: str, batch: bool) -> None:
    log = tmp_path / "rcsfs_calc.log"
    config = log if role == "config" else tmp_path / "batch.toml"
    source = log if role == "input" else tmp_path / "input.c"
    if role != "config":
        source.write_text("original input\n")
    if role == "hardlink":
        log.hardlink_to(source)
    elif role == "symlink":
        log.symlink_to(source)
    output = log.name if role == "output" else "selected.c"
    config.write_text(
        f'conf = "calc"\n[interacting]\nreference = "{source.name}"\n'
        f'candidates = "input.c"\noutput = "{output}"\n'
    )
    before = {path: path.read_bytes() for path in {source, config} if path.exists()}
    assert cli.main([*(["interacting"] if not batch else []), "-c", str(config)]) != 0
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert not (tmp_path / "selected.c").exists()


def test_single_command_logs_details_without_console_noise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "batch.toml").write_text(
        'conf = "calc"\n' + _generation(0).replace("[[csfsgenerate]]", "[csfsgenerate]")
    )
    assert cli.main(["csfsgenerate", "-c", "batch.toml"]) == 0
    captured = capsys.readouterr()
    assert captured.err.strip() == "Run log: rcsfs_calc.log"
    assert "CSFs" in captured.out and "J / parity" in captured.out
    assert "Parsing transcript" not in captured.out
    events = _events(tmp_path / "rcsfs_calc.log")
    assert any(event["event"] == "diagnostic" for event in events)
    assert events[-1]["success"] is True
