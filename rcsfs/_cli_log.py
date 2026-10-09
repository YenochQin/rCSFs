"""Persistent batch summaries without changing the CLI's console output."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime, timezone
from io import TextIOBase
from pathlib import Path
from time import perf_counter
from typing import TextIO

_ACTIVE_LOG: ContextVar[RunLog | None] = ContextVar("rcsfs_run_log", default=None)


class RunLogError(Exception):
    """A log write failed; keep this separate from processing I/O failures."""


class RunLog:
    """One appended run, with a separate result and duration for each step."""

    def __init__(self, stream: TextIO) -> None:
        self.stream: TextIO = stream
        self.exit_code: int = 1

    def write(self, text: str) -> None:
        try:
            _ = self.stream.write(text)
            self.stream.flush()
        except OSError as exc:
            raise RunLogError(str(exc)) from exc

    def record(self, label: str, value: Mapping[str, object]) -> None:
        self.write(
            f"\n{label}:\n{json.dumps(value, indent=2, default=str, ensure_ascii=False)}\n"
        )

    def run_step(
        self,
        label: str,
        parameters: Mapping[str, object],
        command: Callable[[], int],
    ) -> int:
        self.write(f"\n--- [{label}] ---\n")
        started = perf_counter()
        result = 1
        try:
            result = command()
            return result
        except KeyboardInterrupt:
            result = 130
            raise
        finally:
            elapsed = perf_counter() - started
            self.exit_code = result
            # Record after execution so backend defaults resolved by the runner
            # and automatic Parquet generation for splitting are visible.
            self.record("effective_parameters", parameters)
            self.write(
                f"step_status: {'success' if result == 0 else 'interrupted' if result == 130 else 'failed'}\n"
                f"step_exit_code: {result}\n"
                f"step_elapsed_seconds: {elapsed:.6f}\n"
            )


class _Tee(TextIOBase):
    """Mirror Python CLI messages, leaving native progress output alone."""

    def __init__(self, console: TextIO, log: RunLog) -> None:
        self.console: TextIO = console
        self.log: RunLog = log

    def write(self, text: str) -> int:
        written = self.console.write(text)
        self.log.write(text)
        return written

    def flush(self) -> None:
        self.console.flush()

    def isatty(self) -> bool:
        return self.console.isatty()

    def fileno(self) -> int:
        return self.console.fileno()


def record_stats(stats: Mapping[str, object]) -> None:
    """Save the complete returned statistics only during a logged batch run."""
    log = _ACTIVE_LOG.get()
    if log is not None:
        log.record("result_statistics", stats)


@contextmanager
def capture_run(
    stream: TextIO, config: Path, contents: str, version: str
) -> Generator[RunLog]:
    """Append a config snapshot, CLI messages, statistics, and a final status."""
    log = RunLog(stream)
    started = perf_counter()
    log.write(
        f"\n=== rCSFs run ===\nstarted_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"rcsfs_version: {version}\nconfig: {config.resolve()}\n"
        f"working_directory: {Path.cwd()}\n\n--- configuration ---\n"
        f"{contents.rstrip()}\n--- end configuration ---\n"
    )
    token = _ACTIVE_LOG.set(log)
    status = "failed"
    try:
        with (
            redirect_stdout(_Tee(sys.stdout, log)),
            redirect_stderr(_Tee(sys.stderr, log)),
        ):
            yield log
        status = "success" if log.exit_code == 0 else "failed"
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
            log.exit_code = 130
        log.write(f"\nexception: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        _ACTIVE_LOG.reset(token)
        log.write(
            f"\nfinished_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"status: {status}\nexit_code: {log.exit_code}\n"
            f"elapsed_seconds: {perf_counter() - started:.6f}\n=== end run ===\n"
        )
