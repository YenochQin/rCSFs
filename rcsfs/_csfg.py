"""Paired publication for the direct GRASPG-compatible CSFG generator."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from ._publication import publish_outputs
from ._rcsfs import generate_csfgs as _generate_csfgs
from ._types import CsfgGenerationStats


def generate_csfgs(
    *,
    inactive_core: int,
    reference_configuration: Sequence[str],
    labeling_space: str,
    active_space: str,
    j_min: int,
    j_max: int,
    excitations: int,
    rcsfg_out: str | Path = "rcsfg.out",
    rlabel_out: str | Path = "rlabel.out",
    threads: int | None = None,
    overwrite: bool = False,
) -> CsfgGenerationStats:
    """Generate a common contiguous S/SD space directly as CSFG representatives.

    J limits are 2J. All occupied references and occupation restrictions must
    lie in ``labeling_space``. Each correlation channel is a continuous radial
    range; pair-only selectors and more than two excitations are rejected.
    Both files are generated and checked before publication. Publication is
    atomic per file, not across the pair; PartialPublicationError reports any
    successfully published path. Existing files are refused unless overwrite
    is explicitly selected. The original full-CSF generator is unchanged.
    """
    for key, value, upper in [
        ("inactive_core", inactive_core, 6),
        ("j_min", j_min, 65535),
        ("j_max", j_max, 65535),
        ("excitations", excitations, 2),
    ]:
        if type(value) is not int or not 0 <= value <= upper:
            raise ValueError(f"{key} must be an integer in 0..{upper}")
    if threads is not None and (type(threads) is not int or threads <= 0):
        raise ValueError("threads must be a positive integer")
    references = _references(reference_configuration)
    _require_string(labeling_space)
    _require_string(active_space)
    _require_boolean(overwrite)
    outputs = [Path(rcsfg_out), Path(rlabel_out)]
    if outputs[0].resolve() == outputs[1].resolve() or (
        all(p.exists() for p in outputs) and outputs[0].samefile(outputs[1])
    ):
        raise ValueError("rcsfg_out and rlabel_out must be distinct paths")
    for path in outputs:
        if not path.parent.is_dir():
            raise ValueError(f"output parent does not exist: {path.parent}")
        if path.is_dir() or path.is_symlink():
            raise ValueError(f"output must be a regular file path: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    with tempfile.TemporaryDirectory(
        prefix="rcsfs-csfg-", dir=outputs[0].parent
    ) as temporary:
        staged = [Path(temporary) / "rcsfg.out", Path(temporary) / "rlabel.out"]
        stats = _generate_csfgs(
            inactive_core,
            references,
            labeling_space,
            active_space,
            j_min,
            j_max,
            excitations,
            str(staged[0]),
            str(staged[1]),
            threads,
        )
        publish_outputs(staged, outputs, overwrite=overwrite)
    stats["rcsfg_out"] = str(outputs[0])
    stats["rlabel_out"] = str(outputs[1])
    return stats


def _require_string(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("orbital spaces and reference configurations must be strings")


def _require_boolean(value: object) -> None:
    if type(value) is not bool:
        raise TypeError("overwrite must be a boolean")


def _references(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise ValueError(
            "reference_configuration must be a nonempty sequence of strings"
        )
    items = cast(Sequence[object], value)
    for item in items:
        _require_string(item)
    return [cast(str, item) for item in items]
