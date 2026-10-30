"""Strict single-model TOML entry for direct CSFG generation."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import cast

from ._csfg import generate_csfgs


def run(config: Path, threads: int | None, overwrite: bool, json_output: bool) -> int:
    try:
        data = cast(
            dict[str, object], tomllib.loads(config.read_text(encoding="utf-8"))
        )
        if set(data) != {"csfggenerate"} or not isinstance(data["csfggenerate"], dict):
            raise ValueError(
                "CSFG mode requires one [csfggenerate] table; batch lists are not supported"
            )
        settings = cast(dict[str, object], data["csfggenerate"])
        allowed = {
            "inactive_core",
            "reference_configuration",
            "labeling_space",
            "active_space",
            "j_min",
            "j_max",
            "excitations",
            "rcsfg_out",
            "rlabel_out",
            "threads",
            "overwrite",
        }
        if unknown := settings.keys() - allowed:
            raise ValueError(f"unsupported CSFG settings: {', '.join(sorted(unknown))}")

        def integer(key: str) -> int:
            value = settings.get(key)
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            return value

        def string(key: str, default: str | None = None) -> str:
            value = settings.get(key, default)
            if not isinstance(value, str):
                raise TypeError(f"{key} must be a string")
            return value

        references = settings.get("reference_configuration")
        if not isinstance(references, list) or not references:
            raise ValueError(
                "reference_configuration must be a nonempty array of strings"
            )
        reference_items = cast(list[object], references)
        if not all(isinstance(item, str) for item in reference_items):
            raise ValueError("reference_configuration must contain only strings")
        configured_overwrite = settings.get("overwrite", False)
        if type(configured_overwrite) is not bool:
            raise ValueError("overwrite must be a boolean")
        outputs = [
            Path(string("rcsfg_out", "rcsfg.out")),
            Path(string("rlabel_out", "rlabel.out")),
        ]
        for path in outputs:
            if path.resolve() == config.resolve() or (
                path.exists() and path.samefile(config)
            ):
                raise ValueError("CSFG outputs must not overwrite the input config")
        if threads is None and "threads" in settings:
            threads = integer("threads")
        stats = generate_csfgs(
            inactive_core=integer("inactive_core"),
            reference_configuration=cast(list[str], reference_items),
            labeling_space=string("labeling_space"),
            active_space=string("active_space"),
            j_min=integer("j_min"),
            j_max=integer("j_max"),
            excitations=integer("excitations"),
            rcsfg_out=outputs[0],
            rlabel_out=outputs[1],
            threads=threads,
            overwrite=overwrite or configured_overwrite,
        )
        if json_output:
            print(json.dumps(stats, indent=2, sort_keys=True))
        else:
            print(
                f"Generated {stats['record_count']} CSF(G) records in {stats['block_count']} blocks; "
                f"expanded space: {stats['expanded_csf_count']} CSFs"
            )
            print(f"Outputs: {stats['rcsfg_out']}, {stats['rlabel_out']}")
        return 0
    except (OSError, ValueError, TypeError) as error:
        if json_output:
            print(json.dumps({"success": False, "error": str(error)}))
        else:
            print(f"CSFG generation failed: {error}", file=sys.stderr)
        return 1
