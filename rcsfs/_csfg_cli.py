"""Single-model and layered batch TOML entry for CSFG generation."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import cast

from ._csfg import generate_csfgs
from ._publication import PartialPublicationError, publish_outputs
from ._types import CsfgGenerationStats


def _table(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a table")
    return cast(dict[str, object], value)


def _number(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("as must be a nonnegative integer")
    return value


def _space(value: object) -> dict[str, int]:
    if not isinstance(value, str):
        raise TypeError("active space must be a string")
    result: dict[str, int] = {}
    for token in value.split(","):
        match = re.fullmatch(r"\s*([1-9][0-9]*)([spdfghiklmnoqrtuvwxyz])\s*", token)
        if match is None or match[2] in result:
            raise ValueError(f"invalid or duplicate active-space channel: {token}")
        result[match[2]] = int(match[1])
    return result


def _plan(data: dict[str, object]) -> tuple[list[dict[str, object]], bool]:
    if unknown := data.keys() - {"conf", "csfggenerate", "csfg-split", "csfs-split"}:
        raise ValueError(f"unsupported CSFG sections: {', '.join(sorted(unknown))}")
    raw = data.get("csfggenerate")
    batch = isinstance(raw, list)
    items = cast(list[object], raw) if batch else [raw]
    if not items:
        raise ValueError("csfggenerate must contain at least one model")
    conf = data.get("conf")
    if conf is not None and (
        not isinstance(conf, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", conf)
    ):
        raise ValueError("conf must be a nonempty filename stem without directories")
    jobs: list[dict[str, object]] = []
    indexed: dict[int, dict[str, object]] = {}

    def named(settings: dict[str, object], number: int) -> dict[str, object]:
        if conf is None:
            raise ValueError("conf is required for numbered CSFG outputs")
        return {
            **settings,
            "rcsfg_out": f"{conf}_as{number}raw.g",
            "rlabel_out": f"{conf}_as{number}raw.l",
        }

    for item in items:
        settings = dict(_table(item, "csfggenerate"))
        if batch or "as" in settings:
            number = _number(settings.pop("as", None))
            if number in indexed:
                raise ValueError(f"duplicate generation as = {number}")
            if "rcsfg_out" in settings or "rlabel_out" in settings:
                raise ValueError(
                    "numbered models use conf filenames, not explicit output paths"
                )
            settings = named(settings, number)
            indexed[number] = settings
        jobs.append(settings)
    if "csfg-split" in data and "csfs-split" in data:
        raise ValueError("use only one of csfg-split and csfs-split")
    split = data.get("csfg-split", data.get("csfs-split"))
    if split is not None:
        section = _table(split, "csfg-split")
        if section.keys() - {"active_spaces", "source_as"}:
            raise ValueError("csfg-split supports only active_spaces and source_as")
        if not indexed:
            raise ValueError("CSFG layers require numbered models and conf")
        source_as = _number(section.get("source_as", max(indexed)))
        if source_as not in indexed:
            raise ValueError("source_as must identify a csfggenerate model")
        source = indexed[source_as]
        maximum = _space(source.get("active_space"))
        layers = _table(section.get("active_spaces"), "active_spaces")
        if not layers:
            raise ValueError("active_spaces must not be empty")
        for name, space in layers.items():
            match = re.fullmatch(r"AS(0|[1-9][0-9]*)", name)
            if match is None:
                raise ValueError(
                    "layer names must be AS followed by a nonnegative integer"
                )
            number = int(match[1])
            target = _space(space)
            if any(n > maximum.get(l, 0) for l, n in target.items()):
                raise ValueError(f"{name} exceeds the source active space")
            settings = named({**source, "active_space": space}, number)
            if number in indexed:
                if settings != indexed[number]:
                    raise ValueError(
                        f"{name} conflicts with an explicit generation model"
                    )
                continue
            indexed[number] = settings
            jobs.append(settings)
        batch = True
    return jobs, batch


def _generate(
    settings: dict[str, object], config: Path, threads: int | None, overwrite: bool
) -> CsfgGenerationStats:
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
        raise ValueError("reference_configuration must be a nonempty array of strings")
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
    return generate_csfgs(
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


def run(config: Path, threads: int | None, overwrite: bool, json_output: bool) -> int:
    try:
        data = cast(
            dict[str, object], tomllib.loads(config.read_text(encoding="utf-8"))
        )
        jobs, batch = _plan(data)
        destinations: list[Path] = []
        policies: list[bool] = []
        for settings in jobs:
            configured = settings.get("overwrite", False)
            if type(configured) is not bool:
                raise ValueError("overwrite must be a boolean")
            for key, default in [
                ("rcsfg_out", "rcsfg.out"),
                ("rlabel_out", "rlabel.out"),
            ]:
                value = settings.get(key, default)
                if not isinstance(value, str):
                    raise TypeError(f"{key} must be a string")
                path = Path(value)
                for previous in [config, *destinations]:
                    if path.resolve() == previous.resolve() or (
                        path.exists() and previous.exists() and path.samefile(previous)
                    ):
                        raise ValueError(
                            "CSFG outputs must be distinct and must not overwrite the input config"
                        )
                if path.is_symlink() or path.is_dir() or not path.parent.is_dir():
                    raise ValueError(f"invalid output path: {path}")
                if path.exists() and not (overwrite or configured):
                    raise FileExistsError(path)
                destinations.append(path)
                policies.append(overwrite or configured)
        results: list[CsfgGenerationStats] = []
        # Validate/generate every model before publishing any batch output.
        with tempfile.TemporaryDirectory(
            prefix="rcsfs-csfg-batch-", dir=destinations[0].parent
        ) as temporary:
            staged: list[Path] = []
            for index, settings in enumerate(jobs):
                pair = [Path(temporary) / f"{index}.g", Path(temporary) / f"{index}.l"]
                result = _generate(
                    {**settings, "rcsfg_out": str(pair[0]), "rlabel_out": str(pair[1])},
                    config,
                    threads,
                    False,
                )
                result["rcsfg_out"] = str(destinations[2 * index])
                result["rlabel_out"] = str(destinations[2 * index + 1])
                results.append(result)
                staged.extend(pair)
            # Retain cumulative diagnostics across per-model overwrite policies.

            published: list[Path] = []
            for src, dst, policy in zip(staged, destinations, policies, strict=True):
                try:
                    publish_outputs([src], [dst], overwrite=policy)
                except PartialPublicationError as error:
                    raise PartialPublicationError(
                        dst, published + error.published, error
                    ) from error
                published.append(dst)
        if json_output:
            print(
                json.dumps(
                    {"success": True, "results": results} if batch else results[0],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            for stats in results:
                print(
                    f"Generated {stats['record_count']} CSF(G) records; expanded space: {stats['expanded_csf_count']} CSFs"
                )
                print(f"Outputs: {stats['rcsfg_out']}, {stats['rlabel_out']}")
        return 0
    except (OSError, ValueError, TypeError) as error:
        if json_output:
            print(json.dumps({"success": False, "error": str(error)}))
        else:
            print(f"CSFG generation failed: {error}", file=sys.stderr)
        return 1
