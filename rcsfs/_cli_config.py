"""One configuration boundary for every rcsfs CLI subcommand.

The selected command table and optional shared naming stem are applied. CLI
values are parsed sparsely so an explicit argument always wins over the TOML
value, even for false/zero-like values. Paths retain current-directory semantics.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

DEFAULT_CONFIG = Path("rcsfs.toml")
CONFIG_SECTIONS = {
    "csfsgenerate": "csfsgenerate",
    "gen-descriptors": "gen-descriptors",
    "zero-first": "zero-first",
    "csfs-split": "csfs-split",
    "split-active": "csfs-split",
    "rcsfsplit": "csfs-split",
    "interacting": "interacting",
    "restore-csfs": "restore-csfs",
}
_TEMPLATE_HEADER = """# rCSFs CLI configuration template.
# Uncomment the table and edit its values before using this command.
# Paths are relative to the directory where you run rcsfs.
# Command-line arguments override values in this file.
# conf = "e1_vv1_"  # Shared filename stem, before as{number}raw.

"""
_CONFIG_TEMPLATES = {
    "csfsgenerate": """
# [csfsgenerate]
# as = 6
# orbital_order = "*"
# inactive_core = 0
# reference_configuration = ["1s(2,*)"]
# active_space = "2s"
# j_min = 0
# j_max = 0
# excitations = 0
# rcsfs_out = "generated.c"
# generate_parquet = false
# generate_descriptors = false
# rcsfs_parquet = "generated.parquet"
# descriptor = "generated_descriptors.parquet"
# normalize = false
# threads = 8
# memory_budget_mib = 1024
# estimate_only = false
# allow_unchecked_space = false
# generation_storage = "disk"
# continue_lists = false
""".lstrip(),
    "gen-descriptors": """

# [gen-descriptors]
# input_parquet = "generated.parquet"
# output_parquet = "generated_descriptors.parquet"
# header = "generated_header.toml"
# num_workers = 8
# normalize = false
# descriptor_version = 2
# compression = "zstd"
# json = false
""".lstrip(),
    "zero-first": """

# [zero-first]
# zero_csf = "zero.c"
# full_csf = "generated.c"
# output_csf = "zero_first.c"
# keep_parquet = false
# work_dir = "."
# num_workers = 8
# max_line_len = 256
# json = false
""".lstrip(),
    "csfs-split": """

# [csfs-split]
# split_csfs_parquet = "generated.parquet"
# csfs_header = "generated_header.toml"
# active_spaces = ["AS1=2s", "AS2=3s"]
# output_dir = "split"
# json = false
""".lstrip(),
    "interacting": """

# [interacting]
# reference = "reference.c"
# candidates = "generated.c"
# output = "interacting.c"
# hamiltonian = "dc"
# method = "structural-upper-bound"
# num_workers = 8
# json = false
""".lstrip(),
    "restore-csfs": """

# [restore-csfs]
# descriptors = "generated_descriptors.parquet"
# header = "generated_header.toml"
# output = "restored.c"
# indices = [0, 1]
# json = false
""".lstrip(),
}


def create_default_config(
    command: str,
) -> Literal["created", "added", "exists"]:
    """Create or append one command's reference section without replacing data."""
    section = CONFIG_SECTIONS[command]
    template = _CONFIG_TEMPLATES[section]
    try:
        with DEFAULT_CONFIG.open("x", encoding="utf-8") as config_file:
            _ = config_file.write(_TEMPLATE_HEADER + template)
        return "created"
    except FileExistsError:
        if not DEFAULT_CONFIG.is_file():
            raise

    contents = DEFAULT_CONFIG.read_text(encoding="utf-8")
    try:
        root = tomllib.loads(contents)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid {DEFAULT_CONFIG}: {exc}") from exc
    active_sections = (
        {section, "split-active"} if section == "csfs-split" else {section}
    )
    if any(
        name in root or f"# [{name}]" in contents.splitlines()
        for name in active_sections
    ):
        return "exists"

    separator = "\n" if contents.endswith("\n") else "\n\n"
    with DEFAULT_CONFIG.open("a", encoding="utf-8") as config_file:
        _ = config_file.write(separator + template)
    return "added"


def configured_commands(path: Path) -> list[tuple[str, int | None]]:
    """Return active command tables in TOML order for a batch invocation."""
    try:
        root = _table(tomllib.loads(path.read_text(encoding="utf-8")), "config")
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    _ = _conf_prefix(root)
    commands: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for section in root:
        if section == "conf":
            continue
        if section == "output" and "generate" in root:
            continue
        command = (
            "csfsgenerate" if section == "generate" else CONFIG_SECTIONS.get(section)
        )
        if command is None:
            raise ValueError(f"unsupported configuration table: [{section}]")
        if command in seen:
            raise ValueError(f"configuration selects [{command}] more than once")
        seen.add(command)
        if section == "csfsgenerate" and isinstance(root[section], list):
            entries = cast(list[object], root[section])
            if not entries:
                raise ValueError("[[csfsgenerate]] must contain at least one entry")
            for index, item in enumerate(entries):
                _ = _table(item, f"csfsgenerate[{index + 1}]")
                commands.append((command, index))
        else:
            commands.append((command, None))
    if not commands:
        raise ValueError(f"{path} has no active command tables")
    return commands


_GENERATION_KEYS = frozenset(
    {
        "orbital_order",
        "inactive_core",
        "reference_configuration",
        "active_space",
        "j_min",
        "j_max",
        "excitations",
        "continue_lists",
    }
)
_REQUIRED_GENERATION_KEYS = _GENERATION_KEYS - {"orbital_order", "continue_lists"}
_LIST_KEYS = _REQUIRED_GENERATION_KEYS - {"inactive_core"}
_GENERATION_ALIASES = {
    "order": "orbital_order",
    "core": "inactive_core",
    "references": "reference_configuration",
    "active_orbitals": "active_space",
}
_GENERATOR_ALIASES = {
    **_GENERATION_ALIASES,
    "output": "rcsfs_out",
    "parquet": "rcsfs_parquet",
    "descriptor_parquet": "descriptor",
}
_SPLIT_ALIASES = {
    "input_parquet": "split_csfs_parquet",
    "header": "csfs_header",
    "space": "active_spaces",
}


class _CommandAction(Protocol):
    choices: Mapping[str, argparse.ArgumentParser]


def _selected_parser(
    parser: argparse.ArgumentParser, command: str
) -> argparse.ArgumentParser:
    for action in parser._actions:
        if action.dest == "command":
            return cast(_CommandAction, cast(object, action)).choices[command]
    raise AssertionError("CLI has no subparsers")


def _sparse_arguments(
    command_parser: argparse.ArgumentParser,
) -> tuple[dict[str, object], set[str], dict[str, argparse.Action]]:
    """Collect old defaults and requirements, then parse only supplied values."""
    defaults: dict[str, object] = {}
    required: set[str] = set()
    actions: dict[str, argparse.Action] = {}
    for action in command_parser._actions:
        if action.dest == "help":
            continue
        actions[action.dest] = action
        default_value = cast(object, action.default)
        if default_value is not argparse.SUPPRESS:
            defaults[action.dest] = default_value
        if action.required or (not action.option_strings and action.nargs is None):
            required.add(action.dest)
        if action.required:
            action.required = False
        if not action.option_strings and action.nargs is None:
            action.nargs = "?"
        action.default = argparse.SUPPRESS
    return defaults, required, actions


def _value(key: str, value: object, action: argparse.Action) -> object:
    if action.nargs == 0 and cast(object, action.const) is True:
        if type(value) is not bool:
            raise ValueError(f"{key} must be a boolean")
        return value
    if key == "active_spaces" or action.nargs == "+":
        if not isinstance(value, list) or not value:
            raise ValueError(f"{key} must be a nonempty array")
        return [_scalar(key, item, action) for item in cast(list[object], value)]
    return _scalar(key, value, action)


def _scalar(key: str, value: object, action: argparse.Action) -> object:
    if action.type is Path:
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a path string")
        parsed: object = Path(value)
    elif action.type is int or action.type is not None:
        integer_parser = action.type is int or key in {
            "threads",
            "num_workers",
            "memory_budget_mib",
        }
        if integer_parser and type(value) is not int:
            raise ValueError(f"{key} must be an integer")
        if not integer_parser and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        try:
            parser = cast(Callable[[str], object], action.type)
            parsed = parser(str(value))
        except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
            raise ValueError(f"invalid {key}: {exc}") from exc
    else:
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        parsed = value
    if action.choices is not None and parsed not in action.choices:
        raise ValueError(
            f"invalid {key}: {parsed!r}; expected one of {list(action.choices)}"
        )
    return parsed


def _generation(table: dict[str, object]) -> dict[str, object]:
    unknown = table.keys() - _GENERATION_KEYS
    if unknown:
        raise ValueError(f"unknown generation keys: {', '.join(sorted(unknown))}")
    missing = _REQUIRED_GENERATION_KEYS - table.keys()
    if missing:
        raise ValueError(f"missing generation keys: {', '.join(sorted(missing))}")
    result: dict[str, object] = {}
    for key, value in table.items():
        if key == "reference_configuration":
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in cast(list[object], value)
            ):
                raise ValueError("reference_configuration must be an array of strings")
        elif key in {"inactive_core", "j_min", "j_max", "excitations"}:
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        elif key == "continue_lists":
            if type(value) is not bool or value:
                raise ValueError("continue_lists is not supported yet; use false")
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        result[key] = value
    return result


def _generation_lists(table: dict[str, object]) -> dict[str, object]:
    """Keep each continuation list self-contained; only the core is shared."""
    if "inactive_core" not in table:
        raise ValueError("csfsgenerate requires inactive_core for multiple lists")
    lists = table.get("lists")
    if not isinstance(lists, list):
        raise ValueError("lists must contain at least two tables")
    items = cast(list[object], lists)
    if len(items) < 2:
        raise ValueError("lists must contain at least two tables")
    forbidden = (table.keys() & _GENERATION_KEYS) - {"inactive_core", "orbital_order"}
    if forbidden:
        raise ValueError(
            "list-specific keys must be inside [[csfsgenerate.lists]]: "
            + ", ".join(sorted(forbidden))
        )
    result: list[dict[str, object]] = []
    for index, raw in enumerate(items, 1):
        item = _canonicalize(
            _table(raw, f"csfsgenerate.lists[{index}]"),
            _GENERATION_ALIASES,
            f"csfsgenerate.lists[{index}]",
        )
        unexpected = item.keys() - _LIST_KEYS
        if unexpected:
            raise ValueError(
                f"csfsgenerate.lists[{index}] has unsupported keys: "
                + ", ".join(sorted(unexpected))
            )
        merged = {
            "inactive_core": table["inactive_core"],
            "orbital_order": table.get("orbital_order", "*"),
            **item,
        }
        result.append(_generation(merged))
    return {"lists": result}


def _table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    return cast(dict[str, object], value)


def _conf_prefix(root: Mapping[str, object]) -> str | None:
    value = root.get("conf")
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError(
            "top-level conf must be a nonempty filename stem (letters, digits, _ or -)"
        )
    return value


def _as_level(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("as must be a non-negative integer")
    return value


def _split_source_paths(
    root: Mapping[str, object], conf: str
) -> tuple[Path, Path] | None:
    generation = root.get("csfsgenerate")
    if isinstance(generation, list):
        entries = cast(list[object], generation)
        generation = entries[-1] if entries else None
    if not isinstance(generation, dict):
        return None
    values = _canonicalize(
        cast(dict[str, object], generation), _GENERATOR_ALIASES, "csfsgenerate"
    )
    raw_output = values.get("rcsfs_out")
    if raw_output is not None:
        if not isinstance(raw_output, str):
            raise ValueError("rcsfs_out must be a path string")
        csf = Path(raw_output)
    elif "as" in values:
        csf = Path(f"{conf}as{_as_level(values['as'])}raw.c")
    else:
        return None
    raw_parquet = values.get("rcsfs_parquet")
    if raw_parquet is not None and not isinstance(raw_parquet, str):
        raise ValueError("rcsfs_parquet must be a path string")
    parquet = (
        Path(raw_parquet)
        if isinstance(raw_parquet, str)
        else csf.with_suffix(".parquet")
    )
    return parquet, parquet.parent / f"{csf.stem}_header.toml"


def _canonicalize(
    values: dict[str, object], aliases: Mapping[str, str], label: str
) -> dict[str, object]:
    """Translate compatibility spellings once, before command validation."""
    result: dict[str, object] = {}
    for key, value in values.items():
        canonical = aliases.get(key, key)
        if canonical in result:
            raise ValueError(f"[{label}] sets {canonical} more than once")
        result[canonical] = value
    return result


def _load_config(
    path: Path,
    command: str,
    actions: dict[str, argparse.Action],
    *,
    explicit: bool,
    config_index: int | None = None,
) -> tuple[dict[str, object], dict[str, object] | None, bool]:
    try:
        root = _table(tomllib.loads(path.read_text(encoding="utf-8")), "config")
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    conf = _conf_prefix(root)
    section = CONFIG_SECTIONS[command]
    as_level: int | None = None
    is_legacy = section == "csfsgenerate" and "generate" in root
    if is_legacy:
        if section in root:
            raise ValueError("cannot combine [generate] with [csfsgenerate]")
        output = _canonicalize(
            _table(root.get("output", {}), "output"),
            {"csf": "rcsfs_out", **_GENERATOR_ALIASES},
            "output",
        )
        unknown = output.keys() - {
            "rcsfs_out",
            "generate_parquet",
            "generate_descriptors",
            "rcsfs_parquet",
            "descriptor",
            "normalize",
        }
        if unknown:
            raise ValueError(f"unknown output keys: {', '.join(sorted(unknown))}")
        values = output
        generate = _canonicalize(
            _table(root["generate"], "generate"),
            {**_GENERATION_ALIASES, "storage": "generation_storage"},
            "generate",
        )
        unknown_generate = (
            generate.keys()
            - _GENERATION_KEYS
            - {"memory_budget_mib", "generation_storage"}
        )
        if unknown_generate:
            raise ValueError(
                f"unknown generate keys: {', '.join(sorted(unknown_generate))}"
            )
        for key in ("memory_budget_mib", "generation_storage"):
            if key in generate:
                values[key] = generate[key]
        # The legacy generation table also contains resource settings.
        generation = _generation(
            {key: value for key, value in generate.items() if key in _GENERATION_KEYS}
        )
    else:
        if section == "csfs-split" and section in root and "split-active" in root:
            raise ValueError("cannot combine [csfs-split] with [split-active]")
        actual_section = (
            "split-active"
            if section == "csfs-split"
            and section not in root
            and "split-active" in root
            else section
        )
        if actual_section not in root:
            if explicit:
                raise ValueError(f"{path} has no [{section}] table")
            return {}, None, False
        raw_values: object = root[actual_section]
        if section == "csfsgenerate" and isinstance(raw_values, list):
            if config_index is None:
                raise ValueError(
                    "[[csfsgenerate]] requires the top-level 'rcsfs -c FILE' command"
                )
            items = cast(list[object], raw_values)
            if config_index >= len(items):
                raise ValueError(f"csfsgenerate entry {config_index + 1} is missing")
            raw_values = items[config_index]
        values = _table(raw_values, actual_section)
        generation = None
        if section == "csfsgenerate":
            values = _canonicalize(values, _GENERATOR_ALIASES, section)
            as_level = _as_level(values.pop("as")) if "as" in values else None
            generation = (
                _generation_lists(values)
                if "lists" in values
                else _generation(
                    {
                        key: value
                        for key, value in values.items()
                        if key in _GENERATION_KEYS
                    }
                )
            )
            values = {
                key: value
                for key, value in values.items()
                if key not in _GENERATION_KEYS and key != "lists"
            }
        elif section == "csfs-split":
            values = _canonicalize(values, _SPLIT_ALIASES, actual_section)
    converted: dict[str, object] = {}
    for key, value in values.items():
        if key == "config":
            raise ValueError("config cannot be set inside the TOML file")
        action = actions.get(key)
        if action is None:
            raise ValueError(f"unknown [{section}] key: {key}")
        converted[key] = _value(key, value, action)
    if section == "csfsgenerate" and as_level is not None:
        if conf is not None:
            _ = converted.setdefault("rcsfs_out", Path(f"{conf}as{as_level}raw.c"))
        elif "rcsfs_out" not in converted:
            raise ValueError("as requires top-level conf or an explicit rcsfs_out")
    elif section == "csfs-split" and conf is not None:
        source_paths = (
            _split_source_paths(root, conf)
            if "split_csfs_parquet" not in converted or "csfs_header" not in converted
            else None
        )
        if source_paths is not None:
            _ = converted.setdefault("split_csfs_parquet", source_paths[0])
            _ = converted.setdefault("csfs_header", source_paths[1])
        _ = converted.setdefault("output_dir", Path("split"))
        converted["conf"] = conf
    return converted, generation, True


def parse_cli_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    *,
    config_index: int | None = None,
) -> argparse.Namespace:
    tokens = list(argv) if argv is not None else sys.argv[1:]
    if not tokens or tokens[0] not in CONFIG_SECTIONS:
        return parser.parse_args(tokens)
    command = tokens[0]
    subparser = _selected_parser(parser, command)
    defaults, required, actions = _sparse_arguments(subparser)
    parsed = parser.parse_args(tokens)
    supplied = vars(parsed)
    selected_path = supplied.get("config")
    explicit = selected_path is not None
    if command != "csfsgenerate" and selected_path is None and DEFAULT_CONFIG.is_file():
        selected_path = DEFAULT_CONFIG
    config: dict[str, object] = {}
    generation: dict[str, object] | None = None
    applied = False
    if selected_path is not None:
        try:
            config, generation, applied = _load_config(
                selected_path,
                command,
                actions,
                explicit=explicit,
                config_index=config_index,
            )
        except ValueError as exc:
            parser.error(str(exc))
    merged = defaults | config | supplied
    missing = sorted(key for key in required if merged.get(key) is None)
    if missing:
        parser.error(f"{command} requires: {', '.join(missing)}")
    merged["config"] = selected_path if applied else None
    if command == "csfsgenerate":
        merged["generation"] = generation
    return argparse.Namespace(**merged)
