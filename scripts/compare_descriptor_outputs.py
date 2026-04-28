#!/usr/bin/env python3
"""Compare two descriptor parquet outputs.

Example:
    python scripts/compare_descriptor_outputs.py \
        --old /path/to/cv4odd1as3_odd4_1_desc.old.parquet \
        --new /path/to/cv4odd1as3_odd4_1_desc.parquet \
        --header /path/to/cv4odd1as3_odd4_1_header.toml \
        --raw /path/to/cv4odd1as3_odd4_1.parquet
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

pl: Any = None


FIELDS = ("electron_count", "middle_j", "coupling_j")


def read_peel_subshells(header_path: Path | None) -> list[str] | None:
    if header_path is None:
        return None

    with header_path.open("rb") as file:
        header = tomllib.load(file)

    header_lines = header.get("header_info", {}).get("header_lines")
    if not isinstance(header_lines, list) or len(header_lines) <= 3:
        raise ValueError(f"Could not find header_info.header_lines[3] in {header_path}")

    peel_line = header_lines[3]
    if not isinstance(peel_line, str):
        raise ValueError(f"header_info.header_lines[3] is not a string in {header_path}")

    return [
        part
        for part in peel_line.split()
        if any(ch.isalpha() for ch in part)
        and all(ch.isalnum() or ch in "+-_" for ch in part)
    ]


def max_abs_by_column(old: pl.DataFrame, new: pl.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in old.columns:
        max_diff = (old[column] - new[column]).abs().max()
        values[column] = float(max_diff or 0.0)
    return values


def range_report(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "min": frame.select(pl.min_horizontal(pl.all()).min()).item(),
        "max": frame.select(pl.max_horizontal(pl.all()).max()).item(),
        "cells_gt_1": frame.select(
            pl.sum_horizontal(
                [(pl.col(column) > 1.000001).cast(pl.Int64) for column in frame.columns]
            ).sum()
        ).item(),
        "cells_lt_0": frame.select(
            pl.sum_horizontal(
                [(pl.col(column) < -0.000001).cast(pl.Int64) for column in frame.columns]
            ).sum()
        ).item(),
    }


def describe_column(column: str, peel_subshells: list[str] | None) -> str:
    if not column.startswith("col_"):
        return column

    column_idx = int(column.removeprefix("col_"))
    orbital_idx = column_idx // 3
    field = FIELDS[column_idx % 3]

    if peel_subshells is None or orbital_idx >= len(peel_subshells):
        return f"{column} orbital_idx={orbital_idx} field={field}"

    return f"{column} subshell={peel_subshells[orbital_idx]} field={field}"


def differing_row_indices(
    old: pl.DataFrame,
    new: pl.DataFrame,
    changed_columns: list[str],
    tolerance: float,
) -> list[int]:
    if not changed_columns:
        return []

    mask = pl.Series("changed", [False] * old.height)
    for column in changed_columns:
        mask = mask | ((old[column] - new[column]).abs() > tolerance)

    return mask.arg_true().to_list()


def print_sample_rows(
    old: pl.DataFrame,
    new: pl.DataFrame,
    raw: pl.DataFrame | None,
    row_indices: list[int],
    changed_columns: list[str],
    sample_rows: int,
    sample_columns: int,
) -> None:
    if not row_indices:
        return

    columns = changed_columns[:sample_columns]
    print()
    print(f"First differing rows: {row_indices[:sample_rows]}")

    for row_idx in row_indices[:sample_rows]:
        print()
        print(f"row {row_idx}")
        print("old:")
        print(old[row_idx].select(columns))
        print("new:")
        print(new[row_idx].select(columns))
        if raw is not None:
            print("raw:")
            print(raw[row_idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path, help="Baseline descriptor parquet")
    parser.add_argument("--new", required=True, type=Path, help="New descriptor parquet")
    parser.add_argument("--header", type=Path, help="Optional generated header TOML")
    parser.add_argument("--raw", type=Path, help="Optional raw CSF parquet for row samples")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute tolerance")
    parser.add_argument("--sample-rows", type=int, default=10)
    parser.add_argument("--sample-cols", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    global pl

    args = parse_args()

    try:
        import polars as polars
    except ModuleNotFoundError as exc:
        if exc.name == "polars":
            print("This script requires polars: pip install polars", file=sys.stderr)
            return 2
        raise

    pl = polars

    old = pl.read_parquet(args.old)
    new = pl.read_parquet(args.new)
    raw = pl.read_parquet(args.raw) if args.raw else None
    peel_subshells = read_peel_subshells(args.header)

    print(f"old: {args.old}")
    print(f"new: {args.new}")
    print(f"shape: {old.shape} {new.shape}")
    print(f"schema equal: {old.schema == new.schema}")
    print(f"columns equal: {old.columns == new.columns}")

    if old.shape != new.shape or old.schema != new.schema or old.columns != new.columns:
        print("Descriptor outputs are not structurally comparable.")
        return 2

    print(f"old range: {range_report(old)}")
    print(f"new range: {range_report(new)}")

    exact_equal = old.equals(new)
    print(f"exact equal: {exact_equal}")

    column_max = max_abs_by_column(old, new)
    overall_max = max(column_max.values(), default=0.0)
    changed_columns = [
        column for column, max_diff in column_max.items() if max_diff > args.tol
    ]

    print(f"overall max abs diff: {overall_max}")
    print(f"values same: {overall_max <= args.tol}")
    print(f"changed columns: {len(changed_columns)}")

    if changed_columns:
        print()
        print("Changed columns:")
        for column in changed_columns:
            print(f"{describe_column(column, peel_subshells)} max_diff={column_max[column]}")

        row_indices = differing_row_indices(old, new, changed_columns, args.tol)
        print(f"different rows: {len(row_indices)}")
        print_sample_rows(
            old,
            new,
            raw,
            row_indices,
            changed_columns,
            args.sample_rows,
            args.sample_cols,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
