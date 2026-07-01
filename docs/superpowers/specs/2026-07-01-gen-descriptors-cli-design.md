# Gen Descriptors CLI Design

## Scope

Add one installed command surface to the `rcsfs` package:

```bash
rcsfs gen-descriptors INPUT_PARQUET OUTPUT_PARQUET --header HEADER_TOML
```

The command generates descriptor Parquet data from an existing CSF Parquet file.
The user provides the `*_header.toml` file generated during CSF conversion, and
the CLI reads peel subshells from that header before calling the existing Python
API.

## Architecture

The CLI lives in `rcsfs/cli.py` and uses Python's standard `argparse` module.
`pyproject.toml` registers the console script:

```toml
[project.scripts]
rcsfs = "rcsfs.cli:main"
```

The CLI does not implement descriptor logic. It delegates to:

- `read_peel_subshells(header_path)`
- `generate_descriptors_from_parquet(input, output, peel_subshells, ...)`

## Command Behavior

`gen-descriptors` accepts:

- `input_parquet`: source CSF Parquet file
- `output_parquet`: destination descriptor Parquet file
- `--header PATH`: required header TOML file
- `--num-workers N`: optional worker count
- `--normalize`: optional normalized descriptor output

On success, the command writes the returned stats dictionary as formatted JSON to
stdout and exits with code 0. If the underlying API returns `success: false`, it
still writes the stats JSON and exits with code 1. Argument parsing errors use
argparse's standard exit code 2.

## Testing

Tests cover argument wiring without running the Rust descriptor engine by
monkeypatching `read_peel_subshells` and `generate_descriptors_from_parquet`.
They verify that `--header`, `--num-workers`, and `--normalize` are passed
correctly, JSON is printed, and `success: false` maps to exit code 1.
