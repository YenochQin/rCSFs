# Gen Descriptors CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `rcsfs gen-descriptors` for generating descriptor Parquet files from existing CSF Parquet files and a conversion header TOML.

**Architecture:** Implement a small Python CLI wrapper in `rcsfs/cli.py` using `argparse`. Register it as the `rcsfs` console script in `pyproject.toml`; delegate all data processing to the existing public Python API.

**Tech Stack:** Python 3.14, argparse, pytest, existing `rcsfs` Python API.

---

### Task 1: CLI Tests

**Files:**
- Create: `tests/cli_test.py`

- [ ] **Step 1: Write failing tests for `gen-descriptors`**

Create tests that call `rcsfs.cli.main()` directly with monkeypatched API
functions. Verify success output, optional flags, and failure exit code.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `uv run pytest tests/cli_test.py -q`

Expected: import or behavior failure because `rcsfs.cli` does not exist yet.

### Task 2: CLI Implementation

**Files:**
- Create: `rcsfs/cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Implement `rcsfs/cli.py`**

Add an `argparse` parser with one subcommand: `gen-descriptors`. The handler
must read peel subshells from `--header`, call
`generate_descriptors_from_parquet()`, dump stats JSON to stdout, and return 0
for `success: true` or 1 otherwise.

- [ ] **Step 2: Register the console script**

Add:

```toml
[project.scripts]
rcsfs = "rcsfs.cli:main"
```

- [ ] **Step 3: Run the focused tests and verify they pass**

Run: `uv run pytest tests/cli_test.py -q`

Expected: all tests pass.

### Task 3: Validation

**Files:**
- Existing package and tests

- [ ] **Step 1: Run focused Python API/CLI tests**

Run: `uv run pytest tests/cli_test.py tests/rcsfs_test.py -q`

Expected: all selected tests pass.

- [ ] **Step 2: Run lint on changed Python files**

Run: `uv run ruff check rcsfs/cli.py tests/cli_test.py`

Expected: no lint errors.
