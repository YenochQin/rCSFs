# Code Review: Structural CSF Interaction Selection

**Review date:** 2026-09-16  
**Changes reviewed:** 2026-09-15 commits  
**Head:** `b1a57e32c567c97a48460c1baa2f63b8dbc749e7`  
**Base:** `a6dbcae1cab8daf6fe32345d4fe1fcd8e3f52a8a`  
**Comparison:** `git diff a6dbcae...HEAD`  
**Commits:**

- `c9ca96b` — `interaction: add structural CSF selection`
- `b1a57e3` — `docs: standardize shared Python environment`

**Status:** Seven open findings. No implementation changes were made during the
review.

The main change adds a conservative, explicitly non-exact structural interaction
selector. It exposes a Rust implementation through the Python API and a new
`rcsfs interacting` CLI command. The selector compares candidates with reference
CSFs inside aligned symmetry blocks, applies occupation-difference limits,
preserves input order, skips exact reference records, and publishes output through
a same-directory temporary file.

No external issue or PRD was referenced by the commits. The specification axis
therefore used the public API documentation, README, changelog, type stubs, and
`docs/RCSFINTERACT90_CALCULATION_FLOW.md` as the available behavioral contract.
Line references below refer to the reviewed `HEAD`.

## Standards findings

### 1. P2 — The documented shared-environment build command is unavailable

**Location:** `AGENTS.md:25–35`; `CLAUDE.md:19–33`;
`../graspkit-tools/pyproject.toml:174–190`.

The updated repository instructions require contributors to create the only
supported Python environment with `uv sync` in `graspkit-tools`, activate
`../graspkit-tools/.venv`, and then run:

```bash
maturin develop
```

However, the shared environment's default `dev` dependency group does not include
Maturin. Maturin appears only in `rCSFs`' own development dependency group, while
the same instructions prohibit synchronizing or using a repository-local rCSFs
environment.

The reviewed shared environment had no `maturin` executable, so the documented
rebuild command failed. This also prevented the normal Python test workflow from
refreshing the installed native extension after the Rust changes.

**Recommended fix:** Add the supported Maturin version to the shared
`graspkit-tools` development dependency group and resynchronize its lockfile. If
Maturin is intentionally build-isolated instead, document a command that works
from the shared environment without creating another virtual environment.

### 2. P3 — The new PyO3 binding is not thin

**Location:** `src/lib.rs:51–149`; repository rule in `AGENTS.md:82`.

The repository requires PyO3 bindings in `src/lib.rs` to remain thin and pushes
heavy logic into focused Rust modules. The new binding performs several distinct
jobs:

- validates and decodes the worker, Hamiltonian, and method options;
- performs an output-existence preflight check;
- classifies `anyhow` error chains into Python exception classes;
- serializes global statistics into a Python dictionary;
- serializes every per-block result into nested dictionaries and a list.

This leaves the actual selection algorithm in `interaction.rs`, but the boundary
adapter itself is approximately one hundred lines and will grow whenever result
fields or interaction modes change.

**Recommended fix:** Move option parsing, error conversion, and statistics
serialization into focused helpers or conversion implementations. Keep
`src/lib.rs` responsible for registration, Python argument receipt, and one thin
call into the interaction API.

### 3. P3 — The interaction module combines unrelated responsibilities

**Location:** `src/interaction.rs:138–722`; repository rule in `AGENTS.md:82`.

The 748-line module contains the domain selection policy and all of the following
infrastructure:

- reference/candidate block validation;
- exact-CSF and occupation representations;
- filesystem alias and hard-link detection;
- temporary-file naming and cleanup;
- atomic create-if-absent publication;
- Unix replacement behavior;
- a Windows-specific `MoveFileExW` binding.

This is a possible Divergent Change smell: selection physics and platform output
publication evolve for different reasons but currently require edits in the same
module.

**Recommended fix:** Extract the reusable alias checking and atomic-output
publication code, beginning around `ensure_output_does_not_alias_input`, into a
focused filesystem module. Keep `interaction.rs` centered on CSF validation,
selection decisions, and statistics.

### 4. P3 — Interaction option literals are duplicated in the CLI

**Location:** `rcsfs/_types.py:5–9`; `rcsfs/cli.py:71–109`.

The change introduces the public type aliases `InteractionHamiltonian` and
`InteractionMethod`, but the CLI repeats their literal definitions in its
protocol, parser return annotations, and alias dictionary. This creates a small
drift risk when a new Hamiltonian or method is added.

**Recommended fix:** Import the domain aliases from `_types.py` for CLI
annotations and keep the accepted-value mapping in one place. CLI-only spellings
such as `dc` and `dcb` can remain aliases that normalize into those shared types.

## Specification and behavior findings

### 5. P2 — Negative `num_workers` violates the documented exception contract

**Location:** `rcsfs/__init__.py:440–450`; `src/lib.rs:51–63`.

The public wrapper documents `num_workers` as an optional positive value and says
that a non-positive value raises `ValueError`. The native binding accepts
`Option<usize>` and checks only `Some(0)`:

```rust
num_workers: Option<usize>,

if matches!(num_workers, Some(0)) {
    return Err(PyValueError::new_err("num_workers must be greater than 0"));
}
```

For `num_workers=-1`, PyO3 fails while extracting the unsigned integer, before
the function body runs. The observed exception was:

```text
OverflowError: can't convert negative int to unsigned
```

This differs from the promised `ValueError` and bypasses the function's explicit
validation message. Extremely large Python integers can fail at the same boundary
before the Rust check or normal thread-pool error mapping.

**Recommended fix:** Validate the Python value before unsigned conversion, for
example by accepting a signed integer at the binding boundary and rejecting
values less than or equal to zero before converting to `usize`. Add direct Python
API tests for `0`, `-1`, and an out-of-range positive integer.

### 6. P2 — The orbital-basis contract is contradictory

**Location:** `docs/RCSFINTERACT90_CALCULATION_FLOW.md:41–47`;
`rcsfs/__init__.py:432–435`; `src/interaction.rs:392–403`.

The calculation-flow document says the original interaction workflow requires
identical peel subshell lists and ordering in both inputs. The public Python
wrapper describes the candidate file as using the same orbital basis. The Rust
implementation instead accepts the reference peel list when it is only a prefix
of the candidate peel list:

```rust
reference.subshells.len() <= candidates.subshells.len()
    && reference.subshells == candidates.subshells[..reference.subshells.len()]
```

Prefix extension may be an intentional and useful relaxation: reference records
can be copied into an output using the candidate's expanded peel header, and the
extra candidate shells are included in occupation comparisons. The problem is
that this behavior is currently clear only in the Rust implementation comments,
while the user-facing sources imply the stricter original contract.

**Recommended fix:** Decide which contract is public. If exact compatibility with
the documented input rules is required, enforce equality. If prefix extension is
intentional, document it explicitly in the README and Python API, including that
the candidate header becomes the output header, and add a public API regression
test for the relaxed case.

### 7. P3 — The distinct-input restriction is not documented publicly

**Location:** `src/interaction.rs:175–204`; `rcsfs/__init__.py:428–450`.

The implementation rejects reference and candidate inputs that resolve to the
same file, including alternate paths, symlinks, and Unix hard links. The public
API documentation describes the two paths but does not state that they must be
distinct.

Using one file for both roles has a well-defined result under the advertised
algorithm: every reference is written, and matching candidate records are
skipped. Rejecting it may still be a deliberate guardrail, especially because the
changelog advertises alias protection, but callers should not discover the
restriction only through a runtime error.

**Recommended fix:** Document the distinct-file and alias restrictions in the
Python docstring and README. If a reference-only identity selection is considered
valid, remove the input-to-input restriction while retaining output-to-input
alias protection.

## Validation

| Check | Result |
| --- | --- |
| `cargo test` with the shared Python 3.14 runtime selected | 152 passed |
| Python `pytest tests` against a freshly built native module | 53 passed |
| `ruff check rcsfs tests/cli_test.py tests/rcsfs_test.py` | Passed |
| `basedpyright rcsfs` | 0 errors, 0 warnings |
| `cargo fmt --check` | Passed |

The repository's installed `rcsfs._rcsfs` extension predated the reviewed Rust
change, so ordinary Python test collection initially failed with:

```text
ImportError: cannot import name 'select_interacting_csfs' from 'rcsfs._rcsfs'
```

Because the documented shared environment did not contain Maturin, the review
built the Rust library with Cargo and loaded a temporary copy of the fresh native
module from `/private/tmp` for Python validation. This test arrangement did not
modify tracked repository files. The passing tests show that the covered behavior
works with a fresh extension; they do not resolve the documented rebuild-workflow
problem in finding 1.

## Overall assessment

The structural selector is well covered and the tested implementation preserves
the advertised ordering, counting, duplicate-reference suppression, and atomic
publication behavior. The most important follow-up items are to make the sole
documented build environment actually capable of rebuilding the extension, align
`num_workers` errors with the Python contract, and make the intended orbital-list
compatibility rule explicit. The remaining findings are maintainability or
documentation issues rather than demonstrated selection-result corruption.
