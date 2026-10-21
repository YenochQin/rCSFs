# Preparing an rcsfs release

The prepared version is **1.3.1**, read from `Cargo.toml` by Maturin. The
in-tree fallback in `rcsfs/__init__.py` must match it when the version changes.
The local build commands do not upload files or create Git tags. The separate
manual publishing workflow is described near the end of this guide.

## Local build in the paired workspace

Use `../graspkit-tools/.venv` (Python 3.14). Set it up only from Tools with
`uv sync --extra cpu`, or `--extra gpu` on a CUDA host. A Rust toolchain and
platform linker are required. From `rCSFs/`, choose an empty output directory:

```bash
source ../graspkit-tools/.venv/bin/activate
uvx --python ../graspkit-tools/.venv/bin/python --from 'maturin[patchelf]>=1.14,<2.0' maturin build --release --locked --sdist --compatibility pypi --interpreter ../graspkit-tools/.venv/bin/python --out dist
python -m twine check --strict dist/*
python scripts/check_release.py dist --smoke
```

`--sdist` builds the wheel from the source archive, checking that it is
self-contained. `--locked` uses the versioned Cargo.lock; update the lockfile
intentionally with dependency changes. Maturin and patchelf live in an isolated
tool environment, not the shared runtime. The Windows interpreter path is
`../graspkit-tools/.venv/Scripts/python.exe`; initialize the compiler/linker
before building. These commands neither install the new wheel into Tools nor
replace the in-tree extension. See [development.md](https://github.com/YenochQin/rCSFs/blob/HEAD/docs/development.md) for those
separate workflows. Build tools must have access to their package registries.

For a local wheel that was already built, create the source archive separately
with `maturin sdist --out dist` through the same uvx tool environment. Before
publication, still rebuild a wheel from the source archive with `--sdist`.
Do not upload historical files from `target/wheels/` alongside a new build.

## Linux compatibility and platform matrix

`--compatibility pypi` rejects plain `linux_x86_64` wheels and audits/repairs
native dependencies. It does not guarantee compatibility with old glibc.
A wheel built on a new Linux host may need a newer glibc than cluster nodes.
Use the build-only workflow for the intended distribution baseline:

| Runner / architecture | Wheel target | Interpreter |
| --- | --- | --- |
| Linux x86_64 | manylinux_2_28 (glibc >= 2.28) | CPython 3.14 |
| Windows x86_64 | win_amd64 | CPython 3.14 |
| macOS Apple Silicon | macosx arm64 | CPython 3.14 |

The workflow builds from the source archive, installs the produced wheel and
its public dependencies on the runner, runs archive and functional checks,
and retains GitHub Actions artifacts. It runs by manual dispatch or by an
explicit push to a `codex/release-build-check-*` branch. It is also reusable
through `workflow_call`, so the publishing workflow builds the same commit.
The build workflow contains no PyPI upload job, credentials or OIDC permission.
The matrix is a validation target, not a claim that unrun platforms passed.
Intel macOS, Linux ARM, musl, free-threaded Python and future Python versions
are outside this initial wheel matrix. Without a matching wheel, pip may build
from source and require Rust; successful source builds on those targets have
not been established by this preparation.

Until the workflow reaches the default branch, start a check from the release
commit by pushing it to a dedicated branch, for example:

```bash
git push origin HEAD:refs/heads/codex/release-build-check-20261020
```

Choose a fresh suffix for a separate run of an unchanged commit. Normal
development-branch pushes and tags do not trigger this workflow. Once the
workflow is on the default branch, the Actions page also offers **Run workflow**
with a branch selector. Review all three platform jobs before using the artifacts.

## Artifact checks

`python scripts/check_release.py dist --smoke` validates metadata, license
files, type stubs/markers, source archive allowlists and wheel tags. It prints
SHA-256 hashes and rejects mixed versions, plain Linux wheels, debug symbols
and multiple/stale native extensions. The smoke test unpacks a compatible
wheel into a temporary directory and uses a separate interpreter to check
import origins, Python/native versions, CSF conversion, V2 descriptors and
the console entry point. Locally it uses runtime dependencies already in Tools;
CI separately verifies dependency installation on fresh runners.

The source archive contains only build inputs, public readmes, license files,
this guide and the checker with its small CSF fixture. Full tests, benchmark
scripts/reports and research notes remain in the development repository.
Run the maintained Rust and Python regression suites there before tagging;
refresh the in-tree extension first if testing source Python wrappers.
The source archive is intended to rebuild the library, not run the entire
development benchmark suite.

## Licensing and publication checklist

The project's MIT copyright belongs to Yi (Yenoch) Qin. The previous Samuel
Colvin line was a template artifact, confirmed by the maintainer. GRASP's
MIT notice and attribution are bundled in `licenses/GRASP-MIT.txt`, including
in wheels. Preserve this notice for the GRASP-derived tables and behavior.
The locked Rust dependency notices are bundled in `licenses/RUST-DEPENDENCIES.txt`.
They include all resolved targets and build dependencies, with identical license
texts deduplicated by SHA-256. The wheel checker verifies the Cargo.lock fingerprint.
After changing dependencies, from the full development checkout run:

```bash
cargo fetch --locked
python scripts/collect_licenses.py
python scripts/collect_licenses.py --check
```

The collector itself is offline. Four crates omit their license texts from the
published crate archives; their notices are preserved under `licenses/upstream/`
with exact upstream commit URLs. Update these overrides when those crates change.
Review newly introduced license terms and any additional bundled native libraries.

Before a separately authorized publication:

1. Confirm ownership/availability of the `rcsfs` PyPI name and that 1.3.1 is
   unused. A missing project page does not establish name availability.
2. Review the exact release commit and all changes since the prior release.
   `docs/CHANGELOG.md` retains the development history; finalize its release
   date and version heading when a release is actually approved.
3. Run the build matrix on that commit and review archive contents/hashes.
   Public package archives disclose their included sources regardless of
   Git repository visibility. Check public documentation access separately.
4. Test TestPyPI installation only after upload authorization. Configure the
   production Trusted Publisher below before using the manual publishing job.
5. Tag the approved release; publish only artifacts built from that commit.
   Do not rebuild different code under an already published version.

References: [Maturin distribution](https://www.maturin.rs/distribution.html),
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).

## Manual PyPI publishing

`.github/workflows/publish.yml` is named **Publish to PyPI** in GitHub Actions.
It has only a `workflow_dispatch` trigger. Pushing commits, tags, or completing
an ordinary build does not start it. It accepts two inputs:

- `version`: the exact version in the source, currently `1.3.1`.
- `publish`: a checkbox, **false by default**. Leave it unchecked for a dry run.

Only `main` is accepted. Preflight checks the requested version before the
existing three-platform build matrix runs. The assembler validates every
platform's package name, filename/metadata version and Python/platform tags,
then retains one canonical Linux source archive. GraspKit keeps only the Linux
copy of its universal wheel; rCSFs keeps all three platform wheels. This avoids
silently overwriting duplicate filenames from different runners.

The prepared archives are stored in the **pypi-distributions** Actions artifact;
the job summary lists their SHA-256 hashes and source commit. Preparation and
build jobs have no OIDC permission. Only the final, separately gated `publish`
job uses the `pypi` environment and `id-token: write`. That job downloads the
prepared artifact and invokes `pypa/gh-action-pypi-publish`; it does not rebuild.
When `publish` is false, the upload job is skipped entirely. The workflow group
serializes publication attempts. It neither bumps versions nor creates tags.

### Configure authorization in the browser

1. In this GitHub repository, open **Settings → Environments → New environment**
   and create an environment named exactly **pypi**. No PyPI token or environment
   secret is needed. You can restrict its deployment branches to `main`.
2. Log in to [PyPI Publishing](https://pypi.org/manage/account/publishing/).
   Under **Add a new pending publisher**, choose **GitHub** and enter:

   | PyPI field | Value |
   | --- | --- |
   | PyPI Project Name | `rcsfs` |
   | Owner | `YenochQin` |
   | Repository name | `rCSFs` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   Enter only `publish.yml`, without `.github/workflows/`. The owner is the
   GitHub account name, not the PyPI username. Add the pending publisher. This
   does not upload files, create the project or reserve its name; the project
   is created on the first successful publication. If you already own an
   existing project with this name, add the publisher in that project's
   Publishing settings instead.
3. For a rehearsal, choose **Actions → Publish to PyPI → Run workflow**, select
   `main`, enter `1.3.1` and leave **publish unchecked**. All builds and
   package checks run, and the upload job is skipped. A dry run does not test
   the PyPI OIDC exchange because it never enters the publishing job.
4. Only when publication is authorized, run the same workflow from `main` with
   the exact version and **publish checked**. Then install that version from
   PyPI to verify it. TestPyPI is not a destination in this production workflow.

The pending-publisher setup follows the [PyPI guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## GitHub Release drafts from a successful rehearsal

Open **Actions → Draft GitHub Release → Run workflow** and select `main`.
Fill in `run_id` with the numeric ID from a successful **Publish to PyPI**
rehearsal URL (`/actions/runs/NUMBER`), and `tag` with `v1.3.1` (or the
version of that rehearsal). The selected run must have `publish` unchecked:
`preflight`, all three builds and `prepare` must pass, and `publish` must be
skipped. Runs from forks, other workflows or branches are rejected.

The workflow downloads that run's unexpired `pypi-distributions` artifact,
checks its GitHub SHA-256 digest, and verifies package names, metadata versions
and the expected wheel platforms. It reuses the exact archives without
installing dependencies or rebuilding. The script uses Python's standard
library and the GitHub CLI supplied by the hosted runner.

The resulting **draft** contains the wheel(s), source archive and `SHA256SUMS`.
Its notes record the source commit, run URL, artifact ID and per-file hashes.
The Release target is the rehearsal's commit, even if `main` has moved since.
An existing tag must resolve to that commit; an existing Release for the tag
(including a draft) causes a failure instead of overwriting it. Failed partial
drafts are retained for inspection. If an artifact has expired, run a new
rehearsal and use its run ID.

This workflow always creates a draft: it has no formal-publication switch and
no PyPI job or OIDC permission. It needs only the built-in `GITHUB_TOKEN` with
`actions: read` and `contents: write`, not a personal token or PyPI secret.
Review the draft in **Releases**; publishing it later requires a separate
manual action. If the tag does not exist yet, GitHub may defer creating it
until the draft is published; the draft retains the exact target commit.
