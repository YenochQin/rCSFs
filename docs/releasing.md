# Preparing an rcsfs release

The current source version is **1.3.2**, read from `Cargo.toml` by Maturin.
The Python 3.15 migration is unreleased; choose a new version before publishing
changed artifacts because the 1.3.2 release already exists. The in-tree fallback
in `rcsfs/__init__.py` must match it when the version changes.
The local build commands do not upload files or create Git tags. The separate
manual publishing workflow is described near the end of this guide.

## Local build in the paired workspace

Use this repository's `.venv` (Python 3.15), set up with
`uv sync --python 3.15 --no-editable --group lint`. A Rust toolchain and
platform linker are required. From `rCSFs/`, choose an empty output directory:

```bash
source .venv/bin/activate
uvx --python .venv/bin/python --from 'maturin[patchelf]>=1.14,<2.0' maturin build --release --locked --sdist --compatibility pypi --interpreter .venv/bin/python --out dist
uvx --python .venv/bin/python --from 'twine>=6.2' twine check --strict dist/*
python scripts/check_release.py dist --smoke
```

`--sdist` builds the wheel from the source archive, checking that it is
self-contained. `--locked` uses the versioned Cargo.lock; update the lockfile
intentionally with dependency changes. Maturin and patchelf live in an isolated
tool environment. The Windows interpreter path is
`.venv/Scripts/python.exe`; initialize the compiler/linker
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
| Linux x86_64 | manylinux_2_28 (glibc >= 2.28) | CPython 3.15 |
| Windows x86_64 | win_amd64 | CPython 3.15 |
| macOS Apple Silicon | macosx arm64 | CPython 3.15 |

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
the console entry point. Locally it uses runtime dependencies in this repository's environment;
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

The collector itself is offline. Some crates omit their license texts from the
published crate archives; their notices are preserved under `licenses/upstream/`
with exact upstream commit URLs. Update these overrides when those crates change.
Review newly introduced license terms and any additional bundled native libraries.

Before a separately authorized publication:

1. Confirm ownership/availability of the `rcsfs` PyPI name and that 1.3.2 is
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

## Manual publishing to PyPI and GitHub

`.github/workflows/publish.yml` is named **Publish packages** in GitHub Actions.
It runs only through **Run workflow**, from `main`. Pushing commits, tags, or
completing a build does not start a publication. Its inputs are:

| Input | Default | Purpose |
| --- | --- | --- |
| `version` | Required | Exact source version, currently `1.3.2`. |
| `publish_pypi` | Off | Upload the validated packages to PyPI. |
| `github_release` | Off | Create a GitHub Release using the same packages. |
| `draft` | On | Keep the GitHub Release as a draft; turn off only for formal publication. |

Open **Actions → Publish packages → Run workflow**, select `main`, enter the
source version, and select the destinations you intend to use:

| PyPI | GitHub Release | Result |
| --- | --- | --- |
| Off | Off | Build and validate only; no publication. |
| On | Off | Build, validate and upload to PyPI. |
| Off | On | Build, validate and create a GitHub Release (draft by default). |
| On | On | Build and validate once, then upload the same files to both destinations. |

A prior dry run is optional. No separate workflow, run ID or tag input is
needed. The GitHub tag is derived as `v{version}`, for example `v1.3.2`.
The `draft` option has no effect when `github_release` is off.

### One build, two independent destinations

Preflight checks the requested version against the source before the existing
Linux/Windows/macOS build matrix runs. The assembler checks package names,
filename/metadata versions and Python/platform tags across the full matrix.
It retains one Linux source archive, plus one universal wheel for GraspKit or
three native wheels for rCSFs. The canonical files are stored in the
**pypi-distributions** Actions artifact with SHA-256 hashes in the job summary.

Both upload jobs depend directly on the successful `prepare` job. They each
consume that same artifact and do not rebuild. Neither upload job depends on
the other: a PyPI failure does not block GitHub Release creation, and vice versa.
If an upload fails before creating a Release or accepting files, retrying that
failed job reuses the run's existing validated artifact. Resolve any existing
file/Release conflict before retrying; successful uploads are not overwritten.
The workflow has no rollback across services; inspect each job's result.
The concurrency group serializes runs of this publishing workflow.

A dry run (both destination options off) does not exercise PyPI authorization.
PyPI uploads only work for filenames that have not already been used. For a
new release of modified package contents, update the source version and enter
that version in the workflow; do not try to overwrite an existing release.

### GitHub Release behavior

The Release job automatically reads this run's ID, attempt and source commit
from GitHub. It confirms that preflight, all three builds and preparation have
succeeded; it does not require the whole workflow or the PyPI job to finish.
It checks the downloaded artifact's GitHub SHA-256 digest, package metadata and
wheel platforms before creating a Release. These checks run with Python's
standard library and the GitHub CLI supplied by the hosted runner.

The Release contains the wheel(s), source archive and `SHA256SUMS`. Its notes
record the exact source commit, build-run URL, artifact ID and per-file hashes.
An existing tag must resolve to that commit; an existing Release for the same
tag, including a draft, causes a failure instead of being overwritten. Partial
drafts are retained for inspection. Review drafts in the repository's
**Releases** page. When a tag does not exist yet, GitHub may defer creating it
until the draft is published; the draft retains the exact target commit.

The default is a **draft**. Formal GitHub publication happens only when
`github_release` is on and `draft` is explicitly off. The Release job uses the
built-in `GITHUB_TOKEN` with `actions: read` and `contents: write`; it needs no
personal access token, PyPI credential or OIDC permission.

### PyPI authorization

Only the `publish_pypi` job uses the `pypi` environment and `id-token: write`.
It downloads the prepared artifact and invokes `pypa/gh-action-pypi-publish`.
The workflow filename is still `publish.yml`; changing its display name to
**Publish packages** does not change the Trusted Publisher identity.

1. In this GitHub repository, open **Settings → Environments** and ensure an
   environment named exactly `pypi` exists. No PyPI API token is needed.
2. For an existing PyPI project you own, open its **Publishing** settings and
   add or verify the GitHub Trusted Publisher. For a new project, use
   [account Publishing](https://pypi.org/manage/account/publishing/) to add a
   pending publisher instead:

   | PyPI field | Value |
   | --- | --- |
   | PyPI Project Name (pending publisher only) | `rcsfs` |
   | Owner | `YenochQin` |
   | Repository name | `rCSFs` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   Enter only the workflow filename, without `.github/workflows/`. A pending
   publisher does not reserve the project name; the first successful use
   creates the project. Authorization for another PyPI project does not
   automatically authorize this one.
3. To rehearse, leave **publish_pypi** and **github_release** unchecked. To upload
   to PyPI when authorized, enable **publish_pypi**. Enable **github_release**
   too if you also want the same files on GitHub, leaving **draft** checked
   unless a formal Release is intended. This workflow has no TestPyPI target.

See the [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/).
