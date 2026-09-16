//! Path-alias detection and atomic output publication.
//!
//! Writers that must not corrupt their own inputs share two concerns that are
//! independent of any particular file format: proving that an output path is
//! not another name for an input, and publishing a fully written temporary file
//! under its final name.  Both evolve with filesystem and platform behaviour
//! rather than with domain logic, so they live here instead of alongside the
//! algorithms that use them.

use anyhow::{Context, Result, ensure};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMPORARY_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A scratch file that is removed unless it is published.
///
/// Dropping the guard deletes the file, so an early return or a panic cannot
/// leave a partial output behind.
pub struct TemporaryOutput {
    path: PathBuf,
}

impl TemporaryOutput {
    /// Path of the temporary file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Reject an output path that is another name for `input`.
///
/// `input_description` names the role of `input` in the caller's error
/// messages, for example `"reference"`.
pub fn ensure_output_does_not_alias_input(
    output: &Path,
    input: &Path,
    input_description: &str,
) -> Result<()> {
    if output == input {
        anyhow::bail!(
            "output path must not be the {input_description} input path: {}",
            output.display()
        );
    }

    // If both names exist, canonicalization detects symlinks and metadata
    // identity detects hard links.  A missing output cannot alias an existing
    // input yet, so lexical equality above is sufficient for that case.
    if output.exists() {
        let output_canonical = output
            .canonicalize()
            .with_context(|| format!("failed to resolve output path {}", output.display()))?;
        let input_canonical = input.canonicalize().with_context(|| {
            format!(
                "failed to resolve {input_description} input {}",
                input.display()
            )
        })?;
        ensure!(
            output_canonical != input_canonical,
            "output path aliases the {input_description} input: {}",
            output.display()
        );
        ensure!(
            !same_file_identity(output, input)?,
            "output path is a hard link to the {input_description} input: {}",
            output.display()
        );
    }
    Ok(())
}

/// Reject two input paths that resolve to the same file.
///
/// `message` is used verbatim so callers can describe their own roles.
pub fn ensure_distinct_inputs(left: &Path, right: &Path, message: &str) -> Result<()> {
    ensure!(left != right, "{}", message);
    if left.exists() && right.exists() {
        let left_canonical = left
            .canonicalize()
            .with_context(|| format!("failed to resolve input {}", left.display()))?;
        let right_canonical = right
            .canonicalize()
            .with_context(|| format!("failed to resolve input {}", right.display()))?;
        ensure!(
            left_canonical != right_canonical && !same_file_identity(left, right)?,
            "{}",
            message
        );
    }
    Ok(())
}

#[cfg(unix)]
fn same_file_identity(left: &Path, right: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;

    let left = fs::metadata(left)?;
    let right = fs::metadata(right)?;
    Ok(left.dev() == right.dev() && left.ino() == right.ino())
}

#[cfg(not(unix))]
fn same_file_identity(_left: &Path, _right: &Path) -> Result<bool> {
    // Canonical paths catch ordinary names and symlinks on non-Unix targets.
    // Rust's standard library does not expose a portable file-ID comparison.
    Ok(false)
}

/// Create a uniquely named temporary file beside `output_path`.
///
/// Staging in the output's own directory keeps publication a same-filesystem
/// operation, which is what makes the rename and link steps atomic.
pub fn create_temporary_output(output_path: &Path) -> Result<(TemporaryOutput, File)> {
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("atomic-output");
    for _ in 0..128 {
        let sequence = TEMPORARY_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{name}.rcsfs-{}-{sequence}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((TemporaryOutput { path }, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create temporary output in {}", parent.display())
                });
            }
        }
    }
    anyhow::bail!(
        "failed to allocate a unique temporary output in {}",
        parent.display()
    )
}

/// Publish a completed temporary file as `output`.
///
/// With `overwrite == false` this refuses an existing output atomically, so a
/// file created after the caller's preflight check is still not clobbered.
pub fn publish_temporary_output(temporary: &Path, output: &Path, overwrite: bool) -> Result<()> {
    if overwrite {
        replace_output(temporary, output)?;
        return Ok(());
    }

    // A same-filesystem hard link is an atomic create-if-absent operation.
    // Unlike a preflight `exists` check, it also closes the publication race.
    fs::hard_link(temporary, output).with_context(|| {
        format!(
            "failed to publish output {} without overwriting an existing file",
            output.display()
        )
    })?;
    // Publication has succeeded. Cleanup is best-effort so callers never see
    // a failure for an output that is already complete and visible.
    let _ = fs::remove_file(temporary);
    Ok(())
}

#[cfg(not(windows))]
fn replace_output(temporary: &Path, output: &Path) -> Result<()> {
    fs::rename(temporary, output).with_context(|| {
        format!(
            "failed to replace output {} with completed temporary file",
            output.display()
        )
    })
}

#[cfg(windows)]
fn replace_output(temporary: &Path, output: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(
            existing_file_name: *const u16,
            new_file_name: *const u16,
            flags: u32,
        ) -> i32;
    }

    let temporary_wide = temporary
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let output_wide = output
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: both pointers refer to live, NUL-terminated UTF-16 buffers for
    // the duration of the call. The flags request same-filesystem replacement.
    let succeeded = unsafe {
        MoveFileExW(
            temporary_wide.as_ptr(),
            output_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if succeeded == 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "failed to replace output {} with completed temporary file",
                output.display()
            )
        });
    }
    Ok(())
}
