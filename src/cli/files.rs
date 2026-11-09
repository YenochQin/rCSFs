//! Validate destinations and publish completed artifacts without truncating inputs.
use crate::atomic_output::{
    create_temporary_output, ensure_distinct_inputs, publish_temporary_output,
};
use anyhow::{Context, Result, ensure};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

pub(super) fn resolved(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.canonicalize()?);
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(parent
        .canonicalize()?
        .join(path.file_name().context("expected a filename")?))
}
pub(super) fn distinct(left: &Path, right: &Path) -> Result<()> {
    ensure!(
        resolved(left)? != resolved(right)?,
        "paths must differ: {} and {}",
        left.display(),
        right.display()
    );
    if left.exists() && right.exists() {
        ensure_distinct_inputs(left, right, "paths must not refer to the same file")?;
    }
    Ok(())
}
pub(super) fn validate_output(path: &Path, config: Option<&Path>) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    // Windows canonicalize() adds a verbatim prefix; normalize both operands.
    let cwd = std::env::current_dir()?.canonicalize()?;
    ensure!(
        parent.canonicalize().is_ok_and(|parent| parent == cwd),
        "output must name a file in the current directory: {}",
        path.display()
    );
    ensure!(
        path.file_name().is_some() && !path.is_dir(),
        "invalid output path: {}",
        path.display()
    );
    distinct(path, config.unwrap_or(Path::new("rcsfs.toml")))
}
pub(super) fn validate_set(
    outputs: &[PathBuf],
    inputs: &[PathBuf],
    config: Option<&Path>,
) -> Result<()> {
    for (index, output) in outputs.iter().enumerate() {
        validate_output(output, config)?;
        for other in outputs[..index].iter().chain(inputs) {
            distinct(output, other)?;
        }
    }
    Ok(())
}
pub(super) fn publish(
    sources: &[PathBuf],
    destinations: &[PathBuf],
    overwrite: bool,
) -> Result<()> {
    ensure!(
        sources.len() == destinations.len(),
        "publication source/destination count mismatch"
    );
    let mut published = Vec::new();
    for (source, destination) in sources.iter().zip(destinations) {
        let result = (|| -> Result<()> {
            // A target-side temporary file also handles cross-filesystem staging.
            let (guard, mut output) = create_temporary_output(destination)?;
            io::copy(&mut fs::File::open(source)?, &mut output)?;
            output.flush()?;
            output.sync_all()?;
            drop(output);
            publish_temporary_output(guard.path(), destination, overwrite)
        })();
        result.with_context(|| {
            format!(
                "could not publish {}; already published: {published:?}",
                destination.display()
            )
        })?;
        published.push(destination.display().to_string());
    }
    Ok(())
}
pub(super) fn stage() -> Result<tempfile::TempDir> {
    Ok(tempfile::Builder::new()
        .prefix("rcsfs-native-")
        .tempdir_in(".")?)
}
pub(super) fn stem(path: &Path) -> Result<&str> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .context("expected a UTF-8 filename stem")
}
pub(super) fn header(input: &Path, parquet: &Path) -> Result<PathBuf> {
    Ok(parquet.with_file_name(format!("{}_header.toml", stem(input)?)))
}
pub(super) fn safe_stem(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
        "filename stem must contain only letters, digits, _ or -"
    );
    Ok(())
}
pub(super) fn raw_name(conf: &str, level: usize) -> String {
    format!(
        "{conf}{}as{level}raw.c",
        if conf.ends_with('_') { "" } else { "_" }
    )
}
