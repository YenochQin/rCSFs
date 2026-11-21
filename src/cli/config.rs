//! Merge CLI overrides and TOML once, then validate a complete batch before running it.
use super::{
    arguments::{self, Settings},
    commands, files, generation,
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    fs,
    path::{Path, PathBuf},
};
use toml::Table;

#[derive(Clone, Debug)]
pub(super) struct Job {
    pub command: String,
    pub settings: Settings,
}
pub(super) fn read(path: &Path) -> Result<Table> {
    toml::from_str(
        &fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?,
    )
    .with_context(|| format!("invalid TOML: {}", path.display()))
}
fn conf(root: &Table) -> Result<Option<String>> {
    root.get("conf")
        .map(|v| {
            let value = v.as_str().context("conf must be a string")?;
            files::safe_stem(value)?;
            Ok(value.to_owned())
        })
        .transpose()
}
fn model_values(
    root: &Table,
    command: &str,
    index: Option<usize>,
    explicit: bool,
) -> Result<Table> {
    let name = command;
    let Some(raw) = root.get(name) else {
        ensure!(!explicit, "config has no [{command}] table");
        return Ok(Table::new());
    };
    let raw = if let Some(array) = raw.as_array() {
        ensure!(command == "csfsgenerate", "[{command}] must be a table");
        &array[index.context("[[csfsgenerate]] requires top-level rcsfs -c FILE")?]
    } else {
        raw
    };
    Ok(raw
        .as_table()
        .with_context(|| format!("{command} must be a table"))?
        .clone())
}

pub(super) fn settings(
    root: &Table,
    command: &str,
    path: Option<&Path>,
    index: Option<usize>,
    explicit: bool,
    overrides: Table,
    allow_legacy_split_source: bool,
) -> Result<Settings> {
    let mut values = model_values(root, command, index, explicit)?;
    let conf = conf(root)?;
    if command == "csfsgenerate" {
        if let Some(conf) = &conf
            && !values.contains_key("as")
        {
            values
                .entry("output".to_string())
                .or_insert_with(|| format!("{conf}_desc.parquet").into());
        }
        if let Some(level) = values.remove("as") {
            let level = level
                .as_integer()
                .and_then(|n| usize::try_from(n).ok())
                .context("as must be a nonnegative integer")?;
            if let Some(conf) = &conf {
                values
                    .entry("output".to_string())
                    .or_insert_with(|| files::raw_name(conf, level).into());
            } else {
                ensure!(
                    values.contains_key("output") || overrides.contains_key("output"),
                    "as requires conf or output"
                );
            }
        }
    }
    if command == "csfs-split"
        && conf.is_some()
        && (!values.contains_key("split_csfs_parquet") || !values.contains_key("csfs_header"))
        && let Some(generation) = root.get("csfsgenerate")
    {
        let index = generation
            .as_array()
            .map(|a| a.len().checked_sub(1).context("empty csfsgenerate array"))
            .transpose()?;
        let source = settings(root, "csfsgenerate", path, index, true, Table::new(), false)?;
        let outputs = generation::Outputs::from_settings(&source)?;
        let mut descriptor = outputs.descriptor;
        if allow_legacy_split_source
            && !values.contains_key("split_csfs_parquet")
            && !overrides.contains_key("split_csfs_parquet")
            && !descriptor.exists()
            && let Some(stem) = files::stem(&descriptor)?.strip_suffix("_desc")
        {
            let legacy = descriptor.with_file_name(format!("{stem}_descriptors.parquet"));
            if legacy.is_file() {
                descriptor = legacy;
            }
        }
        values
            .entry("split_csfs_parquet".to_string())
            .or_insert_with(|| descriptor.display().to_string().into());
        values
            .entry("csfs_header".to_string())
            .or_insert_with(|| outputs.header.display().to_string().into());
    }
    values.extend(overrides);
    let settings = Settings {
        values,
        config: path.map(Path::to_path_buf),
        conf,
    };
    settings.validate(command)?;
    let required: &[&str] = match command {
        "convert" => &["input_csf", "output_parquet"],
        "info" => &["input_parquet"],
        "zero-first" => &["zero_csf", "full_csf"],
        "csfs-split" => &["split_csfs_parquet", "csfs_header"],
        "jsplit" => &["input_csf"],
        "interacting" => &["reference", "candidates"],
        "restore-csfs" => &["descriptors", "header", "output"],
        _ => &[],
    };
    for key in required {
        settings.required(key)?;
    }
    if command == "csfsgenerate" {
        let configured = generation::configured_transcripts(&settings)?;
        ensure!(
            !explicit || configured.is_some(),
            "generation config requires a complete model or transcript"
        );
    }
    if command == "csfs-split" {
        commands::split_targets(&settings)?;
    }
    Ok(settings)
}

pub(super) fn batch(root: &Table, path: &Path) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for (name, value) in root {
        if name == "conf" {
            continue;
        }
        let command = name.as_str();
        ensure!(
            arguments::COMMANDS.iter().any(|(c, _)| *c == command),
            "unsupported config section: {name}"
        );
        let count = if let Some(array) = value.as_array() {
            ensure!(
                command == "csfsgenerate" && !array.is_empty(),
                "only csfsgenerate accepts a nonempty array of tables"
            );
            array.len()
        } else {
            1
        };
        for i in 0..count {
            jobs.push(Job {
                command: command.into(),
                settings: settings(
                    root,
                    command,
                    Some(path),
                    value.is_array().then_some(i),
                    true,
                    Table::new(),
                    false,
                )?,
            });
        }
    }
    ensure!(!jobs.is_empty(), "config has no active command tables");
    // A split following a generator requests the intermediate Parquet automatically.
    for index in 0..jobs.len() {
        if jobs[index].command != "csfs-split" {
            continue;
        }
        let input = jobs[index].settings.path("split_csfs_parquet")?;
        let header = jobs[index].settings.path("csfs_header")?;
        let mut found = false;
        for earlier in jobs[..index].iter_mut().rev() {
            if earlier.command != "csfsgenerate" {
                continue;
            }
            let outputs = generation::Outputs::from_settings(&earlier.settings)?;
            if files::resolved(&input)? == files::resolved(&outputs.descriptor)?
                && files::resolved(&header)? == files::resolved(&outputs.header)?
            {
                ensure!(
                    !earlier.settings.flag("estimate_only", false)?,
                    "split cannot consume estimate-only generation"
                );
                found = true;
                break;
            }
        }
        ensure!(
            found || input.is_file() && header.is_file(),
            "split inputs must exist or match an earlier generation entry"
        );
    }
    let mut generated: Vec<PathBuf> = Vec::new();
    for job in &jobs {
        let outputs = output_paths(job)?;
        files::validate_set(&outputs, &[], Some(path))?;
        if job.command == "csfsgenerate" {
            for output in &outputs {
                for previous in &generated {
                    files::distinct(output, previous)?;
                }
            }
            generated.extend(outputs);
        }
    }
    Ok(jobs)
}
pub(super) fn output_paths(job: &Job) -> Result<Vec<PathBuf>> {
    let s = &job.settings;
    Ok(match job.command.as_str() {
        "csfsgenerate" => generation::Outputs::from_settings(s)?.paths(),
        "convert" => {
            let output = s.output("output_parquet", None)?;
            vec![
                output.clone(),
                crate::descriptor_dataset::header_path(&output)?,
            ]
        }
        "restore-csfs" => vec![s.output("output", None)?],
        "interacting" | "zero-first" => {
            let (key, default) = if job.command == "zero-first" {
                ("output_csf", "zero_first_desc.parquet")
            } else {
                ("output", "interacting_desc.parquet")
            };
            let output = s.output(key, Some(Path::new(default)))?;
            vec![
                output.clone(),
                crate::descriptor_dataset::header_path(&output)?,
            ]
        }
        "csfs-split" => {
            let mut outputs = Vec::new();
            for target in commands::split_targets(s)? {
                outputs.push(crate::descriptor_dataset::header_path(&target.output)?);
                outputs.push(target.output);
            }
            outputs
        }
        _ => vec![],
    })
}
pub(super) fn init(command: &str) -> Result<()> {
    let path = Path::new("rcsfs.toml");
    let contents = if path.exists() {
        let _ = read(path)?;
        fs::read_to_string(path)?
    } else {
        String::new()
    };
    let tag = format!("[{command}]");
    if contents
        .lines()
        .any(|line| line.trim().trim_start_matches('#').trim() == tag)
    {
        println!("{tag} template already exists");
        return Ok(());
    }
    let Some(template) = super::templates::template(command) else {
        bail!("no template for {command}");
    };
    let stage = files::stage()?;
    let source = stage.path().join("config.toml");
    fs::write(
        &source,
        format!(
            "{contents}\n# Paths are relative to the invocation directory. CLI flags override TOML.\n# conf = \"model\"\n{template}"
        ),
    )?;
    files::publish(&[source], &[path.to_path_buf()], true)?;
    println!(
        "{} rcsfs.toml",
        if contents.is_empty() {
            "Created"
        } else {
            "Updated"
        }
    );
    Ok(())
}
