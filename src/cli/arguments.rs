//! One command schema for help, CLI parsing and TOML validation.
use anyhow::{Context, Result, bail, ensure};
use clap::{Arg, ArgAction, ArgMatches, Command, parser::ValueSource};
use std::path::{Path, PathBuf};
use toml::{Table, Value};

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Text,
    Number,
    Flag,
    Numbers,
    Spaces,
}
#[derive(Clone, Copy)]
pub(super) struct Field {
    pub key: &'static str,
    pub long: Option<&'static str>,
    pub kind: Kind,
    pub help: &'static str,
}
use Kind::*;
macro_rules! field {
    ($key:literal, $long:expr, $kind:ident, $help:literal) => {
        Field {
            key: $key,
            long: $long,
            kind: $kind,
            help: $help,
        }
    };
}
pub(super) const COMMANDS: &[(&str, &str)] = &[
    (
        "convert",
        "Convert CSF text to ordered Parquet and a header TOML.",
    ),
    ("info", "Inspect Parquet metadata."),
    (
        "csfsgenerate",
        "Generate CSFs interactively, from TOML, or from a transcript.",
    ),
    (
        "csfggenerate",
        "Generate GRASPG-compatible CSFG and label files from TOML.",
    ),
    (
        "gen-descriptors",
        "Generate reversible V2 descriptor Parquet and metadata.",
    ),
    (
        "zero-first",
        "Partition each symmetry block into zero-order and first-order CSFs.",
    ),
    ("csfs-split", "Split CSF Parquet by active space."),
    (
        "jsplit",
        "Split a CSF list into J/parity blocks, optionally copying orbitals.",
    ),
    (
        "interacting",
        "Select a structural interaction upper bound (not exact angular algebra).",
    ),
    (
        "restore-csfs",
        "Restore CSFs from V2 descriptors, verifying the source header.",
    ),
];
pub(super) fn canonical(name: &str) -> &str {
    match name {
        "split-active" | "rcsfsplit" => "csfs-split",
        "split-j" | "rasfsplit" => "jsplit",
        _ => name,
    }
}
pub(super) fn fields(command: &str) -> Vec<Field> {
    let mut result = match command {
        "convert" => vec![
            field!("input_csf", None, Text, "Input CSF file."),
            field!("output_parquet", None, Text, "Output Parquet filename."),
            field!(
                "max_line_len",
                Some("max-line-len"),
                Number,
                "Maximum line length (default: 256)."
            ),
            field!(
                "chunk_size",
                Some("chunk-size"),
                Number,
                "Lines per batch (default: 3000000)."
            ),
            field!(
                "num_workers",
                Some("num-workers"),
                Number,
                "Worker threads (positive integer)."
            ),
        ],
        "info" => vec![field!("input_parquet", None, Text, "Parquet input.")],
        "csfsgenerate" => vec![
            field!(
                "rcsfs_out",
                None,
                Text,
                "Output CSF filename (default: rcsf.out)."
            ),
            field!(
                "transcript",
                Some("transcript"),
                Text,
                "Read an rcsfgenerate transcript instead of prompting."
            ),
            field!(
                "generate_parquet",
                Some("generate-parquet"),
                Flag,
                "Also publish CSF Parquet and header."
            ),
            field!(
                "generate_descriptors",
                Some("generate-descriptors"),
                Flag,
                "Also publish V2 descriptors and metadata."
            ),
            field!(
                "rcsfs_parquet",
                Some("rcsfs-parquet"),
                Text,
                "CSF Parquet filename."
            ),
            field!(
                "descriptor",
                Some("descriptor"),
                Text,
                "Descriptor Parquet filename."
            ),
            field!(
                "threads",
                Some("threads"),
                Number,
                "Worker threads (positive integer)."
            ),
            field!(
                "memory_budget_mib",
                Some("memory-budget-mib"),
                Number,
                "Managed memory budget for disk generation."
            ),
            field!(
                "estimate_only",
                Some("estimate-only"),
                Flag,
                "Report disk workload and capacity without generating files."
            ),
            field!(
                "allow_unchecked_space",
                Some("allow-unchecked-space"),
                Flag,
                "Allow a run if free disk space cannot be measured."
            ),
            field!(
                "generation_storage",
                Some("generation-storage"),
                Text,
                "Generation backend: memory or disk."
            ),
        ],
        "csfggenerate" => vec![
            field!("threads", Some("threads"), Number, "Worker threads."),
            field!(
                "overwrite",
                Some("overwrite"),
                Flag,
                "Replace existing CSFG outputs."
            ),
        ],
        "gen-descriptors" => vec![
            field!("input_parquet", None, Text, "Input CSF Parquet."),
            field!("output_parquet", None, Text, "Output descriptor filename."),
            field!(
                "header",
                Some("header"),
                Text,
                "Matching source header TOML (required)."
            ),
            field!(
                "num_workers",
                Some("num-workers"),
                Number,
                "Worker threads."
            ),
            field!(
                "descriptor_version",
                Some("descriptor-version"),
                Number,
                "Descriptor version (only 2)."
            ),
            field!(
                "compression",
                Some("compression"),
                Text,
                "Parquet compression, e.g. zstd-3 or snappy."
            ),
        ],
        "zero-first" => vec![
            field!("zero_csf", None, Text, "Zero-order reference CSF file."),
            field!("full_csf", None, Text, "Full CSF file."),
            field!(
                "output_csf",
                None,
                Text,
                "Output filename (default: FULL_zf.csf)."
            ),
            field!(
                "num_workers",
                Some("num-workers"),
                Number,
                "Worker threads."
            ),
            field!(
                "max_line_len",
                Some("max-line-len"),
                Number,
                "Maximum line length (default: 256)."
            ),
        ],
        "csfs-split" => vec![
            field!("split_csfs_parquet", None, Text, "Input CSF Parquet."),
            field!(
                "csfs_header",
                Some("header"),
                Text,
                "Matching source header TOML."
            ),
            field!(
                "active_spaces",
                Some("space"),
                Spaces,
                "LABEL=5s,4p,3d; repeat for each output."
            ),
            field!("prefix", Some("prefix"), Text, "Output filename prefix."),
        ],
        "jsplit" => vec![
            field!("input_csf", None, Text, "Input CSF file."),
            field!("prefix", Some("prefix"), Text, "Output filename prefix."),
            field!(
                "copy_w",
                Some("no-copy-w"),
                Flag,
                "Do not copy a sibling .w file."
            ),
            field!(
                "overwrite",
                Some("overwrite"),
                Flag,
                "Replace existing outputs (default)."
            ),
        ],
        "interacting" => vec![
            field!("reference", None, Text, "Reference CSF file."),
            field!("candidates", None, Text, "Candidate CSF file."),
            field!(
                "output",
                Some("output"),
                Text,
                "Output filename (default: rcsf.out)."
            ),
            field!(
                "hamiltonian",
                Some("hamiltonian"),
                Text,
                "dc or dcb (default: dc)."
            ),
            field!(
                "method",
                Some("method"),
                Text,
                "structural-upper-bound only."
            ),
            field!(
                "num_workers",
                Some("threads"),
                Number,
                "Worker threads (default: 8)."
            ),
            field!(
                "overwrite",
                Some("overwrite"),
                Flag,
                "Replace existing outputs (default)."
            ),
        ],
        "restore-csfs" => vec![
            field!(
                "descriptors",
                Some("descriptors"),
                Text,
                "V2 descriptor Parquet."
            ),
            field!(
                "header",
                Some("header"),
                Text,
                "Matching source header TOML."
            ),
            field!("output", Some("output"), Text, "Restored CSF filename."),
            field!(
                "indices",
                Some("indices"),
                Numbers,
                "Zero-based row indices, in requested order."
            ),
        ],
        _ => vec![],
    };
    result.push(field!(
        "json",
        Some("json"),
        Flag,
        "Write machine-readable JSON to stdout; progress goes to stderr."
    ));
    result
}

pub(super) fn parser() -> Command {
    let mut root = Command::new("rcsfs")
        .version(concat!(env!("CARGO_PKG_VERSION"), " (native Rust)"))
        .about("Native Rust CSF processing; no Python runtime required.")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Run active TOML tables in file order."),
        )
        .arg_required_else_help(true);
    for &(name, about) in COMMANDS {
        let mut sub = Command::new(name).about(about)
            .after_help("Use 'rcsfs COMMAND init-config' to append a commented example to rcsfs.toml. Output files are placed in the current directory.")
            .arg(Arg::new("config").short('c').long("config").num_args(0..=1).default_missing_value("rcsfs.toml").value_name("FILE"));
        if name == "csfs-split" {
            sub = sub.visible_aliases(["split-active", "rcsfsplit"]);
        }
        if name == "jsplit" {
            sub = sub.visible_aliases(["split-j", "rasfsplit"]);
        }
        for f in fields(name) {
            let mut arg = Arg::new(f.key).help(f.help);
            if let Some(long) = f.long {
                arg = arg.long(long);
            }
            arg = match f.kind {
                Flag => arg.action(ArgAction::SetTrue),
                Spaces => arg.action(ArgAction::Append),
                Numbers => arg.num_args(1..).action(ArgAction::Append),
                _ => arg.action(ArgAction::Set),
            };
            if f.key == "output" {
                arg = arg.short('o');
            }
            if f.key == "active_spaces" {
                arg = arg.visible_alias("active-space");
            }
            if name == "interacting" && f.key == "num_workers" {
                arg = arg.visible_alias("num-workers");
            }
            if name == "csfsgenerate" && f.key == "rcsfs_parquet" {
                arg = arg.visible_alias("parquet");
            }
            if name == "csfsgenerate" && f.key == "descriptor" {
                arg = arg.visible_alias("descriptor-parquet");
            }
            sub = sub.arg(arg);
        }
        root = root.subcommand(sub);
    }
    root
}

pub(super) fn overrides(command: &str, matches: &ArgMatches) -> Result<Table> {
    let mut table = Table::new();
    for f in fields(command) {
        if matches.value_source(f.key) != Some(ValueSource::CommandLine) {
            continue;
        }
        let number = |s: &str| -> Result<Value> {
            Ok(Value::Integer(s.parse().with_context(|| {
                format!("{} must be an integer", f.key)
            })?))
        };
        let value = match f.kind {
            Flag => Value::Boolean(f.key != "copy_w"),
            Number => number(matches.get_one::<String>(f.key).unwrap())?,
            Spaces => Value::Array(
                matches
                    .get_many::<String>(f.key)
                    .unwrap()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
            Numbers => Value::Array(
                matches
                    .get_many::<String>(f.key)
                    .unwrap()
                    .map(|s| number(s))
                    .collect::<Result<_>>()?,
            ),
            Text => Value::String(matches.get_one::<String>(f.key).unwrap().clone()),
        };
        table.insert(f.key.into(), value);
    }
    Ok(table)
}

#[derive(Clone, Debug)]
pub(super) struct Settings {
    pub values: Table,
    pub config: Option<PathBuf>,
    pub conf: Option<String>,
}
impl Settings {
    pub fn string(&self, key: &str) -> Result<Option<&str>> {
        self.values
            .get(key)
            .map(|v| {
                v.as_str()
                    .with_context(|| format!("{key} must be a string"))
            })
            .transpose()
    }
    pub fn required(&self, key: &str) -> Result<&str> {
        self.string(key)?
            .with_context(|| format!("missing required setting: {key}"))
    }
    pub fn text<'a>(&'a self, key: &str, default: &'a str) -> Result<&'a str> {
        Ok(self.string(key)?.unwrap_or(default))
    }
    pub fn path(&self, key: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(self.required(key)?))
    }
    pub fn flag(&self, key: &str, default: bool) -> Result<bool> {
        self.values
            .get(key)
            .map(|v| {
                v.as_bool()
                    .with_context(|| format!("{key} must be a boolean"))
            })
            .transpose()
            .map(|v| v.unwrap_or(default))
    }
    pub fn number(&self, key: &str) -> Result<Option<usize>> {
        self.values
            .get(key)
            .map(|v| {
                v.as_integer()
                    .and_then(|n| usize::try_from(n).ok())
                    .with_context(|| format!("{key} must be a nonnegative integer"))
            })
            .transpose()
    }
    pub fn positive(&self, key: &str) -> Result<Option<usize>> {
        let value = self.number(key)?;
        ensure!(value != Some(0), "{key} must be greater than zero");
        Ok(value)
    }
    pub fn output(&self, key: &str, default: Option<&Path>) -> Result<PathBuf> {
        let path = self
            .string(key)?
            .map(PathBuf::from)
            .or_else(|| default.map(Path::to_path_buf))
            .with_context(|| format!("missing required setting: {key}"))?;
        super::files::validate_output(&path, self.config.as_deref())?;
        Ok(path)
    }
    pub fn validate(&self, command: &str) -> Result<()> {
        let schema = fields(command);
        for (key, value) in &self.values {
            if command == "csfsgenerate" && super::generation::MODEL_KEYS.contains(&key.as_str()) {
                continue;
            }
            let field = schema
                .iter()
                .find(|f| f.key == key)
                .with_context(|| format!("unknown [{command}] key: {key}"))?;
            match field.kind {
                Text => {
                    self.required(key)?;
                }
                Number => {
                    self.number(key)?;
                }
                Flag => {
                    self.flag(key, false)?;
                }
                Spaces => {
                    ensure!(
                        value.is_table() && matches!(field.kind, Spaces)
                            || value
                                .as_array()
                                .is_some_and(|a| a.iter().all(Value::is_str)),
                        "{key} must be an array of strings or an active-space table"
                    );
                }
                Numbers => {
                    ensure!(
                        value.as_array().is_some_and(|a| !a.is_empty()
                            && a.iter().all(|v| v.as_integer().is_some_and(|n| n >= 0))),
                        "{key} must be a nonempty array of nonnegative integers"
                    );
                }
            }
        }
        for key in [
            "threads",
            "num_workers",
            "memory_budget_mib",
            "max_line_len",
            "chunk_size",
        ] {
            self.positive(key)?;
        }
        if let Some(version) = self.number("descriptor_version")? {
            ensure!(version == 2, "only descriptor version 2 is supported");
        }
        if let Some(storage) = self.string("generation_storage")? {
            ensure!(
                matches!(storage, "memory" | "disk"),
                "generation_storage must be memory or disk"
            );
        }
        if self.flag("continue_lists", false)? {
            bail!("continue_lists is unsupported; use [[csfsgenerate.lists]]");
        }
        Ok(())
    }
}
