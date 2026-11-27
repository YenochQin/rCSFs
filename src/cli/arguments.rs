//! One command schema for help, CLI parsing and TOML validation.
use anyhow::{Context, Result, bail, ensure};
use clap::parser::ValueSource;
use clap::{Arg, ArgAction, ArgMatches, Command};
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
use Kind::{Flag, Number, Numbers, Spaces, Text};
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
        "Import CSF text to V3 descriptor Parquet and a bound header TOML.",
    ),
    ("info", "Inspect Parquet metadata."),
    (
        "csfsgenerate",
        "Generate CSFs interactively, from TOML, or from a transcript.",
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
        "Restore CSFs from V3 descriptors, verifying the source header.",
    ),
];
pub(super) fn fields(command: &str) -> Vec<Field> {
    let mut result = match command {
        "convert" => vec![
            field!("input_csf", None, Text, "Text CSF input to import."),
            field!(
                "output_parquet",
                None,
                Text,
                "Output descriptor Parquet; a matching header is also written."
            ),
        ],
        "info" => vec![field!("input_parquet", None, Text, "Parquet input.")],
        "csfsgenerate" => vec![
            field!(
                "output",
                Some("output"),
                Text,
                "Output descriptor Parquet (default: rcsf_desc.parquet)."
            ),
            field!(
                "transcript",
                Some("transcript"),
                Text,
                "Read an rcsfgenerate transcript instead of prompting."
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
                "descriptor_version",
                Some("descriptor-version"),
                Number,
                "Require descriptor format version 3."
            ),
            field!(
                "generation_storage",
                Some("generation-storage"),
                Text,
                "Generation backend: memory or disk."
            ),
        ],
        "zero-first" => vec![
            field!(
                "zero_csf",
                None,
                Text,
                "Zero-order reference descriptor Parquet."
            ),
            field!("full_csf", None, Text, "Full-basis descriptor Parquet."),
            field!(
                "output_csf",
                None,
                Text,
                "Output descriptor Parquet (default: zero_first_desc.parquet)."
            ),
        ],
        "csfs-split" => vec![
            field!(
                "split_csfs_parquet",
                None,
                Text,
                "Input descriptor Parquet."
            ),
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
            field!("input_csf", None, Text, "Input descriptor Parquet."),
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
            field!("reference", None, Text, "Reference descriptor Parquet."),
            field!("candidates", None, Text, "Candidate descriptor Parquet."),
            field!(
                "output",
                Some("output"),
                Text,
                "Output descriptor Parquet (default: interacting_desc.parquet)."
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
                "V3 descriptor Parquet."
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
            if name == "interacting" && f.key == "num_workers" {
                arg = arg.visible_alias("num-workers");
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
        let missing = || format!("missing command-line value for {}", f.key);
        let value = match f.kind {
            Flag => Value::Boolean(f.key != "copy_w"),
            Number => number(matches.get_one::<String>(f.key).with_context(missing)?)?,
            Spaces => Value::Array(
                matches
                    .get_many::<String>(f.key)
                    .with_context(missing)?
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
            Numbers => Value::Array(
                matches
                    .get_many::<String>(f.key)
                    .with_context(missing)?
                    .map(|s| number(s))
                    .collect::<Result<_>>()?,
            ),
            Text => Value::String(
                matches
                    .get_one::<String>(f.key)
                    .with_context(missing)?
                    .clone(),
            ),
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
                    if let Some(table) = value.as_table() {
                        ensure!(!table.is_empty(), "{key} must not be empty");
                        for (label, space) in table {
                            ensure!(
                                !label.trim().is_empty()
                                    && space.as_str().is_some_and(|s| !s.trim().is_empty()),
                                "{key} labels and spaces must be nonempty strings"
                            );
                        }
                    } else {
                        let array = value
                            .as_array()
                            .context("active_spaces must be an array or table")?;
                        ensure!(!array.is_empty(), "{key} must not be empty");
                        for item in array {
                            let (label, space) = item
                                .as_str()
                                .and_then(|s| s.split_once('='))
                                .context("active_spaces entries must be LABEL=SPACE strings")?;
                            ensure!(
                                !label.trim().is_empty() && !space.trim().is_empty(),
                                "{key} labels and spaces must be nonempty strings"
                            );
                        }
                    }
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
            ensure!(
                version == usize::from(crate::descriptor_transport::DescriptorVersion::V3.tag()),
                "only descriptor version 3 is supported"
            );
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
