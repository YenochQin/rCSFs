//! Append full run records and library diagnostics without flooding the console.
use super::{config, csfg, files};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};
use toml::Table;

struct State {
    file: File,
    error: Option<String>,
    command: String,
    step: usize,
    legacy_table: bool,
}
static ACTIVE: Mutex<Option<State>> = Mutex::new(None);

pub(super) struct RunLog;

pub(super) fn now() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis())
}

impl RunLog {
    pub fn open(jobs: &[config::Job], path: Option<&Path>, root: &Table) -> Result<Self> {
        let conf = jobs
            .first()
            .and_then(|job| job.settings.conf.as_deref())
            .or_else(|| root.get("conf").and_then(toml::Value::as_str));
        if let Some(conf) = conf {
            files::safe_stem(conf)?;
        }
        let log_path = PathBuf::from(
            conf.map(|name| format!("rcsfs_{name}.log"))
                .unwrap_or_else(|| "rcsfs.log".into()),
        );
        ensure!(
            !log_path.is_symlink(),
            "run log must not be a symbolic link"
        );
        if let Some(path) = path {
            files::distinct(&log_path, path).context("run log conflicts with configuration")?;
        }
        for job in jobs {
            let outputs = if job.command == "csfggenerate" {
                csfg::output_paths(root)?
            } else {
                config::output_paths(job)?
            };
            // Check output aliases before creating the run log, just as batches do.
            files::validate_set(&outputs, &[], path)?;
            for output in outputs {
                files::distinct(&log_path, &output).context("run log conflicts with output")?;
            }
            for key in [
                "input_csf",
                "input_parquet",
                "zero_csf",
                "full_csf",
                "split_csfs_parquet",
                "csfs_header",
                "header",
                "descriptors",
                "reference",
                "candidates",
                "transcript",
            ] {
                if let Some(value) = job.settings.string(key)? {
                    files::distinct(&log_path, Path::new(value))
                        .context("run log conflicts with input")?;
                }
            }
            if job.command == "jsplit" {
                // J filenames are discovered while splitting. Protect any existing
                // CSF/orbital file aliases before appending to the log.
                for entry in fs::read_dir(".")? {
                    let candidate = entry?.path();
                    if candidate
                        .extension()
                        .is_some_and(|ext| ext == "c" || ext == "w")
                    {
                        files::distinct(&log_path, &candidate)
                            .context("run log conflicts with split file")?;
                    }
                }
                let orbitals = job.settings.path("input_csf")?.with_extension("w");
                files::distinct(&log_path, &orbitals).context("run log conflicts with orbitals")?;
            }
        }
        let config_contents = path.map(fs::read_to_string).transpose()?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .with_context(|| format!("cannot write run log {}", log_path.display()))?;
        *ACTIVE.lock().unwrap() = Some(State {
            file,
            error: None,
            command: String::new(),
            step: 0,
            legacy_table: false,
        });
        crate::reporting::set_sink(diagnostic);
        let log = Self;
        log.event(
            json!({"event": "start", "version": env!("CARGO_PKG_VERSION"),
            "implementation": "native Rust", "config": path, "config_contents": config_contents,
            "working_directory": std::env::current_dir()?, "started_at_unix_ms": now()?}),
        )?;
        eprintln!("Run log: {}", log_path.display());
        Ok(log)
    }

    pub fn step(&self, index: usize, job: &config::Job) -> Result<()> {
        {
            let mut active = ACTIVE.lock().unwrap();
            let state = active.as_mut().context("run log is not open")?;
            state.command.clone_from(&job.command);
            state.step = index;
            state.legacy_table = job.command == "zero-first"
                && !job.settings.flag("json", false)?
                && !std::io::stdout().is_terminal();
        }
        self.event(json!({"event": "step_start", "command": job.command,
            "step": index, "resolved_parameters": job.settings.values}))
    }

    pub fn event(&self, record: Value) -> Result<()> {
        let mut active = ACTIVE.lock().unwrap();
        let state = active.as_mut().context("run log is not open")?;
        ensure!(
            state.error.is_none(),
            "cannot write run log: {}",
            state.error.as_deref().unwrap_or("")
        );
        writeln!(state.file, "{record}").context("cannot write run log")?;
        state.file.flush().context("cannot flush run log")
    }
}

impl Drop for RunLog {
    fn drop(&mut self) {
        *ACTIVE.lock().unwrap() = None;
    }
}

fn diagnostic(kind: crate::reporting::Kind, message: fmt::Arguments<'_>) {
    use crate::reporting::Kind;
    let mut active = ACTIVE.lock().unwrap();
    if let Some(state) = active.as_mut() {
        if kind == Kind::Warning || kind == Kind::LegacyTable && state.legacy_table {
            eprintln!("{message}");
        }
        if state.error.is_some() {
            return;
        }
        let record = json!({"event": "diagnostic", "level": if kind == Kind::Warning { "warning" } else { "detail" },
            "command": state.command, "step": state.step, "message": message.to_string()});
        if let Err(error) = writeln!(state.file, "{record}").and_then(|()| state.file.flush()) {
            state.error = Some(error.to_string());
        }
    }
}
