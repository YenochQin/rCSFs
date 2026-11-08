//! Native executable adapter. Python is not involved in parsing or execution.
mod arguments;
mod commands;
mod config;
mod csfg;
mod files;
mod generation;
mod templates;

use anyhow::{Context, Result, ensure};
use arguments::Settings;
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};
use toml::Table;

fn print_result(command: &str, result: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(result)?);
    } else {
        println!("{command}: completed");
        if let Some(object) = result.as_object() {
            for (key, value) in object {
                if key == "success" {
                    continue;
                }
                if value.is_string() {
                    println!("{key}: {}", value.as_str().unwrap());
                } else {
                    println!("{key}: {value}");
                }
            }
        }
    }
    Ok(())
}
fn run_job(job: &config::Job) -> Result<Value> {
    commands::run(&job.command, &job.settings)
}
fn run_batch(path: &Path, root: &Table) -> Result<()> {
    if root.contains_key("csfggenerate") {
        let overrides = Settings {
            values: Table::new(),
            config: Some(path.to_path_buf()),
            conf: None,
        };
        return print_result("csfggenerate", &csfg::run(root, path, &overrides)?, false);
    }
    let jobs = config::batch(root, path)?;
    let conf = root.get("conf").and_then(toml::Value::as_str);
    let log_path = PathBuf::from(
        conf.map(|c| format!("rcsfs_{c}.log"))
            .unwrap_or_else(|| "rcsfs.log".into()),
    );
    ensure!(
        !log_path.is_symlink(),
        "run log must not be a symbolic link"
    );
    files::distinct(&log_path, path)?;
    for job in &jobs {
        for output in config::output_paths(job)? {
            files::distinct(&log_path, &output)?;
        }
        for (key, value) in &job.settings.values {
            if [
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
            ]
            .contains(&key.as_str())
                && let Some(value) = value.as_str()
            {
                files::distinct(&log_path, Path::new(value))?;
            }
        }
    }
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    eprintln!("Run log: {}", log_path.display());
    writeln!(
        log,
        "{}",
        json!({"event": "start", "version": env!("CARGO_PKG_VERSION"), "implementation": "native Rust", "config": path, "config_contents": fs::read_to_string(path)?})
    )?;
    log.flush()?;
    for (index, job) in jobs.iter().enumerate() {
        eprintln!("Running [{} #{}]...", job.command, index + 1);
        let start = Instant::now();
        let result = run_job(job);
        let record = match &result {
            Ok(stats) => {
                json!({"event": "result", "command": job.command, "step": index + 1, "elapsed_seconds": start.elapsed().as_secs_f64(), "stats": stats})
            }
            Err(error) => {
                json!({"event": "error", "command": job.command, "step": index + 1, "error": format!("{error:#}")})
            }
        };
        writeln!(log, "{record}")?;
        log.flush()?;
        let stats = result?;
        print_result(&job.command, &stats, job.settings.flag("json", false)?)?;
    }
    writeln!(
        log,
        "{}",
        json!({"event": "complete", "success": true, "steps": jobs.len()})
    )?;
    log.flush()?;
    Ok(())
}

/// Parse process arguments and execute a native command. Errors never start Python.
pub fn main() -> ExitCode {
    let tokens: Vec<_> = std::env::args_os().collect();
    if tokens.len() >= 3 && tokens[2] == "init-config" {
        let command = arguments::canonical(tokens[1].to_str().unwrap_or(""));
        if arguments::COMMANDS.iter().any(|(name, _)| *name == command) {
            if tokens.len() == 4 && (tokens[3] == "--help" || tokens[3] == "-h") {
                println!("Append a commented [{command}] template to rcsfs.toml.");
                return ExitCode::SUCCESS;
            }
            if tokens.len() != 3 {
                eprintln!("error: init-config takes no arguments");
                return ExitCode::from(2);
            }
            return match config::init(command) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("error: {error:#}");
                    ExitCode::FAILURE
                }
            };
        }
    }
    let matches = match arguments::parser().try_get_matches_from(tokens) {
        Ok(matches) => matches,
        Err(error) => {
            let code = error.exit_code();
            let _ = error.print();
            return ExitCode::from(code as u8);
        }
    };
    let mut json_output = false;
    let preparation = (|| -> Result<Option<(String, Settings, Table)>> {
        let batch = matches.get_one::<String>("config");
        if let Some((command, sub)) = matches.subcommand() {
            ensure!(
                batch.is_none(),
                "top-level --config cannot be combined with a subcommand"
            );
            let command = arguments::canonical(command);
            let overrides = arguments::overrides(command, sub)?;
            json_output = overrides
                .get("json")
                .and_then(toml::Value::as_bool)
                .unwrap_or(false);
            let explicit = sub.get_one::<String>("config").map(PathBuf::from);
            let path = explicit.clone().or_else(|| {
                (command == "csfggenerate"
                    || command != "csfsgenerate" && Path::new("rcsfs.toml").is_file())
                .then(|| PathBuf::from("rcsfs.toml"))
            });
            let root = path
                .as_ref()
                .map(|p| config::read(p))
                .transpose()?
                .unwrap_or_default();
            let settings = if command == "csfggenerate" {
                let s = Settings {
                    values: overrides,
                    config: path,
                    conf: None,
                };
                s.validate(command)?;
                s
            } else {
                config::settings(
                    &root,
                    command,
                    path.as_deref(),
                    None,
                    explicit.is_some(),
                    overrides,
                )?
            };
            json_output = settings.flag("json", false)?;
            Ok(Some((command.to_owned(), settings, root)))
        } else {
            let path = Path::new(batch.context("a subcommand or --config is required")?);
            let root = config::read(path)?;
            // Validate here so malformed batches have the argument-error exit code.
            if !root.contains_key("csfggenerate") {
                config::batch(&root, path)?;
            }
            Ok(None)
        }
    })();
    let prepared = match preparation {
        Ok(value) => value,
        Err(error) => {
            report_error(&error, json_output);
            return ExitCode::from(2);
        }
    };
    let execution = (|| -> Result<()> {
        if let Some((command, settings, root)) = prepared {
            let result = if command == "csfggenerate" {
                csfg::run(
                    &root,
                    settings
                        .config
                        .as_deref()
                        .context("CSFG requires a config")?,
                    &settings,
                )?
            } else {
                run_job(&config::Job {
                    command: command.clone(),
                    settings,
                })?
            };
            print_result(&command, &result, json_output)
        } else {
            let path = Path::new(matches.get_one::<String>("config").unwrap());
            run_batch(path, &config::read(path)?)
        }
    })();
    match execution {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            report_error(&error, json_output);
            ExitCode::FAILURE
        }
    }
}
fn report_error(error: &anyhow::Error, json_output: bool) {
    eprintln!("error: {error:#}");
    if json_output {
        println!(
            "{}",
            json!({"success": false, "error": format!("{error:#}")})
        );
    }
}
