//! Native executable adapter. Python is not involved in parsing or execution.
mod arguments;
mod commands;
mod config;
mod csfg;
mod files;
mod generation;
mod logging;
mod output;
mod templates;

use anyhow::{Context, Result, ensure};
use arguments::Settings;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};
use toml::Table;

fn print_result(command: &str, result: &Value, json_output: bool, elapsed: f64) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(result)?);
    } else {
        print!("{}", output::summary(command, result, elapsed));
    }
    Ok(())
}

fn run_logged(jobs: &[config::Job], path: Option<&Path>, root: &Table) -> Result<()> {
    let log = logging::RunLog::open(jobs, path, root)?;
    let run_start = Instant::now();
    let mut completed = 0;
    let outcome = (|| -> Result<()> {
        for (index, job) in jobs.iter().enumerate() {
            log.step(index + 1, job)?;
            if job.command == "csfggenerate" {
                log.event(json!({"event": "models", "configuration": root}))?;
            }
            let start = Instant::now();
            let result = if job.command == "csfggenerate" {
                csfg::run(root, path.context("CSFG requires a config")?, &job.settings)
            } else {
                commands::run(&job.command, &job.settings)
            };
            let elapsed = start.elapsed().as_secs_f64();
            let record = match &result {
                Ok(stats) => json!({"event": "result", "command": job.command,
                    "step": index + 1, "elapsed_seconds": elapsed, "stats": stats}),
                Err(error) => json!({"event": "error", "command": job.command,
                    "step": index + 1, "elapsed_seconds": elapsed, "error": format!("{error:#}")}),
            };
            log.event(record)?;
            let stats = result?;
            completed += 1;
            print_result(
                &job.command,
                &stats,
                job.settings.flag("json", false)?,
                elapsed,
            )?;
        }
        Ok(())
    })();
    log.event(json!({"event": "complete", "success": outcome.is_ok(),
        "exit_code": if outcome.is_ok() { 0 } else { 1 }, "steps": jobs.len(),
        "completed_steps": completed, "elapsed_seconds": run_start.elapsed().as_secs_f64(),
        "finished_at_unix_ms": logging::now()?}))?;
    outcome
}

fn run_batch(path: &Path, root: &Table) -> Result<()> {
    let jobs = if root.contains_key("csfggenerate") {
        vec![config::Job {
            command: "csfggenerate".into(),
            settings: Settings {
                values: Table::new(),
                config: Some(path.to_path_buf()),
                conf: None,
            },
        }]
    } else {
        config::batch(root, path)?
    };
    run_logged(&jobs, Some(path), root)
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
            let path = settings.config.clone();
            run_logged(&[config::Job { command, settings }], path.as_deref(), &root)
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
