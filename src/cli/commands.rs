//! File commands call the same conversion, descriptor and selection algorithms as Python.
use super::{arguments::Settings, files};
use crate::parquet_info;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub(super) fn split_targets(
    settings: &Settings,
) -> Result<Vec<crate::descriptor_operations::ActiveSpaceTarget>> {
    let input = settings.path("split_csfs_parquet")?;
    let prefix = settings.text("prefix", files::stem(&input)?)?;
    if settings.conf.is_none() {
        files::safe_stem(prefix)?;
    }
    let raw = settings
        .values
        .get("active_spaces")
        .context("active_spaces / --space is required")?;
    let items: Vec<(String, String)> = if let Some(table) = raw.as_table() {
        table
            .iter()
            .map(|(k, v)| {
                Ok((
                    k.clone(),
                    v.as_str()
                        .context("active space must be a string")?
                        .to_owned(),
                ))
            })
            .collect::<Result<_>>()?
    } else {
        raw.as_array()
            .context("active_spaces must be a table or array")?
            .iter()
            .map(|v| {
                let (label, space) = v
                    .as_str()
                    .context("active space must be a string")?
                    .split_once('=')
                    .context("expected LABEL=5s,4p,3d")?;
                Ok((label.to_owned(), space.to_owned()))
            })
            .collect::<Result<_>>()?
    };
    ensure!(!items.is_empty(), "at least one active space is required");
    let mut targets = Vec::new();
    for (label, maximum_orbitals) in items {
        files::safe_stem(&label)?;
        let name = if let Some(conf) = &settings.conf {
            let upper = label.to_ascii_uppercase();
            let number = upper
                .strip_prefix("AS")
                .context("with conf, labels must be AS0, AS1, ...")?;
            ensure!(
                number == "0" || !number.starts_with('0'),
                "invalid AS label: {label}"
            );
            files::raw_name(conf, number.parse()?)
        } else {
            format!("{prefix}{label}_descriptors.parquet")
        };
        targets.push(crate::descriptor_operations::ActiveSpaceTarget {
            maximum_orbitals,
            output: PathBuf::from(name),
        });
    }
    files::validate_set(
        &targets.iter().map(|t| t.output.clone()).collect::<Vec<_>>(),
        &[input, settings.path("csfs_header")?],
        settings.config.as_deref(),
    )?;
    Ok(targets)
}

pub(super) fn run(command: &str, s: &Settings) -> Result<Value> {
    match command {
        "csfsgenerate" => super::generation::run(s),
        "convert" => {
            let input = s.path("input_csf")?;
            let output = s.output("output_parquet", None)?;
            let header = crate::descriptor_dataset::header_path(&output)?;
            files::validate_set(
                &[output.clone(), header.clone()],
                std::slice::from_ref(&input),
                s.config.as_deref(),
            )?;
            let stage = files::stage()?;
            let staged = stage
                .path()
                .join(output.file_name().context("missing output name")?);
            let staged_header = stage
                .path()
                .join(header.file_name().context("missing header name")?);
            let mut stats = serde_json::to_value(crate::descriptor_dataset::import_text(
                &input,
                &staged,
                &staged_header,
            )?)?;
            files::publish(
                &[staged, staged_header],
                &[output.clone(), header.clone()],
                true,
            )?;
            stats["output_file"] = json!(output);
            stats["header_file"] = json!(header);
            Ok(stats)
        }
        "info" => Ok(serde_json::to_value(
            parquet_info::read_parquet_info(&s.path("input_parquet")?)
                .map_err(anyhow::Error::from_boxed)?,
        )?),
        "restore-csfs" => {
            let input = s.path("descriptors")?;
            let header = s.path("header")?;
            let output = s.output("output", None)?;
            files::validate_set(
                std::slice::from_ref(&output),
                &[input.clone(), header.clone()],
                s.config.as_deref(),
            )?;
            let indices = s.values.get("indices").map(|v| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_integer().unwrap() as u64)
                    .collect::<Vec<_>>()
            });
            let stage = files::stage()?;
            let staged = stage.path().join("restored.c");
            let (count, bytes) = crate::descriptor_dataset::restore_text(
                &input,
                &header,
                &staged,
                indices.as_deref(),
            )?;
            files::publish(&[staged], std::slice::from_ref(&output), true)?;
            Ok(
                json!({"success": true, "output_file": output, "record_count": count, "output_bytes": bytes}),
            )
        }
        "zero-first" => {
            let zero = s.path("zero_csf")?;
            let full = s.path("full_csf")?;
            let output = s.output(
                "output_csf",
                Some(Path::new("zero_first_descriptors.parquet")),
            )?;
            let zh = crate::descriptor_dataset::header_path(&zero)?;
            let fh = crate::descriptor_dataset::header_path(&full)?;
            let header = crate::descriptor_dataset::header_path(&output)?;
            files::validate_set(
                &[output.clone(), header.clone()],
                &[zero.clone(), zh.clone(), full.clone(), fh.clone()],
                s.config.as_deref(),
            )?;
            let stage = files::stage()?;
            let staged = stage
                .path()
                .join(output.file_name().context("missing output name")?);
            let staged_header = stage
                .path()
                .join(header.file_name().context("missing header name")?);
            let stats = crate::descriptor_operations::zero_first(
                &zero,
                &zh,
                &full,
                &fh,
                &staged,
                &staged_header,
            )?;
            let mut result = serde_json::to_value(&stats)?;
            result["output_csf_count"] = json!(stats.dataset.record_count);
            files::publish(
                &[staged, staged_header],
                &[output.clone(), header.clone()],
                true,
            )?;
            result["output_file"] = json!(output);
            result["header_file"] = json!(header);
            Ok(result)
        }
        "csfs-split" => {
            let targets = split_targets(s)?;
            let input = s.path("split_csfs_parquet")?;
            let header = s.path("csfs_header")?;
            let stage = files::stage()?;
            let (mut sources, mut destinations, mut specs) = (Vec::new(), Vec::new(), Vec::new());
            for target in targets {
                let staged = stage
                    .path()
                    .join(target.output.file_name().context("missing output name")?);
                let staged_header = crate::descriptor_dataset::header_path(&staged)?;
                let output_header = crate::descriptor_dataset::header_path(&target.output)?;
                specs.push((
                    staged.clone(),
                    staged_header.clone(),
                    target.maximum_orbitals,
                ));
                sources.extend([staged, staged_header]);
                destinations.extend([target.output, output_header]);
            }
            files::validate_set(
                &destinations,
                &[input.clone(), header.clone()],
                s.config.as_deref(),
            )?;
            let results =
                crate::descriptor_operations::split_active_spaces(&input, &header, &specs)?;
            let results = results
                .into_iter()
                .enumerate()
                .map(|(index, stat)| {
                    let mut result = serde_json::to_value(stat)?;
                    result["output_file"] = json!(destinations[2 * index]);
                    result["header_file"] = json!(destinations[2 * index + 1]);
                    result["maximum_orbitals"] = json!(specs[index].2);
                    Ok(result)
                })
                .collect::<Result<Vec<_>>>()?;
            files::publish(&sources, &destinations, true)?;
            Ok(json!({"success": true, "outputs": results}))
        }
        "jsplit" => {
            let input = s.path("input_csf")?;
            let header = crate::descriptor_dataset::header_path(&input)?;
            let stem = files::stem(&input)?.trim_end_matches("_descriptors");
            let prefix = s.text("prefix", stem)?;
            let stage = files::stage()?;
            let stats =
                crate::descriptor_operations::split_blocks(&input, &header, stage.path(), prefix)?;
            let (mut sources, mut destinations, mut results) = (Vec::new(), Vec::new(), Vec::new());
            let wave = header.with_file_name(format!("{stem}.w"));
            for stat in stats {
                let source = PathBuf::from(&stat.output_file);
                let source_header = PathBuf::from(&stat.header_file);
                let output = PathBuf::from(source.file_name().context("missing split name")?);
                let output_header = crate::descriptor_dataset::header_path(&output)?;
                let mut result = serde_json::to_value(stat)?;
                result["output_file"] = json!(output);
                result["header_file"] = json!(output_header);
                sources.extend([source, source_header]);
                destinations.extend([output.clone(), output_header]);
                if s.flag("copy_w", true)? && wave.exists() {
                    let wave_output = output.with_file_name(format!(
                        "{}.w",
                        files::stem(&output)?.trim_end_matches("_descriptors")
                    ));
                    sources.push(wave.clone());
                    destinations.push(wave_output.clone());
                    result["w_file"] = json!(wave_output);
                }
                results.push(result);
            }
            files::validate_set(&destinations, &[input, header, wave], s.config.as_deref())?;
            files::publish(&sources, &destinations, s.flag("overwrite", false)?)?;
            Ok(json!({"success": true, "outputs": results}))
        }
        "interacting" => {
            use crate::{
                complete_csf::Parity,
                interaction::{self, HamiltonianMode, InteractionMethod},
            };
            let reference = s.path("reference")?;
            let candidates = s.path("candidates")?;
            let output = s.output("output", Some(Path::new("interacting_descriptors.parquet")))?;
            crate::reporting::warning!(
                "STRUCTURAL UPPER BOUND ONLY: this is not an exact rcsfinteract90 angular-algebra calculation; false positives may be retained."
            );
            let stats = interaction::select_interacting_csfs(
                &reference,
                &candidates,
                &output,
                HamiltonianMode::parse(
                    match s.text("hamiltonian", "dc")?.to_ascii_lowercase().as_str() {
                        "dc" | "dirac-coulomb" | "dirac_coulomb" => "dirac_coulomb",
                        "dcb" | "dirac-coulomb-breit" | "dirac_coulomb_breit" => {
                            "dirac_coulomb_breit"
                        }
                        other => anyhow::bail!("unknown hamiltonian: {other}"),
                    },
                )?,
                InteractionMethod::parse(
                    &s.text("method", "structural-upper-bound")?
                        .replace('-', "_"),
                )?,
                s.positive("num_workers")?.or(Some(8)),
                s.flag("overwrite", true)?,
            )?;
            let blocks: Vec<_> = stats.blocks.iter().map(|b| json!({"block_index": b.block_index, "total_two_j": b.total_two_j, "parity": if b.parity == Parity::Odd { "-" } else { "+" }, "reference_count": b.reference_count, "candidate_count": b.candidate_count, "exact_reference_skipped": b.exact_reference_skipped, "selected_count": b.selected_count, "rejected_count": b.rejected_count, "output_count": b.output_count})).collect();
            Ok(
                json!({"success": true, "exact": stats.exact, "hamiltonian": stats.mode.as_str(), "method": stats.method.as_str(), "reference_file": reference, "candidate_file": candidates, "output_file": output, "block_count": stats.block_count, "reference_count": stats.reference_count, "candidate_count": stats.candidate_count, "exact_reference_skipped": stats.exact_reference_skipped, "selected_count": stats.selected_count, "rejected_count": stats.rejected_count, "output_count": stats.output_count, "output_bytes": stats.output_bytes, "blocks": blocks}),
            )
        }
        _ => anyhow::bail!("unsupported command: {command}"),
    }
}
