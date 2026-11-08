//! File commands call the same conversion, descriptor and selection algorithms as Python.
use super::{arguments::Settings, files};
use crate::{
    csfs_conversion,
    csfs_descriptor::{self, parquet_batch},
    descriptor_schema::DescriptorVersion,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub(super) fn convert(input: &Path, output: &Path, settings: &Settings) -> Result<Value> {
    let stats = csfs_conversion::convert_csfs_to_parquet_parallel(
        input,
        output,
        settings.positive("max_line_len")?.unwrap_or(256),
        settings.positive("chunk_size")?.unwrap_or(3_000_000),
        settings.positive("num_workers")?,
    )
    .map_err(anyhow::Error::from_boxed)?;
    let mut result = serde_json::to_value(stats)?;
    result["success"] = true.into();
    result["input_file"] = json!(input);
    result["output_file"] = json!(output);
    result["header_file"] = json!(files::header(input, output)?);
    Ok(result)
}
pub(super) fn sidecar(path: &Path, count: usize, shells: &[String]) -> Result<()> {
    #[derive(serde::Serialize)]
    struct Metadata<'a> {
        format_version: u8,
        encoding: &'a str,
        normalized: bool,
        record_count: usize,
        subshells: &'a [String],
    }
    fs::write(
        path,
        toml::to_string(&Metadata {
            format_version: 2,
            encoding: "parquet",
            normalized: false,
            record_count: count,
            subshells: shells,
        })?,
    )?;
    Ok(())
}
pub(super) fn descriptors(
    input: &Path,
    header: &Path,
    output: &Path,
    workers: Option<usize>,
    compression: Option<&str>,
) -> Result<Value> {
    let shells = parquet_batch::read_peel_subshells_from_header(header)?;
    let stats = parquet_batch::generate_descriptors_from_parquet_parallel(
        input,
        output,
        shells.clone(),
        workers,
        DescriptorVersion::V2,
        Some(header),
        compression,
    )?;
    sidecar(
        &output.with_extension("toml"),
        stats.descriptor_count,
        &shells,
    )?;
    Ok(
        json!({"success": true, "input_file": input, "output_file": output,
        "csf_count": stats.csf_count, "descriptor_count": stats.descriptor_count,
        "orbital_count": stats.orbital_count, "descriptor_size": stats.descriptor_size,
        "descriptor_version": stats.descriptor_version, "channels_per_subshell": stats.channels_per_subshell}),
    )
}
pub(super) fn split_targets(
    settings: &Settings,
) -> Result<Vec<crate::csf_active_space_split::ActiveSpaceTarget>> {
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
            format!("{prefix}{label}.c")
        };
        targets.push(crate::csf_active_space_split::ActiveSpaceTarget {
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
            let header = files::header(&input, &output)?;
            files::validate_set(
                &[output.clone(), header.clone()],
                std::slice::from_ref(&input),
                s.config.as_deref(),
            )?;
            let stage = files::stage()?;
            let staged = stage.path().join("csfs.parquet");
            let mut stats = convert(&input, &staged, s)?;
            files::publish(
                &[staged.clone(), files::header(&input, &staged)?],
                &[output.clone(), header.clone()],
                true,
            )?;
            stats["output_file"] = json!(output);
            stats["header_file"] = json!(header);
            Ok(stats)
        }
        "info" => Ok(serde_json::to_value(
            csfs_conversion::read_parquet_info(&s.path("input_parquet")?)
                .map_err(anyhow::Error::from_boxed)?,
        )?),
        "gen-descriptors" => {
            let input = s.path("input_parquet")?;
            let header = s.path("header")?;
            let output = s.output("output_parquet", None)?;
            let metadata = output.with_extension("toml");
            files::validate_set(
                &[output.clone(), metadata.clone()],
                &[input.clone(), header.clone()],
                s.config.as_deref(),
            )?;
            let stage = files::stage()?;
            let staged = stage.path().join("descriptors.parquet");
            let mut stats = descriptors(
                &input,
                &header,
                &staged,
                s.positive("num_workers")?,
                s.string("compression")?,
            )?;
            files::publish(
                &[staged.clone(), staged.with_extension("toml")],
                &[output.clone(), metadata],
                true,
            )?;
            stats["output_file"] = json!(output);
            Ok(stats)
        }
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
            let (count, bytes) = csfs_descriptor::restore_csfs_from_descriptors(
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
                Some(Path::new(&format!("{}_zf.csf", files::stem(&full)?))),
            )?;
            files::validate_set(
                std::slice::from_ref(&output),
                &[zero.clone(), full.clone()],
                s.config.as_deref(),
            )?;
            let stage = files::stage()?;
            let zp = stage.path().join("zero/data.parquet");
            let fp = stage.path().join("full/data.parquet");
            fs::create_dir(zp.parent().unwrap())?;
            fs::create_dir(fp.parent().unwrap())?;
            convert(&zero, &zp, s)?;
            convert(&full, &fp, s)?;
            let staged = stage.path().join("result.c");
            let stats = crate::csf_partition::partition_csfs(
                &zp,
                &files::header(&zero, &zp)?,
                &fp,
                &files::header(&full, &fp)?,
                &staged,
            )
            .map_err(anyhow::Error::from_boxed)?;
            files::publish(&[staged], std::slice::from_ref(&output), true)?;
            let mut result = serde_json::to_value(stats)?;
            result["success"] = true.into();
            result["output_file"] = json!(output);
            Ok(result)
        }
        "csfs-split" => {
            let targets = split_targets(s)?;
            let stage = files::stage()?;
            let staged: Vec<_> = targets
                .iter()
                .map(|t| crate::csf_active_space_split::ActiveSpaceTarget {
                    output: stage.path().join(t.output.file_name().unwrap()),
                    maximum_orbitals: t.maximum_orbitals.clone(),
                })
                .collect();
            let mut stats = crate::csf_active_space_split::split_csfs_by_active_spaces(
                &s.path("split_csfs_parquet")?,
                &s.path("csfs_header")?,
                &staged,
            )?;
            files::publish(
                &staged.iter().map(|t| t.output.clone()).collect::<Vec<_>>(),
                &targets.iter().map(|t| t.output.clone()).collect::<Vec<_>>(),
                true,
            )?;
            for (stat, target) in stats.outputs.iter_mut().zip(&targets) {
                stat.output_file = target.output.display().to_string();
            }
            let mut result = serde_json::to_value(stats)?;
            result["success"] = true.into();
            Ok(result)
        }
        "jsplit" => {
            let input = s.path("input_csf")?;
            let prefix = s.text("prefix", files::stem(&input)?)?;
            // Stage all outputs to validate them against config and inputs before publication.
            let stage = files::stage()?;
            let mut stats = crate::csf_block_split::split_csfs_by_j(
                &input,
                stage.path(),
                prefix,
                s.flag("copy_w", true)?,
                false,
            )?;
            let mut sources = Vec::new();
            let mut destinations = Vec::new();
            for stat in &mut stats.outputs {
                for name in std::iter::once(&mut stat.output_file).chain(stat.w_file.iter_mut()) {
                    let src = PathBuf::from(&*name);
                    let dst = PathBuf::from(src.file_name().context("missing split filename")?);
                    sources.push(src);
                    destinations.push(dst.clone());
                    *name = dst.display().to_string();
                }
            }
            let mut inputs = vec![input.clone()];
            if input.with_extension("w").exists() {
                inputs.push(input.with_extension("w"));
            }
            files::validate_set(&destinations, &inputs, s.config.as_deref())?;
            files::publish(&sources, &destinations, s.flag("overwrite", true)?)?;
            let mut result = serde_json::to_value(stats)?;
            result["success"] = true.into();
            Ok(result)
        }
        "interacting" => {
            use crate::{
                complete_csf::Parity,
                interaction::{self, HamiltonianMode, InteractionMethod},
            };
            let reference = s.path("reference")?;
            let candidates = s.path("candidates")?;
            let output = s.output("output", Some(Path::new("rcsf.out")))?;
            eprintln!(
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
