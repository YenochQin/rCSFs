//! Generation orchestration; all CSF algorithms and capacity models remain shared.
use super::{arguments::Settings, commands, files};
use crate::configuration_generation::{
    self,
    csf_generation::{ArtifactKind, disk_csf_generation_pipeline as disk},
    shared_generation::GenerationOptions,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value as Json, json};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};
use toml::{Table, Value};

pub(super) const MODEL_KEYS: &[&str] = &[
    "orbital_order",
    "inactive_core",
    "reference_configuration",
    "active_space",
    "j_min",
    "j_max",
    "excitations",
    "continue_lists",
    "lists",
    "as",
];

pub(super) fn transcript(model: &Table) -> Result<String> {
    let string = |key: &str| -> Result<&str> {
        model
            .get(key)
            .and_then(Value::as_str)
            .with_context(|| format!("{key} must be a string"))
    };
    let integer = |key: &str| -> Result<i64> {
        model
            .get(key)
            .and_then(Value::as_integer)
            .with_context(|| format!("{key} must be an integer"))
    };
    let order = model
        .get("orbital_order")
        .map(|_| string("orbital_order"))
        .transpose()?
        .unwrap_or("*");
    ensure!(
        !order.contains(['\n', '\r']),
        "orbital_order must be one line"
    );
    let references = model
        .get("reference_configuration")
        .and_then(Value::as_array)
        .context("reference_configuration must be a nonempty array of strings")?;
    ensure!(
        !references.is_empty() && references.len() <= 100,
        "expected 1..=100 reference configurations"
    );
    let mut lines = vec![
        format!("{order} ! Orbital order"),
        integer("inactive_core")?.to_string(),
    ];
    for value in references {
        let reference = value
            .as_str()
            .context("reference_configuration must contain only strings")?;
        ensure!(
            !reference.trim().is_empty() && !reference.contains(['\n', '\r']),
            "each reference must be a nonempty single line"
        );
        lines.push(reference.into());
    }
    let active = string("active_space")?;
    ensure!(
        !active.contains(['\n', '\r']),
        "active_space must be one line"
    );
    lines.extend([
        String::new(),
        active.into(),
        format!("{},{}", integer("j_min")?, integer("j_max")?),
        integer("excitations")?.to_string(),
        "n".into(),
    ]);
    let text = lines.join("\n");
    // Fail at the configuration boundary, before opening outputs.
    configuration_generation::ExcitationRequest::from_transcript(&text)?;
    Ok(text)
}

fn prompt(message: &str) -> Result<String> {
    eprint!("{message}");
    io::stderr().flush()?;
    let mut line = String::new();
    ensure!(
        io::stdin().read_line(&mut line)? != 0,
        "unexpected end of input; use -c FILE or --transcript FILE for noninteractive generation"
    );
    Ok(line.trim().to_owned())
}
fn interactive() -> Result<Table> {
    let mut model = Table::new();
    model.insert(
        "orbital_order".into(),
        prompt("Default orbital ordering? (*) ")?.into(),
    );
    eprintln!("Core: 0=None, 1=He, 2=Ne, 3=Ar, 4=Kr, 5=Xe, 6=Rn");
    model.insert(
        "inactive_core".into(),
        Value::Integer(prompt("Core? (0-6) ")?.parse()?),
    );
    let mut references = Vec::new();
    loop {
        let line = prompt(&format!(
            "Give configuration {} (blank or * to finish): ",
            references.len() + 1
        ))?;
        if line.is_empty() || line == "*" {
            break;
        }
        references.push(Value::String(line));
        if references.len() == 100 {
            break;
        }
    }
    model.insert("reference_configuration".into(), Value::Array(references));
    model.insert(
        "active_space".into(),
        prompt("Active orbitals (e.g. 5s,4p,3d): ")?.into(),
    );
    let range = prompt("Resulting 2*J-number? lower,higher: ")?;
    let numbers = range
        .split([',', ' '])
        .filter(|s| !s.is_empty())
        .map(str::parse::<i64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(numbers.len() == 2, "expected two 2J integers");
    model.insert("j_min".into(), numbers[0].into());
    model.insert("j_max".into(), numbers[1].into());
    model.insert(
        "excitations".into(),
        Value::Integer(prompt("Number of excitations: ")?.parse()?),
    );
    ensure!(
        prompt("Generate more lists? (y/n) ")?.eq_ignore_ascii_case("n"),
        "interactive continuation is unsupported; use [[csfsgenerate.lists]] in TOML"
    );
    transcript(&model)?;
    Ok(model)
}

pub(super) fn configured_transcripts(s: &Settings) -> Result<Option<Vec<String>>> {
    if let Some(path) = s.string("transcript")? {
        ensure!(
            !s.values.contains_key("inactive_core") && !s.values.contains_key("lists"),
            "--transcript cannot be combined with a generation model"
        );
        let text = fs::read_to_string(path)?;
        configuration_generation::ExcitationRequest::from_transcript(&text)?;
        return Ok(Some(vec![text]));
    }
    if let Some(raw) = s.values.get("lists") {
        let lists = raw.as_array().context("lists must be an array of tables")?;
        ensure!(lists.len() >= 2, "lists must contain at least two tables");
        for key in [
            "reference_configuration",
            "active_space",
            "j_min",
            "j_max",
            "excitations",
            "continue_lists",
        ] {
            ensure!(
                !s.values.contains_key(key),
                "{key} belongs inside each list"
            );
        }
        let mut result = Vec::new();
        for item in lists {
            let mut model = super::config::canonicalize(
                "csfsgenerate",
                item.as_table()
                    .context("each list must be a table")?
                    .clone(),
            )?;
            for key in model.keys() {
                ensure!(
                    [
                        "reference_configuration",
                        "active_space",
                        "j_min",
                        "j_max",
                        "excitations"
                    ]
                    .contains(&key.as_str()),
                    "unsupported list key: {key}"
                );
            }
            for key in ["inactive_core", "orbital_order"] {
                if let Some(value) = s.values.get(key) {
                    model.insert(key.into(), value.clone());
                }
            }
            result.push(transcript(&model)?);
        }
        return Ok(Some(result));
    }
    if s.values
        .keys()
        .any(|k| MODEL_KEYS.contains(&k.as_str()) && k != "as")
    {
        return Ok(Some(vec![transcript(&s.values)?]));
    }
    Ok(None)
}

pub(super) struct Outputs {
    pub csf: PathBuf,
    pub parquet: PathBuf,
    pub header: PathBuf,
    pub descriptor: PathBuf,
    pub metadata: PathBuf,
    pub write_parquet: bool,
    pub write_descriptors: bool,
}
impl Outputs {
    pub fn from_settings(s: &Settings) -> Result<Self> {
        let csf = s.output("rcsfs_out", Some(Path::new("rcsf.out")))?;
        let parquet = s.output("rcsfs_parquet", Some(&csf.with_extension("parquet")))?;
        let header = files::header(&csf, &parquet)?;
        let descriptor = s.output(
            "descriptor",
            Some(&csf.with_file_name(format!("{}_descriptors.parquet", files::stem(&csf)?))),
        )?;
        let write_descriptors = s.flag("generate_descriptors", false)?;
        Ok(Self {
            csf,
            parquet,
            header,
            metadata: descriptor.with_extension("toml"),
            descriptor,
            write_parquet: s.flag("generate_parquet", false)? || write_descriptors,
            write_descriptors,
        })
    }
    pub fn paths(&self) -> Vec<PathBuf> {
        let mut result = vec![self.csf.clone()];
        if self.write_parquet {
            result.extend([self.parquet.clone(), self.header.clone()]);
        }
        if self.write_descriptors {
            result.extend([self.descriptor.clone(), self.metadata.clone()]);
        }
        result
    }
}
fn estimate_json(
    estimate: &disk::DiskEstimate,
    checks: &[configuration_generation::csf_generation::SpaceCheck],
) -> Json {
    let c = &estimate.capacity;
    json!({"success": true, "estimate_only": true, "unique_occupations": estimate.plan_stats.unique_occupations,
        "pre_deduplication_records": c.pre_deduplication_records, "peel_subshells": c.peel_subshells,
        "v2_columns": c.v2_columns, "segment_codec": estimate.segment_codec, "deduplication": estimate.deduplication,
        "enumeration_millis": estimate.enumeration_millis, "planning_millis": estimate.planning_millis,
        "plan_stats": estimate.plan_stats, "space_checks": checks, "failure_recovery": "restart",
        "assumptions": c.assumption_lines(), "bytes": {"segments": c.segment_bytes, "root_buckets": c.root_bucket_bytes,
        "recursive_buckets": c.recursive_bucket_bytes, "survivor_bitsets": c.survivor_bitset_bytes,
        "scratch_peak": c.scratch_peak_bytes, "descriptor": c.descriptor_bytes, "csf_text": c.csf_text_bytes,
        "csf_parquet": c.csf_parquet_bytes, "staged_outputs": c.staged_output_bytes,
        "required_scratch": c.required_scratch_bytes, "required_output": c.required_output_bytes}})
}

pub(super) fn run(s: &Settings) -> Result<Json> {
    let outputs = Outputs::from_settings(s)?;
    let inputs = s
        .string("transcript")?
        .map(PathBuf::from)
        .into_iter()
        .collect::<Vec<_>>();
    files::validate_set(&outputs.paths(), &inputs, s.config.as_deref())?;
    let mut interactive_model = None;
    let transcripts = match configured_transcripts(s)? {
        Some(t) => t,
        None => {
            let model = interactive()?;
            let text = transcript(&model)?;
            interactive_model = Some(model);
            vec![text]
        }
    };
    let multiple = transcripts.len() > 1;
    let storage = s.text(
        "generation_storage",
        if multiple || outputs.write_parquet && s.config.is_some() {
            "disk"
        } else {
            "memory"
        },
    )?;
    ensure!(
        !multiple || storage == "disk",
        "multiple lists require disk generation"
    );
    let disk_mode = storage == "disk";
    ensure!(
        s.positive("memory_budget_mib")?.is_none() || disk_mode,
        "memory_budget_mib requires disk generation"
    );
    let estimate_only = s.flag("estimate_only", false)?;
    ensure!(
        !estimate_only || disk_mode,
        "estimate_only requires disk generation"
    );
    let allow_unchecked = s.flag("allow_unchecked_space", false)?;
    let threads = s.positive("threads")?;
    let estimate_options =
        GenerationOptions::from_api(threads, s.positive("memory_budget_mib")?, None)?;
    if disk_mode {
        let mut destinations = vec![(outputs.csf.clone(), ArtifactKind::CsfText)];
        if outputs.write_parquet {
            destinations.extend([
                (outputs.parquet.clone(), ArtifactKind::CsfParquet),
                (outputs.header.clone(), ArtifactKind::Header),
            ]);
        }
        if outputs.write_descriptors {
            destinations.extend([
                (outputs.descriptor.clone(), ArtifactKind::Descriptor),
                (outputs.metadata.clone(), ArtifactKind::DescriptorMetadata),
            ]);
        }
        let layout = disk::SpaceLayout {
            scratch_dir: Some(PathBuf::from(".")),
            staging_dir: Some(PathBuf::from(".")),
            destinations,
        };
        let (estimate, checks) = if multiple {
            disk::estimate_disk_generation_from_transcripts_with_layout(
                &transcripts,
                &estimate_options,
                &layout,
            )?
        } else {
            disk::estimate_disk_generation_with_layout(&transcripts[0], &estimate_options, &layout)?
        };
        if estimate_only {
            return Ok(estimate_json(&estimate, &checks));
        }
        for check in &checks {
            ensure!(
                check.sufficient != Some(false),
                "insufficient disk space at {}: need {} bytes, free {:?}",
                check.path,
                check.required_bytes,
                check.free_bytes
            );
            ensure!(
                check.sufficient.is_some() || allow_unchecked,
                "free space unavailable at {}; use --allow-unchecked-space to proceed",
                check.path
            );
        }
    }
    let stage = files::stage()?;
    let text_dir = stage.path().join("text");
    fs::create_dir(&text_dir)?;
    let csf = text_dir.join(outputs.csf.file_name().context("missing output filename")?);
    let parquet = stage.path().join("csfs.parquet");
    let header = files::header(&csf, &parquet)?;
    let descriptor = stage.path().join("features.parquet");
    let mut stats = if disk_mode {
        let options = GenerationOptions::from_api_with_space_policy(
            threads,
            s.positive("memory_budget_mib")?,
            Some(stage.path().join("scratch")),
            allow_unchecked,
        )?;
        let stats = if multiple {
            disk::generate_disk_outputs_from_transcripts_with_options(
                &transcripts,
                &csf,
                &parquet,
                &descriptor,
                &header,
                &options,
            )?
        } else {
            disk::generate_disk_outputs_from_transcript_with_options(
                &transcripts[0],
                &csf,
                &parquet,
                &descriptor,
                &header,
                &options,
            )?
        };
        let mut result = serde_json::to_value(&stats)?;
        result["record_count"] = json!(stats.unique_count);
        result["descriptor_count"] = json!(stats.unique_count);
        result
    } else {
        let stats = configuration_generation::generate_csfs_from_transcript(
            &transcripts[0],
            &csf,
            threads,
        )?;
        if outputs.write_parquet {
            commands::convert(&csf, &parquet, s)?;
        }
        json!({"record_count": stats.record_count, "block_count": stats.block_count, "unique_occupations": stats.unique_occupations, "output_bytes": stats.output_bytes})
    };
    let mut sources = vec![csf.clone()];
    if outputs.write_parquet {
        sources.extend([parquet.clone(), header.clone()]);
    }
    if outputs.write_descriptors {
        if disk_mode {
            let shells =
                crate::csfs_descriptor::parquet_batch::read_peel_subshells_from_header(&header)?;
            commands::sidecar(
                &descriptor.with_extension("toml"),
                stats["record_count"]
                    .as_u64()
                    .context("missing record_count")? as usize,
                &shells,
            )?;
        } else {
            let desc =
                commands::descriptors(&parquet, &header, &descriptor, threads, Some("zstd"))?;
            stats["descriptor_count"] = desc["descriptor_count"].clone();
        }
        sources.extend([descriptor.clone(), descriptor.with_extension("toml")]);
    } else {
        stats.as_object_mut().unwrap().remove("descriptor_count");
    }
    files::publish(&sources, &outputs.paths(), true)?;
    stats["success"] = true.into();
    stats["output_file"] = json!(outputs.csf);
    if outputs.write_parquet {
        stats["parquet_file"] = json!(outputs.parquet);
        stats["header_file"] = json!(outputs.header);
    }
    if outputs.write_descriptors {
        stats["descriptor_parquet_file"] = json!(outputs.descriptor);
        stats["descriptor_metadata_file"] = json!(outputs.metadata);
    }
    if let Some(model) = interactive_model {
        let mut values = s.values.clone();
        values.extend(model);
        values.remove("transcript");
        values.insert("generation_storage".into(), storage.into());
        values.insert("rcsfs_out".into(), outputs.csf.display().to_string().into());
        let mut root = Table::new();
        root.insert("csfsgenerate".into(), Value::Table(values));
        let source = stage.path().join("interactive.toml");
        fs::write(&source, toml::to_string(&root)?)?;
        files::publish(&[source], &[PathBuf::from("rcsfs.toml")], true)?;
    }
    Ok(stats)
}
