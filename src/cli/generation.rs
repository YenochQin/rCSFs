//! Generation orchestration; all CSF algorithms and capacity models remain shared.
use super::{arguments::Settings, files};
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
            let mut model = item
                .as_table()
                .context("each list must be a table")?
                .clone();
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
    pub descriptor: PathBuf,
    pub header: PathBuf,
}
impl Outputs {
    pub fn from_settings(s: &Settings) -> Result<Self> {
        let descriptor = s.output("output", Some(Path::new("rcsf_descriptors.parquet")))?;
        let header = crate::descriptor_dataset::header_path(&descriptor)?;
        Ok(Self { descriptor, header })
    }
    pub fn paths(&self) -> Vec<PathBuf> {
        vec![self.descriptor.clone(), self.header.clone()]
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
        "scratch_peak": c.scratch_peak_bytes, "descriptor": c.descriptor_bytes,
        "staged_outputs": c.staged_output_bytes,
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
    let storage = s.text("generation_storage", "disk")?;
    ensure!(
        matches!(storage, "disk" | "memory"),
        "generation_storage must be disk or memory"
    );
    let disk_mode = storage == "disk";
    ensure!(
        transcripts.len() == 1 || disk_mode,
        "multiple lists require disk generation"
    );
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
        let layout = disk::SpaceLayout {
            scratch_dir: Some(PathBuf::from(".")),
            staging_dir: Some(PathBuf::from(".")),
            destinations: vec![
                (outputs.descriptor.clone(), ArtifactKind::Descriptor),
                (outputs.header.clone(), ArtifactKind::Header),
            ],
        };
        let (estimate, checks) = if transcripts.len() > 1 {
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
    let descriptor = stage.path().join(
        outputs
            .descriptor
            .file_name()
            .context("missing descriptor filename")?,
    );
    let header = stage.path().join(
        outputs
            .header
            .file_name()
            .context("missing header filename")?,
    );
    let mut stats = if disk_mode {
        let options = GenerationOptions::from_api_with_space_policy(
            threads,
            s.positive("memory_budget_mib")?,
            Some(stage.path().join("scratch")),
            allow_unchecked,
        )?;
        let stats =
            disk::generate_descriptor_dataset(&transcripts, &descriptor, &header, &options)?;
        // The capacity model releases scratch before staging the published copies.
        fs::remove_dir_all(stage.path().join("scratch"))?;
        let mut result = serde_json::to_value(&stats)?;
        result["record_count"] = json!(stats.unique_count);
        result
    } else {
        let stats = configuration_generation::generate_csfs_from_transcript(
            &transcripts[0],
            &descriptor,
            threads,
        )?;
        json!({"record_count": stats.record_count, "block_count": stats.block_count, "blocks": stats.blocks, "unique_occupations": stats.unique_occupations, "output_bytes": stats.output_bytes})
    };
    files::publish(&[descriptor, header], &outputs.paths(), true)?;
    stats["success"] = true.into();
    stats["output_file"] = json!(outputs.descriptor);
    stats["header_file"] = json!(outputs.header);
    stats["descriptor_count"] = stats["record_count"].clone();
    if let Some(model) = interactive_model {
        let mut values = s.values.clone();
        values.extend(model);
        values.remove("transcript");
        values.insert("generation_storage".into(), storage.into());
        values.insert(
            "output".into(),
            outputs.descriptor.display().to_string().into(),
        );
        let mut root = Table::new();
        root.insert("csfsgenerate".into(), Value::Table(values));
        let source = stage.path().join("interactive.toml");
        fs::write(&source, toml::to_string(&root)?)?;
        files::publish(&[source], &[PathBuf::from("rcsfs.toml")], true)?;
    }
    Ok(stats)
}
