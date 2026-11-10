//! TOML model/layer planning for the direct CSFG generator.
use super::{arguments::Settings, files};
use crate::configuration_generation::csfg_generation::{CsfgRequest, generate_csfgs};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use toml::Table;

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Model {
    #[serde(rename = "as")]
    level: Option<usize>,
    inactive_core: u8,
    reference_configuration: Vec<String>,
    labeling_space: String,
    active_space: String,
    j_min: u16,
    j_max: u16,
    excitations: u8,
    rcsfg_out: Option<PathBuf>,
    rlabel_out: Option<PathBuf>,
    threads: Option<usize>,
    #[serde(default)]
    overwrite: bool,
}
impl Model {
    fn name(&mut self, conf: Option<&str>, level: usize) -> Result<()> {
        let conf = conf.context("numbered CSFG outputs require conf")?;
        self.level = Some(level);
        self.rcsfg_out = Some(format!("{conf}_as{level}raw.g").into());
        self.rlabel_out = Some(format!("{conf}_as{level}raw.l").into());
        Ok(())
    }
    fn paths(&self) -> [PathBuf; 2] {
        [
            self.rcsfg_out.clone().unwrap_or_else(|| "rcsfg.out".into()),
            self.rlabel_out
                .clone()
                .unwrap_or_else(|| "rlabel.out".into()),
        ]
    }
    fn request(&self) -> CsfgRequest {
        CsfgRequest {
            inactive_core: self.inactive_core,
            reference_configuration: self.reference_configuration.clone(),
            labeling_space: self.labeling_space.clone(),
            active_space: self.active_space.clone(),
            j_min: self.j_min,
            j_max: self.j_max,
            excitations: self.excitations,
        }
    }
}
fn space(value: &str) -> Result<BTreeMap<char, usize>> {
    let mut result = BTreeMap::new();
    for item in value.split(',') {
        let token = item.trim();
        let channel = token.chars().last().context("empty active-space channel")?;
        ensure!(
            "spdfghiklmnoqrtuvwxyz".contains(channel),
            "invalid active-space channel: {token}"
        );
        let n: usize = token[..token.len() - channel.len_utf8()]
            .parse()
            .context("invalid principal quantum number")?;
        ensure!(
            n > 0 && result.insert(channel, n).is_none(),
            "invalid or duplicate active-space channel: {token}"
        );
    }
    Ok(result)
}
fn plan(root: &Table) -> Result<(Vec<Model>, bool)> {
    for key in root.keys() {
        ensure!(
            ["conf", "csfggenerate", "csfg-split", "csfs-split"].contains(&key.as_str()),
            "unsupported CSFG section: {key}"
        );
    }
    let conf = root
        .get("conf")
        .map(|v| v.as_str().context("conf must be a string"))
        .transpose()?;
    if let Some(conf) = conf {
        ensure!(
            conf.as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphanumeric)
                && conf
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)),
            "conf must be a filename stem without directories"
        );
    }
    let raw = root
        .get("csfggenerate")
        .context("config has no csfggenerate table")?;
    let mut batch = raw.is_array();
    let items: Vec<_> = raw
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_else(|| vec![raw]);
    ensure!(
        !items.is_empty(),
        "csfggenerate requires at least one model"
    );
    let mut jobs: Vec<Model> = Vec::new();
    let mut indexed = BTreeMap::new();
    for item in items {
        let mut model: Model = item.clone().try_into()?;
        if batch || model.level.is_some() {
            let level = model.level.context("array models require as")?;
            ensure!(
                !indexed.contains_key(&level),
                "duplicate generation as = {level}"
            );
            ensure!(
                model.rcsfg_out.is_none() && model.rlabel_out.is_none(),
                "numbered models use conf filenames, not explicit output paths"
            );
            model.name(conf, level)?;
            indexed.insert(level, jobs.len());
        }
        jobs.push(model);
    }
    ensure!(
        !(root.contains_key("csfg-split") && root.contains_key("csfs-split")),
        "use only one of csfg-split and csfs-split"
    );
    if let Some(raw) = root.get("csfg-split").or_else(|| root.get("csfs-split")) {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Layers {
            source_as: Option<usize>,
            active_spaces: BTreeMap<String, String>,
        }
        let layers: Layers = raw.clone().try_into()?;
        ensure!(
            !layers.active_spaces.is_empty(),
            "active_spaces must not be empty"
        );
        let source_as = layers
            .source_as
            .or_else(|| indexed.keys().last().copied())
            .context("layers require numbered CSFG models")?;
        let source = jobs[*indexed
            .get(&source_as)
            .context("source_as must identify a generation model")?]
        .clone();
        let maximum = space(&source.active_space)?;
        for (label, active) in layers.active_spaces {
            let digits = label
                .strip_prefix("AS")
                .context("layer names must be AS0, AS1, ...")?;
            ensure!(
                digits == "0" || !digits.starts_with('0'),
                "invalid layer name: {label}"
            );
            let level: usize = digits.parse()?;
            for (channel, n) in space(&active)? {
                ensure!(
                    n <= *maximum.get(&channel).unwrap_or(&0),
                    "{label} exceeds the source active space"
                );
            }
            let mut model = source.clone();
            model.active_space = active;
            model.name(conf, level)?;
            if let Some(&index) = indexed.get(&level) {
                ensure!(
                    jobs[index] == model,
                    "{label} conflicts with an explicit model"
                );
            } else {
                indexed.insert(level, jobs.len());
                jobs.push(model);
            }
        }
        batch = true;
    }
    Ok((jobs, batch))
}
pub(super) fn output_paths(root: &Table) -> Result<Vec<PathBuf>> {
    Ok(plan(root)?.0.iter().flat_map(Model::paths).collect())
}

pub(super) fn run(root: &Table, config: &Path, overrides: &Settings) -> Result<Json> {
    let (jobs, batch) = plan(root)?;
    let mut destinations = Vec::new();
    let mut policies = Vec::new();
    for model in &jobs {
        let policy = overrides.flag("overwrite", false)? || model.overwrite;
        let paths = model.paths();
        files::validate_set(&paths, &destinations, Some(config))?;
        ensure!(model.threads != Some(0), "threads must be positive");
        ensure!(
            !model.reference_configuration.is_empty(),
            "reference_configuration must not be empty"
        );
        for path in paths {
            ensure!(
                policy || !path.exists(),
                "output exists (use --overwrite): {}",
                path.display()
            );
            destinations.push(path);
            policies.push(policy);
        }
    }
    let stage = files::stage()?;
    let mut sources = Vec::new();
    let mut results = Vec::new();
    for (index, model) in jobs.iter().enumerate() {
        let g = stage.path().join(format!("{index}.g"));
        let l = stage.path().join(format!("{index}.l"));
        let stats = generate_csfgs(
            &model.request(),
            &g,
            &l,
            overrides.positive("threads")?.or(model.threads),
        )?;
        let blocks: Vec<_> = stats.blocks.iter().map(|b| json!({"two_j": b.two_j, "parity": if b.odd { "-" } else { "+" }, "record_count": b.record_count, "expanded_csf_count": b.expanded_csf_count, "label_csf_count": b.label_csf_count})).collect();
        results.push(json!({"success": true, "rcsfg_out": destinations[2 * index], "rlabel_out": destinations[2 * index + 1],
            "record_count": stats.record_count, "label_csf_count": stats.label_csf_count, "generator_count": stats.record_count - stats.label_csf_count,
            "expanded_csf_count": stats.expanded_csf_count, "type_counts": stats.type_counts, "unique_occupations": stats.unique_occupations,
            "representative_occupations": stats.representative_occupations, "block_count": stats.blocks.len(), "blocks": blocks,
            "csfg_bytes": stats.csfg_bytes, "label_bytes": stats.label_bytes, "planning_seconds": stats.planning_seconds,
            "generation_seconds": stats.generation_seconds, "writing_seconds": stats.writing_seconds}));
        sources.extend([g, l]);
    }
    for (index, ((source, destination), policy)) in
        sources.iter().zip(&destinations).zip(policies).enumerate()
    {
        files::publish(
            std::slice::from_ref(source),
            std::slice::from_ref(destination),
            policy,
        )
        .with_context(|| {
            format!(
                "CSFG publication failed; earlier published outputs: {:?}",
                &destinations[..index]
            )
        })?;
    }
    Ok(if batch {
        json!({"success": true, "results": results})
    } else {
        results.remove(0)
    })
}
