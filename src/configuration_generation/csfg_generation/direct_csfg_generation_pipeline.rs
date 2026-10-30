//! Direct GRASPG generators for a common contiguous S/SD correlation space.
//! Plan at occupation level; angular recursion only sees unique representatives.

use anyhow::{Result, ensure};
use std::path::Path;
use std::time::Instant;

use super::CsfgStats;
use super::{graspg_csfg_output_writer, representative_occupation_planning};
use crate::configuration_generation::shared_generation::{
    ExcitationRequest, GenerationRequest, ReferenceConfiguration, generate_csfs_parallel,
};

/// Physical input, independent of GRASPG's two-list interactive transcript.
#[derive(Clone, Debug)]
pub struct CsfgRequest {
    pub inactive_core: u8,
    pub reference_configuration: Vec<String>,
    pub labeling_space: String,
    pub active_space: String,
    pub j_min: u16,
    pub j_max: u16,
    pub excitations: u8,
}

impl CsfgRequest {
    fn physical_request(&self) -> Result<ExcitationRequest> {
        ensure!(
            self.excitations <= 2,
            "CSFG generation supports only S/SD (0..=2 excitations)"
        );
        ensure!(
            !self.reference_configuration.is_empty(),
            "at least one reference is required"
        );
        // Reuse the existing normalization of user core selectors and reference
        // syntax. This in-memory adapter never invokes Fortran or writes input.
        for reference in &self.reference_configuration {
            ensure!(
                reference.is_ascii(),
                "reference_configuration must use ASCII spectroscopic notation"
            );
            ReferenceConfiguration::from_spectroscopic(reference)?;
        }
        ensure!(
            !self.active_space.contains(['\n', '\r']),
            "active_space must be one line"
        );
        ExcitationRequest::from_transcript(&format!(
            "* ! Orbital order\n{}\n{}\n\n{}\n{},{}\n{}\nn\n",
            self.inactive_core,
            self.reference_configuration.join("\n"),
            self.active_space,
            self.j_min,
            self.j_max,
            self.excitations
        ))
    }
}

/// Generate two NEW staged files. Callers own their final publication policy.
/// No full-CSF text, descriptor file, or external executable is involved.
pub fn generate_csfgs(
    request: &CsfgRequest,
    csfg_output: &Path,
    label_output: &Path,
    threads: Option<usize>,
) -> Result<CsfgStats> {
    ensure!(threads != Some(0), "threads must be greater than zero");
    ensure!(
        csfg_output != label_output,
        "CSFG and label outputs must be distinct"
    );
    for path in [csfg_output, label_output] {
        ensure!(
            std::fs::symlink_metadata(path).is_err(),
            "output already exists: {}",
            path.display()
        );
        ensure!(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."))
                .is_dir(),
            "output parent does not exist: {}",
            path.display()
        );
    }
    let start = Instant::now();
    let physical = request.physical_request()?;
    let plan = representative_occupation_planning::Plan::build(&physical, &request.labeling_space)?;
    let planning_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let requests = plan
        .templates
        .iter()
        .map(|template| GenerationRequest {
            core_subshells: plan.core.clone(),
            configuration: template.occupations.clone(),
            min_two_j: physical.min_two_j,
            max_two_j: physical.max_two_j,
        })
        .collect::<Vec<_>>();
    let chunks = generate_csfs_parallel(&requests, threads)?;
    let generation_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut stats = graspg_csfg_output_writer::write(&plan, &chunks, csfg_output, label_output)?;
    stats.planning_seconds = planning_seconds;
    stats.generation_seconds = generation_seconds;
    stats.writing_seconds = start.elapsed().as_secs_f64();
    Ok(stats)
}
