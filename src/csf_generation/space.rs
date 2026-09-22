//! Free-space probing and the pre-flight policy for a disk generation run.
//!
//! This module answers one question — can these volumes hold this run? — and
//! decides what to do when the answer is no or cannot be determined. The
//! estimate it is asked about comes from [`super::capacity`], which owns the
//! byte model; keeping the two apart means a change to how space is measured or
//! refused does not touch the measured ratios, and vice versa.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

use super::capacity::{ArtifactKind, CapacityEstimate, with_margin};

/// Free bytes available to the current user on `path`'s filesystem.
///
/// `None` means the platform does not report it; it never means "enough".
#[cfg(unix)]
pub(crate) fn free_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    // SAFETY: statvfs initializes the structure when it returns 0.
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the successful statvfs call initialized `stats`.
    let stats = unsafe { stats.assume_init() };
    u64::try_from(stats.f_bavail)
        .ok()?
        .checked_mul(u64::try_from(stats.f_frsize).ok()?)
}

#[cfg(not(unix))]
pub(crate) fn free_bytes(_path: &Path) -> Option<u64> {
    None
}

/// What a single volume's requirement found.
#[derive(Clone, Debug)]
pub(crate) struct SpaceCheck {
    pub(crate) path: String,
    /// The largest simultaneous requirement any phase places on this volume.
    pub(crate) required_bytes: u64,
    /// `None` when the platform did not report the available space. The run is
    /// then refused unless the caller explicitly accepted an unchecked space.
    pub(crate) free_bytes: Option<u64>,
    pub(crate) sufficient: Option<bool>,
}

/// How a path participates in the storage a run needs.
///
/// Scratch and staged outputs coexist while records are generated; staged
/// outputs and the published copies coexist while the CLI publishes. Adding
/// every requirement on one volume together would be needlessly pessimistic, so
/// the roles are summed per phase and the phases are compared by their maximum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpaceRole {
    /// Temporary segments, buckets and bitsets.
    Scratch,
    /// The staged artifact set, present in both phases.
    Staging,
    /// A published copy of one artifact, present only during publication.
    Published { artifact: ArtifactKind, bytes: u64 },
}

/// Identifies the filesystem a path lives on, so requirements that share a
/// volume are added rather than each compared against the whole volume.
///
/// `None` means the platform cannot report it, in which case each path is
/// treated as its own group: that over-estimates rather than under-estimates.
#[cfg(unix)]
fn device_id(path: &Path) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;

    std::fs::metadata(path).ok().map(|metadata| metadata.dev())
}

#[cfg(not(unix))]
fn device_id(_path: &Path) -> Option<u64> {
    None
}

/// Resolve a path to a directory that exists, so it can be measured.
fn measuring_directory(path: &Path) -> PathBuf {
    if path.is_dir() {
        return path.to_path_buf();
    }
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// What a caller wants done with an insufficient or unmeasurable volume.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpacePolicy {
    /// About to write: refuse rather than start a run that cannot finish.
    ///
    /// A volume that is *known* to be too small is always refused; the opt-out
    /// exists only for a volume the platform cannot measure, where the caller
    /// is accepting an unknown rather than a known failure.
    Require { allow_unchecked: bool },
    /// Reporting only: record what was found and let the caller decide.
    ///
    /// An estimate exists to answer "would this fit", so refusing to produce it
    /// would withhold the answer.
    Report,
}

impl SpacePolicy {
    /// Whether a known-insufficient volume stops the caller.
    fn refuses_insufficient(self) -> bool {
        matches!(self, Self::Require { .. })
    }

    /// Whether an unmeasurable volume stops the caller.
    fn refuses_unmeasurable(self) -> bool {
        matches!(
            self,
            Self::Require {
                allow_unchecked: false
            }
        )
    }
}

/// Check every volume a run touches, summing the requirements that coexist.
///
/// Each published artifact may be named once: two entries for one artifact
/// would charge its size twice and report a requirement the run never reaches.
pub(crate) fn check_space(
    estimate: &CapacityEstimate,
    entries: &[(PathBuf, SpaceRole)],
    policy: SpacePolicy,
) -> Result<Vec<SpaceCheck>> {
    let mut seen_artifacts = std::collections::HashSet::new();
    for (path, role) in entries {
        if let SpaceRole::Published { artifact, .. } = role
            && !seen_artifacts.insert(*artifact)
        {
            bail!(
                "{} is the destination of two {} artifacts; each artifact has one \
                 destination",
                path.display(),
                artifact.name()
            );
        }
    }
    struct Group {
        directory: PathBuf,
        scratch: u64,
        staging: u64,
        published: u64,
    }
    let mut groups: Vec<(Option<u64>, Group)> = Vec::new();
    for (path, role) in entries {
        let directory = measuring_directory(path);
        let device = device_id(&directory);
        // Group by volume, not by directory: requirements in different
        // directories of one volume still compete for the same bytes, and
        // checking them separately would let each pass while their sum does
        // not fit. When the volume cannot be identified, every path becomes its
        // own group, which over-estimates instead of merging unrelated storage.
        let existing = device
            .and_then(|device| groups.iter_mut().find(|(known, _)| *known == Some(device)))
            .map(|(_, group)| group);
        let group = match existing {
            Some(group) => group,
            None => {
                groups.push((
                    device,
                    Group {
                        directory: directory.clone(),
                        scratch: 0,
                        staging: 0,
                        published: 0,
                    },
                ));
                &mut groups.last_mut().expect("a group was just pushed").1
            }
        };
        match role {
            SpaceRole::Scratch => {
                group.scratch = group.scratch.max(estimate.required_scratch_bytes)
            }
            SpaceRole::Staging => group.staging = group.staging.max(estimate.required_output_bytes),
            SpaceRole::Published { bytes, .. } => {
                group.published = group
                    .published
                    .checked_add(with_margin(*bytes)?)
                    .context("published space requirement overflow")?;
            }
        }
    }

    let mut checks = Vec::new();
    for (_, group) in groups {
        // Generation holds scratch and the staged set at the same time;
        // publication holds the staged set and the published copies.
        let generation = group
            .scratch
            .checked_add(group.staging)
            .context("space requirement overflow")?;
        let publication = group
            .staging
            .checked_add(group.published)
            .context("space requirement overflow")?;
        let required = generation.max(publication);
        let path = group.directory.display().to_string();
        match free_bytes(&group.directory) {
            Some(free) => {
                let sufficient = free >= required;
                // A volume that is known to be too small stops a run whoever
                // asked: "unchecked" is not "checked and fine".
                if !sufficient && policy.refuses_insufficient() {
                    bail!(
                        "not enough free space at {path}: the run needs {required} bytes \
                         (scratch plus staged outputs or staged outputs plus published \
                         copies, including the safety margin) and {free} bytes are available"
                    );
                }
                checks.push(SpaceCheck {
                    path,
                    required_bytes: required,
                    free_bytes: Some(free),
                    sufficient: Some(sufficient),
                });
            }
            None => {
                if policy.refuses_unmeasurable() {
                    bail!(
                        "cannot check free space at {path}: this platform does not report it, so \
                         the run would proceed without the pre-flight it asked for. Pass \
                         allow_unchecked_space (CLI: --allow-unchecked-space) to accept that, or \
                         run where free space can be measured"
                    );
                }
                checks.push(SpaceCheck {
                    path,
                    required_bytes: required,
                    free_bytes: None,
                    sufficient: None,
                });
            }
        }
    }
    Ok(checks)
}

/// Check the volumes a disk generation owns: scratch and its staging outputs.
pub(crate) fn preflight_run(
    estimate: &CapacityEstimate,
    scratch_dir: &Path,
    output_paths: &[&Path],
    allow_unchecked_space: bool,
) -> Result<Vec<SpaceCheck>> {
    let policy = SpacePolicy::Require {
        allow_unchecked: allow_unchecked_space,
    };
    let mut entries = vec![(scratch_dir.to_path_buf(), SpaceRole::Scratch)];
    let mut seen = std::collections::HashSet::new();
    for output in output_paths {
        let directory = measuring_directory(output);
        if seen.insert(directory.clone()) {
            entries.push((directory, SpaceRole::Staging));
        }
    }
    check_space(estimate, &entries, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csf_generation::DeduplicationStrategy;
    use crate::csf_generation::capacity::with_margin;
    use crate::csf_generation::estimate_capacity;

    /// Scratch and staging compete for the same volume, so their requirements
    /// add up. Checking them separately would let two half-sized checks pass on
    /// a volume that cannot hold both.
    #[test]
    fn requirements_on_one_volume_are_added_together() {
        let estimate = estimate_capacity(56, 1_000_000, 2, DeduplicationStrategy::Exact).unwrap();
        let temporary = std::env::temp_dir();
        let scratch = temporary.join("rcsfs-capacity-test-scratch");
        let staged = temporary.join("rcsfs-capacity-test-staged");
        let checks = check_space(
            &estimate,
            &[
                (scratch.clone(), SpaceRole::Scratch),
                (staged.clone(), SpaceRole::Staging),
            ],
            SpacePolicy::Report,
        )
        .unwrap();
        let combined = checks
            .iter()
            .map(|check| check.required_bytes)
            .max()
            .unwrap();
        assert_eq!(
            combined,
            estimate.required_scratch_bytes + estimate.required_output_bytes,
            "two directories on one volume must be checked against their sum"
        );
        assert!(
            combined > estimate.required_scratch_bytes,
            "the combined check must be stricter than the scratch check alone"
        );
        // The phases are compared by their maximum, so a published copy that
        // shares the volume with scratch does not stack on top of it: scratch
        // is gone by the time publication starts.
        let with_publication = check_space(
            &estimate,
            &[
                (scratch, SpaceRole::Scratch),
                (staged, SpaceRole::Staging),
                (
                    temporary.join("rcsfs-capacity-test-published"),
                    SpaceRole::Published {
                        artifact: ArtifactKind::Descriptor,
                        bytes: estimate.staged_output_bytes,
                    },
                ),
            ],
            SpacePolicy::Report,
        )
        .unwrap();
        let phases = with_publication
            .iter()
            .map(|check| check.required_bytes)
            .max()
            .unwrap();
        let generation = estimate.required_scratch_bytes + estimate.required_output_bytes;
        let publication =
            estimate.required_output_bytes + with_margin(estimate.staged_output_bytes).unwrap();
        assert_eq!(phases, generation.max(publication));
        assert_eq!(phases, generation);
    }

    /// The policy matrix: three verdicts against three policies.
    ///
    /// A volume measured and found short stops any caller about to write,
    /// whatever `allow_unchecked` says: the opt-out is about an *unknown*
    /// volume, not about overriding a measurement. An unmeasurable volume stops
    /// a `Require` caller unless the opt-out accepts the unknown. A reporting
    /// caller always gets the verdict, including the two failing ones.
    #[test]
    fn the_space_policy_decides_each_verdict_independently() {
        let roomy = estimate_capacity(56, 1_000_000, 2, DeduplicationStrategy::Exact).unwrap();
        let mut short = estimate_capacity(56, 1_000_000, 2, DeduplicationStrategy::Exact).unwrap();
        short.required_scratch_bytes = u64::MAX / 2;
        let existing = [(std::env::temp_dir(), SpaceRole::Scratch)];
        // Neither this path nor its parent exists, so the volume cannot be
        // measured at all.
        let missing = [(
            std::env::temp_dir()
                .join("rcsfs-space-test-absent")
                .join("scratch"),
            SpaceRole::Scratch,
        )];
        let policies = [
            SpacePolicy::Require {
                allow_unchecked: false,
            },
            SpacePolicy::Require {
                allow_unchecked: true,
            },
            SpacePolicy::Report,
        ];

        for (verdict, estimate, entries) in [
            ("sufficient", &roomy, &existing[..]),
            ("insufficient", &short, &existing[..]),
            ("unmeasurable", &roomy, &missing[..]),
        ] {
            for policy in policies {
                let outcome = check_space(estimate, entries, policy);
                let expected_refusal = match (verdict, policy) {
                    ("insufficient", SpacePolicy::Require { .. }) => true,
                    (
                        "unmeasurable",
                        SpacePolicy::Require {
                            allow_unchecked: false,
                        },
                    ) => true,
                    _ => false,
                };
                assert_eq!(
                    outcome.is_err(),
                    expected_refusal,
                    "{verdict} volume under {policy:?}"
                );
                // The two refusals must stay distinguishable: one says the
                // volume is too small, the other that it could not be measured.
                match (&outcome, verdict) {
                    (Err(error), "insufficient") => {
                        assert!(error.to_string().contains("not enough free space at"));
                    }
                    (Err(error), "unmeasurable") => {
                        assert!(error.to_string().contains("cannot check free space"));
                    }
                    _ => {}
                }
                let Ok(checks) = outcome else {
                    continue;
                };
                assert_eq!(checks.len(), 1);
                let expected_sufficient = match verdict {
                    "sufficient" => Some(true),
                    "insufficient" => Some(false),
                    _ => None,
                };
                assert_eq!(
                    checks[0].sufficient, expected_sufficient,
                    "{verdict} volume under {policy:?} reported the wrong verdict"
                );
            }
        }
    }

    /// One artifact, one destination: naming it twice would charge its size
    /// twice and report a requirement the run never reaches.
    #[test]
    fn one_artifact_cannot_be_published_twice() {
        let estimate = estimate_capacity(56, 1_000_000, 2, DeduplicationStrategy::Exact).unwrap();
        let published = |name: &str, artifact: ArtifactKind| {
            (
                std::env::temp_dir().join(name),
                SpaceRole::Published {
                    artifact,
                    bytes: 1024,
                },
            )
        };
        let twice = [
            published("rcsfs-space-test-a.parquet", ArtifactKind::Descriptor),
            published("rcsfs-space-test-b.parquet", ArtifactKind::Descriptor),
        ];
        let error = check_space(&estimate, &twice, SpacePolicy::Report)
            .expect_err("the same artifact must not be charged twice");
        assert!(error.to_string().contains("two descriptor artifacts"));

        // Different artifacts of one run are fine, and share a volume.
        let distinct = [
            published("rcsfs-space-test.c", ArtifactKind::CsfText),
            published("rcsfs-space-test.parquet", ArtifactKind::Descriptor),
        ];
        let checks = check_space(&estimate, &distinct, SpacePolicy::Report).unwrap();
        assert_eq!(checks.len(), 1);
        assert_eq!(
            checks[0].required_bytes,
            with_margin(1024 + 1024).unwrap(),
            "two artifacts on one volume add up"
        );
    }
}
