//! Counted workload planning for the disk generation path.
//!
//! Scheduling used to divide configurations into fixed-size blocks, so a range
//! task could carry anywhere between a handful and tens of millions of CSFs.
//! This module counts what each configuration will produce first and then cuts
//! the work into tasks of comparable size.
//!
//! Two invariants govern the whole module:
//!
//! * A plan is a partition of the configuration list and of each configuration's
//!   generation order. Concatenating a plan's tasks in ordinal order reproduces
//!   the unsplit record order exactly, so scheduling never changes the result.
//! * Every count uses checked arithmetic. An estimate that overflowed to zero
//!   would silently excuse a task from the workload it really carries, so
//!   overflow is reported as an error instead.

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::ops::Range;

use super::{
    EnumeratedOccupations, ExcitationRequest, PreparedCounter, SubshellOccupation,
    count_configuration_records, prepare_configuration, state_prefixes,
};

/// Smallest estimate one task is allowed to carry.
///
/// The target comes from the total and the thread count, so this only binds on
/// small inputs — and it must not bind hard enough to leave the pool idle. On
/// the registered B2 input at 8 threads, a 250,000 floor produced 3 tasks and
/// 5.55 s end to end, while 10,000-50,000 produced 11-56 tasks and 4.56-4.82 s,
/// almost all of the difference in the generation stage. A 32,768 floor keeps
/// the task count bounded for small inputs (at most `records / 32768`) without
/// starving a 8-10 thread pool.
const RECORDS_PER_TASK_LOWER_BOUND: u64 = 32_768;

/// Upper bound of one task's estimated record count.
///
/// A V2 row is roughly one kilobyte of uncompressed Arrow, so this caps a
/// task's segment data at a few gigabytes rather than letting a single task
/// grow without limit on a large input.
const RECORDS_PER_TASK_UPPER_BOUND: u64 = 4_000_000;

/// How many tasks each worker thread should have to choose from.
const TASKS_PER_THREAD: u64 = 4;

/// Largest state-prefix split applied to a single oversized configuration.
///
/// The split is a scheduling device, not a storage device: a configuration
/// whose chain count stays above the target even at this width is reported as
/// unsplittable instead of being cut further.
const MAX_STATE_PREFIX_BRANCHES: usize = 4096;

/// Environment override for the task size target, for benchmark experiments.
/// It is intentionally not a public option: the documented interface exposes
/// only user-meaningful settings.
const RECORDS_PER_TASK_ENV: &str = "RCSFS_RECORDS_PER_TASK";

/// What one planned task generates.
///
/// The variants are ordered the same way the serial generator walks the work,
/// so a plan's task ordinals are a total order over the original record order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TaskSpan {
    /// Whole configurations `start..end`, each over the full 2J range.
    Configurations { start: usize, end: usize },
    /// One configuration restricted to an ascending subset of its 2J targets.
    Targets {
        configuration: usize,
        targets: Vec<u16>,
    },
    /// One configuration, one 2J target, and a slice of the state-prefix tree.
    StatePrefixes {
        configuration: usize,
        target: u16,
        branch_count: usize,
        branches: Range<usize>,
    },
}

impl TaskSpan {
    /// Configurations this task touches, as a half-open range.
    pub(crate) fn configurations(&self) -> Range<usize> {
        match self {
            Self::Configurations { start, end } => *start..*end,
            Self::Targets { configuration, .. } | Self::StatePrefixes { configuration, .. } => {
                *configuration..*configuration + 1
            }
        }
    }
}

/// One scheduling unit with its counted workload.
#[derive(Clone, Debug)]
pub(crate) struct PlannedTask {
    /// Position in the plan's total order. It is the range ordinal the disk
    /// writer stamps on segments, so it must never depend on completion order.
    pub(crate) ordinal: u32,
    pub(crate) span: TaskSpan,
    pub(crate) estimated_records: u64,
}

/// A task that stays above the size target because the planner cannot divide it
/// any further.
///
/// `target` is the 2J the work belongs to, or `None` when the whole
/// configuration is indivisible — which happens when its intermediate
/// couplings can exceed the output field, so no state-prefix count exists for
/// it.
#[derive(Clone, Debug)]
pub(crate) struct UnsplittableWork {
    pub(crate) configuration: usize,
    pub(crate) target: Option<u16>,
    pub(crate) estimated_records: u64,
}

/// Distribution of one quantity across the plan, for benchmark reporting.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RecordDistribution {
    pub(crate) count: usize,
    pub(crate) total: u64,
    pub(crate) minimum: u64,
    pub(crate) p50: u64,
    pub(crate) p95: u64,
    pub(crate) maximum: u64,
}

impl RecordDistribution {
    fn from_samples(mut samples: Vec<u64>) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        samples.sort_unstable();
        let total = samples
            .iter()
            .try_fold(0u64, |total, &value| total.checked_add(value))
            .unwrap_or(u64::MAX);
        let pick = |numerator: usize| samples[(samples.len() - 1) * numerator / 100];
        Self {
            count: samples.len(),
            total,
            minimum: samples[0],
            p50: pick(50),
            p95: pick(95),
            maximum: samples[samples.len() - 1],
        }
    }
}

/// The counted workload of a request, before any scheduling decision.
#[derive(Clone, Debug)]
pub(crate) struct WorkloadEstimate {
    /// Estimated pre-deduplication records per enumerated configuration, in
    /// enumeration order, so index `i` is `occupations.configurations[i]`.
    pub(crate) configuration_records: Vec<u64>,
    pub(crate) total_records: u64,
    pub(crate) zero_record_configurations: usize,
    pub(crate) unique_occupations: usize,
}

/// A complete schedule plus the measurements that justified it.
#[derive(Clone, Debug)]
pub(crate) struct GenerationPlan {
    pub(crate) tasks: Vec<PlannedTask>,
    pub(crate) workload: WorkloadEstimate,
    pub(crate) target_records_per_task: u64,
    pub(crate) unsplittable: Vec<UnsplittableWork>,
}

impl GenerationPlan {
    pub(crate) fn task_record_distribution(&self) -> RecordDistribution {
        RecordDistribution::from_samples(
            self.tasks.iter().map(|task| task.estimated_records).collect(),
        )
    }

    /// Reporting view carried with the generation result.
    pub(crate) fn stats(&self) -> PlanStats {
        PlanStats {
            task_count: self.tasks.len(),
            target_records_per_task: self.target_records_per_task,
            estimated_records_per_task: self.task_record_distribution(),
            estimated_total_records: self.workload.total_records,
            unique_occupations: self.workload.unique_occupations,
            zero_record_configurations: self.workload.zero_record_configurations,
            unsplittable_tasks: self.unsplittable.len(),
            unsplittable_records: self
                .unsplittable
                .iter()
                .map(|work| work.estimated_records)
                .max()
                .unwrap_or(0),
        }
    }
}

/// Counted workload and scheduling outcome, exposed with a generation result.
///
/// The estimates are pre-deduplication record counts. They are not a promise:
/// the generated total is reported separately and the two are compared by the
/// benchmark so a drifting counter is visible instead of silent.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanStats {
    pub(crate) task_count: usize,
    pub(crate) target_records_per_task: u64,
    pub(crate) estimated_records_per_task: RecordDistribution,
    pub(crate) estimated_total_records: u64,
    pub(crate) unique_occupations: usize,
    pub(crate) zero_record_configurations: usize,
    pub(crate) unsplittable_tasks: usize,
    pub(crate) unsplittable_records: u64,
}

/// Print the scheduling summary that accompanies generation progress.
pub(crate) fn report_plan(plan: &GenerationPlan) {
    let stats = plan.stats();
    let distribution = stats.estimated_records_per_task;
    eprintln!(
        "Planned {} tasks from {} configurations: estimated {} CSFs before de-duplication",
        stats.task_count, stats.unique_occupations, stats.estimated_total_records
    );
    eprintln!(
        "Task size target {} CSFs; per-task p50 {} p95 {} max {}",
        stats.target_records_per_task,
        distribution.p50,
        distribution.p95,
        distribution.maximum
    );
    if stats.zero_record_configurations > 0 {
        eprintln!(
            "{} configurations are expected to produce no CSF",
            stats.zero_record_configurations
        );
    }
    if stats.unsplittable_tasks > 0 {
        eprintln!(
            "{} tasks stay above the {} CSF size target because neither their configuration \
             nor their state prefixes can be divided further; largest {} CSFs",
            stats.unsplittable_tasks, stats.target_records_per_task, stats.unsplittable_records
        );
    }
}

/// Count what every enumerated configuration is expected to produce.
///
/// The configuration count and the record totals are the inputs of both the
/// task planner and the disk-space pre-check, so they are computed once by the
/// caller that needs them.
pub(crate) fn estimate_workload(
    request: &ExcitationRequest,
    occupations: &EnumeratedOccupations,
    threads: Option<usize>,
) -> Result<WorkloadEstimate> {
    let targets = request_targets(request)?;
    let counted = run_parallel(threads, || {
        occupations
            .configurations
            .par_iter()
            .map(|configuration| {
                count_configuration_records(
                    &occupations.core_subshells,
                    &configuration.occupations,
                    request.min_two_j,
                    request.max_two_j,
                    &targets,
                )
            })
            .collect::<Result<Vec<u64>>>()
    })?;
    let total_records = counted
        .iter()
        .try_fold(0u64, |total, &value| total.checked_add(value))
        .context("total estimated record count overflow")?;
    let zero_record_configurations = counted.iter().filter(|&&value| value == 0).count();
    Ok(WorkloadEstimate {
        unique_occupations: counted.len(),
        configuration_records: counted,
        total_records,
        zero_record_configurations,
    })
}

/// The 2J values a request asks for, in ascending order.
pub(crate) fn request_targets(request: &ExcitationRequest) -> Result<Vec<u16>> {
    ensure!(
        request.min_two_j <= request.max_two_j,
        "minimum 2J exceeds maximum 2J"
    );
    let span = u32::from(request.max_two_j) - u32::from(request.min_two_j);
    let count = usize::try_from(span / 2 + 1).context("2J target count exceeds usize")?;
    Ok((0..count)
        .map(|index| request.min_two_j + 2 * u16::try_from(index).expect("count fits u16"))
        .collect())
}

/// Choose the record budget of one task.
///
/// A fixed size would either under-fill the thread pool on small inputs or
/// produce oversized segment files on large ones, so the target is derived from
/// the counted total and clamped to a safe range. An explicit request wins, and
/// the environment override exists so a benchmark can hold the schedule fixed
/// while it varies something else.
pub(crate) fn choose_task_target(
    total_records: u64,
    threads: Option<usize>,
    requested: Option<u64>,
) -> Result<u64> {
    if let Some(value) = requested {
        ensure!(value > 0, "records_per_task must be greater than 0");
        return Ok(value);
    }
    if let Ok(value) = std::env::var(RECORDS_PER_TASK_ENV) {
        let parsed = value
            .trim()
            .parse::<u64>()
            .with_context(|| format!("invalid {RECORDS_PER_TASK_ENV} {value:?}"))?;
        ensure!(parsed > 0, "{RECORDS_PER_TASK_ENV} must be greater than 0");
        return Ok(parsed);
    }
    let workers = threads.unwrap_or_else(rayon::current_num_threads).max(1);
    let workers = u64::try_from(workers).context("worker count exceeds u64")?;
    let share = total_records / workers.saturating_mul(TASKS_PER_THREAD).max(1);
    Ok(share.clamp(
        RECORDS_PER_TASK_LOWER_BOUND,
        RECORDS_PER_TASK_UPPER_BOUND,
    ))
}

/// Cut the counted workload into tasks of comparable size.
///
/// Configurations below the target are grouped into multi-configuration tasks.
/// A configuration above it is first split by 2J target and then, if a single
/// target is still too large, by state prefix. Work that cannot be split far
/// enough is reported in [`GenerationPlan::unsplittable`] rather than being
/// hidden inside a large task.
pub(crate) fn plan_generation(
    request: &ExcitationRequest,
    occupations: &EnumeratedOccupations,
    workload: WorkloadEstimate,
    threads: Option<usize>,
    requested_target: Option<u64>,
) -> Result<GenerationPlan> {
    ensure!(
        workload.configuration_records.len() == occupations.configurations.len(),
        "workload estimate does not match the enumerated configurations"
    );
    let target = choose_task_target(workload.total_records, threads, requested_target)?;
    let targets = request_targets(request)?;
    let mut tasks: Vec<PlannedTask> = Vec::new();
    let mut unsplittable: Vec<UnsplittableWork> = Vec::new();
    // Configurations at or below the target accumulate here until the next one
    // would push the task past the target.
    let mut span_start: Option<usize> = None;
    let mut span_records = 0u64;

    for (index, &records) in workload.configuration_records.iter().enumerate() {
        if records <= target && records.saturating_add(span_records) <= target {
            span_start.get_or_insert(index);
            span_records = span_records
                .checked_add(records)
                .context("task record estimate overflow")?;
            continue;
        }
        if records <= target {
            close_span(&mut tasks, &mut span_start, &mut span_records, index)?;
            span_start = Some(index);
            span_records = records;
            continue;
        }
        close_span(&mut tasks, &mut span_start, &mut span_records, index)?;
        split_configuration(
            request,
            occupations,
            index,
            records,
            &targets,
            target,
            &mut tasks,
            &mut unsplittable,
        )?;
    }
    close_span(
        &mut tasks,
        &mut span_start,
        &mut span_records,
        workload.configuration_records.len(),
    )?;
    ensure!(
        !tasks.is_empty(),
        "the generation plan contains no tasks for {} configurations",
        occupations.configurations.len()
    );
    for (ordinal, task) in tasks.iter_mut().enumerate() {
        task.ordinal = u32::try_from(ordinal).context("too many generation tasks")?;
    }
    Ok(GenerationPlan {
        tasks,
        workload,
        target_records_per_task: target,
        unsplittable,
    })
}

fn close_span(
    tasks: &mut Vec<PlannedTask>,
    span_start: &mut Option<usize>,
    span_records: &mut u64,
    end: usize,
) -> Result<()> {
    let Some(start) = span_start.take() else {
        return Ok(());
    };
    ensure!(
        start < end,
        "a task span must contain at least one configuration"
    );
    let estimated_records = std::mem::take(span_records);
    tasks.push(PlannedTask {
        ordinal: 0,
        span: TaskSpan::Configurations { start, end },
        estimated_records,
    });
    Ok(())
}

/// Divide one oversized configuration into target and state-prefix tasks.
#[allow(clippy::too_many_arguments)]
fn split_configuration(
    request: &ExcitationRequest,
    occupations: &EnumeratedOccupations,
    index: usize,
    records: u64,
    targets: &[u16],
    target_size: u64,
    tasks: &mut Vec<PlannedTask>,
    unsplittable: &mut Vec<UnsplittableWork>,
) -> Result<()> {
    let configuration: &[SubshellOccupation] = &occupations.configurations[index].occupations;
    let prepared = prepare_configuration(
        &occupations.core_subshells,
        configuration,
        request.min_two_j,
        request.max_two_j,
    )?;
    let counter = PreparedCounter::new(&prepared);
    // The dynamic program cannot see the generator's per-selection `FIRST`
    // flag, so a configuration whose intermediate couplings can exceed the
    // output field is counted by the generator instead. That configuration must
    // still be scheduled: refusing it here would fail a plan the workload
    // counter already accepted, so it is split by 2J target only and reported
    // as a long tail rather than rejected.
    let exact = counter.chain_count_is_exact();
    let first_task = tasks.len();
    let mut pending: Vec<u16> = Vec::new();
    let mut pending_records = 0u64;
    for &target in targets {
        let count = if exact {
            counter.count(target, &[])?
        } else {
            count_configuration_records(
                &occupations.core_subshells,
                configuration,
                request.min_two_j,
                request.max_two_j,
                &[target],
            )?
        };
        if count == 0 {
            continue;
        }
        if count > target_size {
            flush_targets(tasks, index, &mut pending, &mut pending_records)?;
            if exact {
                split_target_by_prefix(
                    &prepared,
                    &counter,
                    index,
                    target,
                    count,
                    target_size,
                    tasks,
                    unsplittable,
                )?;
            } else {
                // One target cannot be divided further without prefix counts.
                unsplittable.push(UnsplittableWork {
                    configuration: index,
                    target: Some(target),
                    estimated_records: count,
                });
                tasks.push(PlannedTask {
                    ordinal: 0,
                    span: TaskSpan::Targets {
                        configuration: index,
                        targets: vec![target],
                    },
                    estimated_records: count,
                });
            }
            continue;
        }
        if pending_records.saturating_add(count) > target_size {
            flush_targets(tasks, index, &mut pending, &mut pending_records)?;
        }
        pending.push(target);
        pending_records = pending_records
            .checked_add(count)
            .context("configuration target estimate overflow")?;
    }
    flush_targets(tasks, index, &mut pending, &mut pending_records)?;
    // Every record of an oversized configuration must end up in exactly one of
    // its tasks. This is the boundary where a split can silently drop work:
    // a group of merged prefixes that publishes only its last prefix, or a
    // target that is counted but never scheduled.
    let scheduled = tasks[first_task..]
        .iter()
        .try_fold(0u64, |total, task| total.checked_add(task.estimated_records))
        .context("configuration schedule total overflow")?;
    ensure!(
        scheduled == records,
        "configuration {index} scheduled {scheduled} records but was counted at {records}"
    );
    Ok(())
}

fn flush_targets(
    tasks: &mut Vec<PlannedTask>,
    configuration: usize,
    pending: &mut Vec<u16>,
    pending_records: &mut u64,
) -> Result<()> {
    if pending.is_empty() {
        ensure!(
            *pending_records == 0,
            "record estimate without a target cannot be scheduled"
        );
        return Ok(());
    }
    tasks.push(PlannedTask {
        ordinal: 0,
        span: TaskSpan::Targets {
            configuration,
            targets: std::mem::take(pending),
        },
        estimated_records: std::mem::take(pending_records),
    });
    Ok(())
}

/// Split one (configuration, target) pair along the state-prefix tree.
///
/// The prefix width is the smallest power-of-two split that brings the largest
/// prefix under the target. Work that stays above the target is still scheduled
/// — dropping it would lose records — and is additionally reported in
/// [`GenerationPlan::unsplittable`] so a long tail is visible instead of hidden
/// in the summary.
#[allow(clippy::too_many_arguments)]
fn split_target_by_prefix(
    prepared: &super::PreparedGeneration,
    counter: &PreparedCounter,
    configuration: usize,
    target: u16,
    count: u64,
    target_size: u64,
    tasks: &mut Vec<PlannedTask>,
    unsplittable: &mut Vec<UnsplittableWork>,
) -> Result<()> {
    let wanted = count.div_ceil(target_size.max(1));
    let mut branch_count = 2usize;
    while u64::try_from(branch_count).context("state-prefix width exceeds u64")? < wanted
        && branch_count < MAX_STATE_PREFIX_BRANCHES
    {
        branch_count *= 2;
    }
    let prefixes = state_prefixes(prepared, branch_count);
    ensure!(
        !prefixes.is_empty(),
        "a state-prefix split must contain at least one prefix"
    );
    if prefixes.len() < 2 {
        // The state tree is too narrow to divide. The pair still has to be
        // generated, so it becomes one whole-target task.
        unsplittable.push(UnsplittableWork {
            configuration,
            target: Some(target),
            estimated_records: count,
        });
        tasks.push(PlannedTask {
            ordinal: 0,
            span: TaskSpan::Targets {
                configuration,
                targets: vec![target],
            },
            estimated_records: count,
        });
        return Ok(());
    }
    let mut pending = 0usize;
    let mut pending_records = 0u64;
    for (prefix_index, prefix) in prefixes.iter().enumerate() {
        let prefix_records = counter.count(target, prefix)?;
        if prefix_records == 0 {
            continue;
        }
        if pending_records > 0 && pending_records.saturating_add(prefix_records) > target_size {
            push_prefix_task(
                tasks,
                configuration,
                target,
                branch_count,
                pending..prefix_index,
                pending_records,
            );
            pending_records = 0;
        }
        // `pending` is the first prefix of the open group and must only move
        // when a group starts. Advancing it on every prefix would publish a
        // range that begins at the group's last prefix while claiming the
        // records of all of them, silently dropping the earlier ones.
        if pending_records == 0 {
            pending = prefix_index;
        }
        pending_records = pending_records
            .checked_add(prefix_records)
            .context("state-prefix record estimate overflow")?;
        if pending_records > target_size {
            // A single prefix is already over target and cannot be divided
            // further: keep it as its own task and report the long tail.
            push_prefix_task(
                tasks,
                configuration,
                target,
                branch_count,
                prefix_index..prefix_index + 1,
                pending_records,
            );
            unsplittable.push(UnsplittableWork {
                configuration,
                target: Some(target),
                estimated_records: pending_records,
            });
            pending = prefix_index + 1;
            pending_records = 0;
        }
    }
    if pending_records > 0 {
        push_prefix_task(
            tasks,
            configuration,
            target,
            branch_count,
            pending..prefixes.len(),
            pending_records,
        );
    }
    Ok(())
}

fn push_prefix_task(
    tasks: &mut Vec<PlannedTask>,
    configuration: usize,
    target: u16,
    branch_count: usize,
    branches: Range<usize>,
    estimated_records: u64,
) {
    tasks.push(PlannedTask {
        ordinal: 0,
        span: TaskSpan::StatePrefixes {
            configuration,
            target,
            branch_count,
            branches,
        },
        estimated_records,
    });
}

/// Run `work` on a private pool when the caller pinned the thread count.
pub(crate) fn run_parallel<T: Send>(
    threads: Option<usize>,
    work: impl FnOnce() -> Result<T> + Send,
) -> Result<T>
where
    T: Send,
{
    match threads {
        Some(count) => rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .build()
            .context("failed to build the generation thread pool")?
            .install(work),
        None => work(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csf_generation::Subshell;

    /// A state-prefix split must cover every non-empty prefix exactly once for
    /// *every* task size.
    ///
    /// Merging several small prefixes into one task is where a split can lose
    /// records: a group published as `first..last` while claiming the records of
    /// the whole group drops every prefix before the first. Sweeping the task
    /// size exercises the merge, the single-prefix overflow and the whole-target
    /// fallback against one counted total.
    #[test]
    fn a_state_prefix_split_covers_every_prefix_at_every_task_size() {
        let configuration = [SubshellOccupation {
            subshell: "5g".parse::<Subshell>().unwrap(),
            electrons: 4,
        }];
        let prepared = prepare_configuration(&[], &configuration, 0, 0).unwrap();
        let counter = PreparedCounter::new(&prepared);
        assert!(counter.chain_count_is_exact());
        let prefixes = state_prefixes(&prepared, 4);
        assert!(prefixes.len() >= 4, "the split needs several prefixes");
        let counts = prefixes
            .iter()
            .map(|prefix| counter.count(0, prefix).unwrap())
            .collect::<Vec<_>>();
        let total: u64 = counts.iter().sum();
        assert!(total > 0);
        // At least one task size must merge two non-empty prefixes, or the
        // sweep would not exercise the path it exists to protect.
        let mut merged = false;

        for target_size in 1..=total {
            let mut tasks = Vec::new();
            let mut unsplittable = Vec::new();
            split_target_by_prefix(
                &prepared,
                &counter,
                0,
                0,
                total,
                target_size,
                &mut tasks,
                &mut unsplittable,
            )
            .unwrap();

            let scheduled = tasks
                .iter()
                .try_fold(0u64, |sum, task| sum.checked_add(task.estimated_records))
                .unwrap();
            assert_eq!(
                scheduled, total,
                "task size {target_size} scheduled {scheduled} of {total} records"
            );

            let mut covered = vec![0usize; prefixes.len()];
            for task in &tasks {
                let TaskSpan::StatePrefixes { branches, .. } = &task.span else {
                    panic!("a prefix split must only emit prefix tasks");
                };
                for index in branches.clone() {
                    covered[index] += 1;
                }
                if branches.len() > 1 {
                    merged = true;
                }
            }
            for (index, &count) in counts.iter().enumerate() {
                if count > 0 {
                    // A non-empty prefix that no task walks is work the plan
                    // counted and then dropped.
                    assert_eq!(
                        covered[index], 1,
                        "prefix {index} holds {count} records but was covered {} times \
                         at task size {target_size}",
                        covered[index]
                    );
                } else {
                    // An empty prefix may fall inside a neighbouring group's
                    // range; walking it emits nothing.
                    assert!(
                        covered[index] <= 1,
                        "prefix {index} was covered {} times at task size {target_size}",
                        covered[index]
                    );
                }
            }
        }
        assert!(merged, "the sweep never merged two prefixes");
    }

    /// The distribution summary is the reporting surface for the plan, so its
    /// percentiles have to stay on real samples rather than interpolations.
    #[test]
    fn record_distribution_reports_registered_percentiles() {
        let distribution = RecordDistribution::from_samples((1..=100).collect());
        assert_eq!(distribution.count, 100);
        assert_eq!(distribution.total, 5050);
        assert_eq!(distribution.minimum, 1);
        assert_eq!(distribution.p50, 50);
        assert_eq!(distribution.p95, 95);
        assert_eq!(distribution.maximum, 100);
        assert_eq!(RecordDistribution::from_samples(Vec::new()).count, 0);
    }

    #[test]
    fn task_target_stays_inside_the_supported_range() {
        assert_eq!(
            choose_task_target(0, Some(8), None).unwrap(),
            RECORDS_PER_TASK_LOWER_BOUND
        );
        assert_eq!(
            choose_task_target(u64::MAX, Some(1), None).unwrap(),
            RECORDS_PER_TASK_UPPER_BOUND
        );
        assert_eq!(choose_task_target(1, Some(8), Some(7)).unwrap(), 7);
        assert!(choose_task_target(1, Some(8), Some(0)).is_err());
    }

    /// The size target must not leave the pool idle on a small input.
    ///
    /// A floor that is too high is invisible in the plan (it is a valid size)
    /// but shows up as a task count below the thread count.
    #[test]
    fn a_small_input_still_yields_more_tasks_than_threads() {
        let threads = 8usize;
        let total = 560_351; // the registered B2 input
        let target = choose_task_target(total, Some(threads), None).unwrap();
        let tasks = total.div_ceil(target);
        assert!(
            tasks >= threads as u64,
            "{tasks} tasks for {threads} threads at a {target}-record target"
        );
    }
}
