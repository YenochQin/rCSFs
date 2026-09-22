//! One ordered pass over the generated segments that writes every final
//! artifact: the descriptor Parquet, the CSF text and the three-line CSF
//! Parquet.
//!
//! The pre-P4 tail wrote the descriptor Parquet and then read it back to format
//! the CSF text, so every row was encoded once, decoded once and formatted once
//! across two full passes over disk. This pass reads each surviving row from its
//! segment exactly once: its integers go straight into the descriptor Parquet,
//! and its decoded record is validated and formatted for the CSF text and CSF
//! Parquet.
//!
//! Two things this module has to get right beyond producing the right bytes:
//!
//! * **Accounting before allocation.** The managed budget is charged for the
//!   structures a batch will hold *before* they exist and for what they really
//!   are, including both live Parquet writers. A reserve that happens after the
//!   allocation, or that counts survivors while the batch was sized for every
//!   row, is a budget that reports a peak lower than the process ever held.
//! * **Complete phase timing.** The phases are named after the operations they
//!   contain, and the time to finish each one — the writer footers, the text
//!   flush, publication — is charged to it. The phases therefore add up to the
//!   whole pass, which is what makes any claim about them checkable.

use anyhow::{Context, Result, ensure};
use arrow::array::{Array, Int32Array, UInt32Array, UInt64Array};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::FileReader;
use parquet::arrow::arrow_writer::ArrowWriter;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::streaming::{
    DeduplicatedSegments, SurvivorBitset, csf_parquet_writer_permit, descriptor_output_properties,
    grouped_segments, parquet_writer_permit, process_cpu_millis,
};
use crate::atomic_output::{create_temporary_output, publish_temporary_output};
use crate::descriptor_schema::{output_schema, validate_record};
use crate::descriptor_v2::decode_v2_into;

/// Wall and CPU time accumulated over the interleaved sections of one phase.
///
/// The pass interleaves its phases batch by batch — prepare a batch, encode it
/// into the descriptor Parquet, write it out — so a phase cannot be timed with
/// one start/finish pair the way a whole stage can. Each section's wall and
/// rusage delta is accumulated instead.
#[derive(Default)]
pub(crate) struct PhaseAccumulator {
    nanos: u128,
    cpu_millis: Option<u128>,
}

impl PhaseAccumulator {
    /// Run one section of this phase, adding its wall and CPU time.
    fn add<R>(&mut self, section: impl FnOnce() -> R) -> R {
        let started = Instant::now();
        let cpu_before = process_cpu_millis();
        let result = section();
        self.nanos = self.nanos.saturating_add(started.elapsed().as_nanos());
        if let (Some(before), Some(after)) = (cpu_before, process_cpu_millis()) {
            self.cpu_millis = Some(
                self.cpu_millis
                    .unwrap_or(0)
                    .saturating_add(after.saturating_sub(before)),
            );
        }
        result
    }

    pub(crate) fn elapsed_millis(&self) -> u128 {
        self.nanos / 1_000_000
    }

    fn cpu_millis(&self) -> Option<u128> {
        self.cpu_millis
    }

    /// This phase's wall time minus the part spent inside its writer's own
    /// `write` calls: the conversion and compression, without the I/O.
    fn compute_millis(&self, io_nanos: &AtomicU64) -> u128 {
        self.elapsed_millis().saturating_sub(io_millis(io_nanos))
    }
}

/// A writer that accumulates the time spent inside the writes it performs.
///
/// This is how the pass separates "encode" from "write" without an API fight:
/// the Parquet writer decides when to encode and when to emit, and everything
/// it hands to the underlying writer is real output I/O. Time inside those
/// calls is charged to the write side; the rest of the phase is conversion,
/// compression and bookkeeping.
struct CountedWriter<W> {
    inner: W,
    io_nanos: Arc<AtomicU64>,
}

impl<W> CountedWriter<W> {
    fn new(inner: W, io_nanos: Arc<AtomicU64>) -> Self {
        Self { inner, io_nanos }
    }
}

impl<W: Write> Write for CountedWriter<W> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let started = Instant::now();
        let result = self.inner.write(buffer);
        self.io_nanos
            .fetch_add(elapsed_nanos(started), Ordering::Relaxed);
        result
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let started = Instant::now();
        let result = self.inner.flush();
        self.io_nanos
            .fetch_add(elapsed_nanos(started), Ordering::Relaxed);
        result
    }
}

fn io_millis(io_nanos: &AtomicU64) -> u128 {
    u128::from(io_nanos.load(Ordering::Relaxed)) / 1_000_000
}

fn elapsed_nanos(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// What the pass produced, with each phase timed over every section that
/// belongs to it — including the sections that finish it.
///
/// `descriptor` covers the descriptor Parquet's conversion, compression, writes
/// and footer; `descriptor_io` is the share of it spent inside the file writes.
/// `csf_outputs` covers the CSF text and CSF Parquet the same way, including the
/// text flush and the publication of both staged files.
pub(crate) struct FinalEncodingStats {
    pub(crate) record_count: usize,
    pub(crate) descriptor_bytes: u64,
    pub(crate) csf_bytes: u64,
    pub(crate) csf_parquet_bytes: u64,
    /// Filtering each batch's rows by the survivor bitset and gathering the
    /// selected columns: serial, and the only phase that touches every row.
    pub(crate) select: PhaseAccumulator,
    pub(crate) prepare: PhaseAccumulator,
    pub(crate) descriptor: PhaseAccumulator,
    pub(crate) descriptor_io_nanos: Arc<AtomicU64>,
    pub(crate) csf_outputs: PhaseAccumulator,
    pub(crate) csf_outputs_io_nanos: Arc<AtomicU64>,
}

impl FinalEncodingStats {
    /// The five numbers the pipeline reports, in the order they happen.
    ///
    /// Each phase is split into the time its writer spent writing and the rest;
    /// the two halves sum to the phase, so the five entries sum to the pass.
    /// CPU is reported on the compute side only: `getrusage` cannot attribute a
    /// syscall's CPU to the caller that made it, and inventing a split would be
    /// worse than saying so.
    pub(crate) fn phase_entries(&self) -> [(&'static str, u128, Option<u128>); 6] {
        [
            (
                "final_encoding_select",
                self.select.elapsed_millis(),
                self.select.cpu_millis(),
            ),
            (
                "final_encoding_prepare",
                self.prepare.elapsed_millis(),
                self.prepare.cpu_millis(),
            ),
            (
                "final_encoding_descriptor_encode",
                self.descriptor.compute_millis(&self.descriptor_io_nanos),
                self.descriptor.cpu_millis(),
            ),
            (
                "final_encoding_descriptor_write",
                io_millis(&self.descriptor_io_nanos),
                None,
            ),
            (
                "final_encoding_csf_outputs_encode",
                self.csf_outputs.compute_millis(&self.csf_outputs_io_nanos),
                self.csf_outputs.cpu_millis(),
            ),
            (
                "final_encoding_csf_outputs_write",
                io_millis(&self.csf_outputs_io_nanos),
                None,
            ),
        ]
    }
}

/// Build every final artifact from the segments in one ordered pass.
///
/// Preparation (decode, validate, format) runs in the thread pool over each
/// batch's rows; the descriptor Parquet still has a single writer and the CSF
/// text a single pen. Block separators and the global `idx` sequence come from
/// this ordered side, so no batch or thread boundary can put a row in the wrong
/// block.
pub(crate) fn build_final_outputs_from_segments(
    deduplicated: &DeduplicatedSegments,
    header_lines: &[String; 5],
    descriptor_output: &Path,
    csf_output: &Path,
    csf_parquet_output: &Path,
    threads: Option<usize>,
) -> Result<FinalEncodingStats> {
    super::planning::run_parallel(threads, || {
        build_final_outputs_from_segments_inner(
            deduplicated,
            header_lines,
            descriptor_output,
            csf_output,
            csf_parquet_output,
        )
    })
}

fn build_final_outputs_from_segments_inner(
    deduplicated: &DeduplicatedSegments,
    header_lines: &[String; 5],
    descriptor_output: &Path,
    csf_output: &Path,
    csf_parquet_output: &Path,
) -> Result<FinalEncodingStats> {
    for (path, label) in [
        (descriptor_output, "descriptor output"),
        (csf_output, "CSF output"),
        (csf_parquet_output, "CSF Parquet output"),
    ] {
        ensure!(!path.exists(), "{label} already exists: {}", path.display());
    }
    let layout = deduplicated.layout;
    let row_len = layout.row_len();
    let peel_subshells: &[String] = &deduplicated.peel_subshells;

    // Both writers are alive for the whole pass, so both are charged for it,
    // before either one exists.
    let _descriptor_writer_permit =
        parquet_writer_permit(&deduplicated.budget, layout, "final descriptor encoding")?;
    let _csf_parquet_writer_permit =
        csf_parquet_writer_permit(&deduplicated.budget, layout, "CSF Parquet encoding")?;

    let schema = output_schema(layout, false)?;
    let properties = descriptor_output_properties(
        layout,
        peel_subshells,
        deduplicated.generated_count,
        deduplicated.unique_count,
        deduplicated.duplicate_count,
        &deduplicated.block_lengths,
    )?;
    let (descriptor_temporary, descriptor_file) = create_temporary_output(descriptor_output)?;
    let descriptor_io_nanos = Arc::new(AtomicU64::new(0));
    let mut descriptor_writer = ArrowWriter::try_new(
        CountedWriter::new(descriptor_file, Arc::clone(&descriptor_io_nanos)),
        schema.clone(),
        Some(properties),
    )
    .context("failed to create the final descriptor Parquet writer")?;

    let (text_temporary, text_file) = create_temporary_output(csf_output)?;
    let csf_outputs_io_nanos = Arc::new(AtomicU64::new(0));
    let mut text_writer = BufWriter::new(CountedWriter::new(
        text_file,
        Arc::clone(&csf_outputs_io_nanos),
    ));
    for line in header_lines {
        writeln!(text_writer, "{line}")?;
    }

    let csf_schema = crate::csfs_descriptor::csf_parquet::schema();
    let csf_parquet_file = File::options()
        .write(true)
        .create_new(true)
        .open(csf_parquet_output)
        .with_context(|| {
            format!(
                "failed to create CSF Parquet {}",
                csf_parquet_output.display()
            )
        })?;
    let mut csf_parquet_writer = ArrowWriter::try_new(
        CountedWriter::new(csf_parquet_file, Arc::clone(&csf_outputs_io_nanos)),
        csf_schema.clone(),
        Some(crate::csfs_descriptor::csf_parquet::properties()),
    )
    .context("failed to create CSF Parquet writer")?;

    let mut select = PhaseAccumulator::default();
    let mut prepare = PhaseAccumulator::default();
    let mut descriptor = PhaseAccumulator::default();
    let mut csf_outputs = PhaseAccumulator::default();

    // The widest line this layout can format, so the text buffers are charged
    // for what they can hold rather than for a guess.
    let fields_per_record = row_len / 4;
    let max_line_bytes = fields_per_record
        .checked_mul(crate::complete_csf::FIELD_WIDTH)
        .and_then(|bytes| bytes.checked_add(2))
        .context("formatted line width overflow")?;

    let mut blocks = grouped_segments(&deduplicated.segments);
    let mut record_count = 0usize;
    let mut block_lengths = Vec::with_capacity(blocks.len());
    for (block_index, (&key, segments)) in blocks.iter_mut().enumerate() {
        segments.sort_by_key(|segment| (segment.range_ordinal, segment.local_start));
        // The separator belongs between blocks, and this side is the ordered
        // one: emitting it here cannot depend on how batches were scheduled.
        if block_index > 0 {
            csf_outputs
                .add(|| writeln!(text_writer, " *"))
                .context("failed to write the CSF block separator")?;
        }
        let bitset_file = deduplicated
            .survivor_bitsets
            .get(&key)
            .context("missing survivor bitset for descriptor block")?;
        let bitset = SurvivorBitset::read(bitset_file)?;
        let mut ordinal = 0u64;
        let mut block_count = 0usize;
        for segment in segments {
            let file = File::open(&segment.path)
                .with_context(|| format!("failed to open segment {}", segment.path.display()))?;
            let reader = FileReader::try_new(file, None).with_context(|| {
                format!("failed to read Arrow segment {}", segment.path.display())
            })?;
            let expected_columns = row_len
                .checked_add(2)
                .context("segment column count overflow")?;
            let mut next_local = segment.local_start;
            for batch in reader {
                let batch = batch.with_context(|| {
                    format!("failed to decode segment {}", segment.path.display())
                })?;
                ensure!(
                    batch.num_columns() == expected_columns,
                    "segment {} has {} columns, expected {expected_columns}",
                    segment.path.display(),
                    batch.num_columns()
                );
                let range_column = batch
                    .column(row_len)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .context("segment range_ordinal column is not UInt32")?;
                let local_column = batch
                    .column(row_len + 1)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .context("segment local_ordinal column is not UInt64")?;
                let data_columns = batch.columns()[..row_len]
                    .iter()
                    .map(|column| {
                        column
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .context("segment descriptor column is not Int32")
                    })
                    .collect::<Result<Vec<_>>>()?;

                // The filter pass keeps the ordinal accounting the merge stage
                // performed, so a bitset or segment disagreement fails here
                // exactly as it would have failed the merge.
                let (survivors, mut selected_columns) = select.add(|| {
                    let mut survivors: Vec<usize> = Vec::new();
                    for row in 0..batch.num_rows() {
                        ensure!(
                            range_column.value(row) == segment.range_ordinal,
                            "segment range ordinal does not match file metadata"
                        );
                        ensure!(
                            local_column.value(row) == next_local,
                            "segment local ordinal is not contiguous"
                        );
                        next_local += 1;
                        if bitset.contains(ordinal)? {
                            survivors.push(row);
                        }
                        ordinal = ordinal
                            .checked_add(1)
                            .context("survivor ordinal overflow")?;
                    }
                    // Sized for exactly the rows that survive, so the columns
                    // never hold a batch's worth of memory the reservation did
                    // not cover.
                    let mut selected_columns = (0..row_len)
                        .map(|_| Vec::with_capacity(survivors.len()))
                        .collect::<Vec<_>>();
                    for (selected, column) in selected_columns.iter_mut().zip(&data_columns) {
                        for &row in &survivors {
                            selected.push(column.value(row));
                        }
                    }
                    Ok::<_, anyhow::Error>((survivors, selected_columns))
                })?;
                if survivors.is_empty() {
                    continue;
                }

                // Charge for what this batch is about to hold, before it holds
                // it: the selected columns sized for exactly the surviving rows,
                // the formatted three-line records, and the per-thread decode
                // scratch. The Arrow arrays the descriptor batch is built from
                // are the selected columns themselves — moved, never cloned —
                // so they add nothing here; the two writers were charged above.
                let threads = rayon::current_num_threads().max(1);
                let batch_bytes =
                    batch_managed_bytes(row_len, survivors.len(), threads, max_line_bytes)?;
                let _batch_permit = deduplicated
                    .budget
                    .try_reserve(batch_bytes, "final-encoding batch")?;

                // Prepare every surviving row in parallel: decode, validate and
                // format. The publication side stays in row order below, so the
                // parallel section changes who formats a row, not where it lands.
                let formatted = prepare.add(|| {
                    survivors
                        .par_iter()
                        .map(|&row| {
                            let mut values = vec![0i32; row_len];
                            for (value, column) in values.iter_mut().zip(&data_columns) {
                                *value = column.value(row);
                            }
                            let mut occupied = Vec::new();
                            let mut couplings = Vec::new();
                            let (total_two_j, parity) =
                                decode_v2_into(&values, layout, &mut occupied, &mut couplings)?;
                            validate_record(peel_subshells, &occupied, &couplings)?;
                            let (line1, line2, line3) =
                                crate::complete_csf::CompleteCsfFile::format_record_parts(
                                    peel_subshells,
                                    &occupied,
                                    &couplings,
                                    total_two_j,
                                    parity,
                                )?;
                            Ok([line1, line2, line3])
                        })
                        .collect::<Result<Vec<[String; 3]>>>()
                })?;

                descriptor.add(|| {
                    let arrays: Vec<Arc<dyn Array>> = selected_columns
                        .drain(..)
                        .map(|column| Arc::new(Int32Array::from(column)) as Arc<dyn Array>)
                        .collect();
                    let batch = RecordBatch::try_new(schema.clone(), arrays)
                        .context("failed to construct the final descriptor batch")?;
                    descriptor_writer.write(&batch).with_context(|| {
                        format!(
                            "failed to write the final descriptor Parquet for segment {}",
                            segment.path.display()
                        )
                    })
                })?;

                csf_outputs.add(|| {
                    let mut builder = crate::csfs_descriptor::csf_parquet::BatchBuilder::new(
                        csf_schema.clone(),
                        formatted.len(),
                    );
                    for (position, record_lines) in formatted.iter().enumerate() {
                        writeln!(text_writer, "{}", record_lines[0])?;
                        writeln!(text_writer, "{}", record_lines[1])?;
                        writeln!(text_writer, "{}", record_lines[2])?;
                        builder.push(u64::try_from(record_count + position)?, record_lines)?;
                    }
                    let batch = builder.finish()?;
                    csf_parquet_writer
                        .write(&batch)
                        .context("failed to write the CSF Parquet batch")
                })?;

                record_count = record_count
                    .checked_add(formatted.len())
                    .context("final-encoding record count overflow")?;
                block_count = block_count
                    .checked_add(formatted.len())
                    .context("final-encoding block count overflow")?;
                // Releasing the batch's formatted records is real work — three
                //Strings per row, tens of millions of them on a large input —
                // and it belongs to a phase. Left to the end of the iteration it
                // would appear as an unexplained gap between the stages and the
                // call the caller times.
                csf_outputs.add(|| drop(formatted));
            }
        }
        ensure!(
            usize::try_from(ordinal).ok() == Some(bitset.bit_len),
            "survivor bitset length does not match source block"
        );
        ensure!(
            block_count == bitset_file.unique_count,
            "final pass selected {block_count} rows for a block whose bitset kept {}",
            bitset_file.unique_count
        );
        block_lengths.push(block_count);
    }
    ensure!(
        record_count == deduplicated.unique_count,
        "final encoding wrote {record_count} rows, expected {}",
        deduplicated.unique_count
    );
    ensure!(
        block_lengths == deduplicated.block_lengths,
        "final encoding block lengths do not match de-duplication result"
    );

    // Finishing is part of the phases, not a free epilogue: the footers and the
    // flush are I/O that the phase's writer performs, and publication is I/O
    // the phase's outputs require.
    csf_outputs
        .add(|| {
            text_writer
                .flush()
                .context("failed to flush the CSF text writer")
        })
        .context("failed to flush the CSF text writer")?;
    drop(text_writer);
    csf_outputs
        .add(|| {
            csf_parquet_writer
                .close()
                .context("failed to close the CSF Parquet writer")
        })
        .context("failed to close the CSF Parquet writer")?;
    descriptor
        .add(|| {
            descriptor_writer
                .close()
                .context("failed to close the final descriptor Parquet writer")
        })
        .context("failed to close the final descriptor Parquet writer")?;
    csf_outputs
        .add(|| {
            let started = Instant::now();
            let result = publish_temporary_output(text_temporary.path(), csf_output, false);
            csf_outputs_io_nanos.fetch_add(elapsed_nanos(started), Ordering::Relaxed);
            result
        })
        .context("failed to publish the CSF text")?;
    descriptor
        .add(|| {
            let started = Instant::now();
            let result =
                publish_temporary_output(descriptor_temporary.path(), descriptor_output, false);
            descriptor_io_nanos.fetch_add(elapsed_nanos(started), Ordering::Relaxed);
            result
        })
        .context("failed to publish the descriptor")?;

    let descriptor_bytes = std::fs::metadata(descriptor_output)?.len();
    let csf_bytes = std::fs::metadata(csf_output)?.len();
    let csf_parquet_bytes = std::fs::metadata(csf_parquet_output)?.len();
    Ok(FinalEncodingStats {
        record_count,
        select,
        descriptor_bytes,
        csf_bytes,
        csf_parquet_bytes,
        prepare,
        descriptor,
        descriptor_io_nanos,
        csf_outputs,
        csf_outputs_io_nanos,
    })
}

/// The managed bytes one batch of this pass may hold at once.
///
/// Sized for the structures that really coexist, and reserved before any of
/// them is allocated: the surviving rows' columns (which become the descriptor
/// batch's Arrow arrays by move), the formatted three-line records, and one
/// decode scratch per worker thread.
fn batch_managed_bytes(
    row_len: usize,
    survivors: usize,
    threads: usize,
    max_line_bytes: usize,
) -> Result<u64> {
    let columns = survivors
        .checked_mul(row_len)
        .and_then(|ints| ints.checked_mul(std::mem::size_of::<i32>()))
        .context("final-encoding column bytes overflow")?;
    // Three lines per record plus the `String` headers, and a NUL-free bound on
    // each line, so the charge cannot be smaller than what formatting produces.
    let text_per_record = max_line_bytes
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(3 * std::mem::size_of::<String>()))
        .context("formatted record byte bound overflow")?;
    let formatted = survivors
        .checked_mul(text_per_record)
        .context("formatted text bytes overflow")?;
    let survivor_indices = survivors
        .checked_mul(std::mem::size_of::<usize>())
        .context("survivor index bytes overflow")?;
    let scratch_per_thread = row_len
        .checked_mul(std::mem::size_of::<i32>())
        .and_then(|bytes| bytes.checked_add(4096))
        .context("decode scratch byte count overflow")?;
    let scratch = threads
        .checked_mul(scratch_per_thread)
        .context("decode scratch total overflow")?;
    let total = columns
        .checked_add(formatted)
        .and_then(|bytes| bytes.checked_add(survivor_indices))
        .and_then(|bytes| bytes.checked_add(scratch))
        .context("final-encoding batch managed byte count overflow")?;
    u64::try_from(total).context("final-encoding batch bytes exceed u64")
}
