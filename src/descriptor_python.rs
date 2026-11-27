//! Python interfaces for the descriptor-native data model.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Int8Array, Int32Builder, ListBuilder, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3_arrow::PyRecordBatchReader;
use serde::Serialize;
use serde_json::json;

use crate::descriptor_dataset::{self as dataset, DescriptorRow, DescriptorRows};
use crate::descriptor_operations as operations;
use crate::descriptor_v3::decode_v3_into;

pub(super) fn python_value(py: Python<'_>, value: &impl Serialize) -> PyResult<Py<PyAny>> {
    let text = serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.import("json")?.call_method1("loads", (text,))?.unbind())
}

fn io_error(error: anyhow::Error) -> PyErr {
    PyIOError::new_err(format!("{error:#}"))
}

fn companion(path: &str, explicit: Option<String>) -> PyResult<PathBuf> {
    explicit.map(PathBuf::from).map_or_else(
        || dataset::header_path(Path::new(path)).map_err(io_error),
        Ok,
    )
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, header_path=None))]
fn convert_csfs(
    py: Python<'_>,
    input_path: String,
    output_path: String,
    header_path: Option<String>,
) -> PyResult<Py<PyAny>> {
    let header = companion(&output_path, header_path)?;
    let stats = py
        .detach(|| dataset::import_text(Path::new(&input_path), Path::new(&output_path), &header))
        .map_err(io_error)?;
    python_value(py, &stats)
}

#[pyfunction]
#[pyo3(signature = (input_path, header_path=None))]
fn read_zero_order_counts(
    py: Python<'_>,
    input_path: String,
    header_path: Option<String>,
) -> PyResult<Option<Vec<usize>>> {
    let header = companion(&input_path, header_path)?;
    py.detach(|| {
        DescriptorRows::open(Path::new(&input_path), &header)?
            .header()
            .zero_order_counts()
    })
    .map_err(io_error)
}

#[pyfunction]
#[pyo3(signature = (input_path, header_path=None, include_block_id=false, include_coupling_signature=false))]
fn read_csfs_arrow(
    py: Python<'_>,
    input_path: String,
    header_path: Option<String>,
    include_block_id: bool,
    include_coupling_signature: bool,
) -> PyResult<(Py<PyAny>, PyRecordBatchReader)> {
    let header_path = companion(&input_path, header_path)?;
    let (header, batch) = py
        .detach(|| -> anyhow::Result<_> {
            let mut reader = DescriptorRows::open(Path::new(&input_path), &header_path)?;
            let layout = reader.header().layout();
            let mut columns = vec![Vec::<i8>::new(); layout.row_len()];
            let mut indices = Vec::new();
            let mut blocks = Vec::new();
            let mut signatures = ListBuilder::new(Int32Builder::new());
            let (mut occupied, mut couplings) = (Vec::new(), Vec::new());
            while let Some(DescriptorRow {
                ordinal,
                block_index,
                values: row,
            }) = reader.next_descriptor_row()?
            {
                indices.push(u64::try_from(ordinal)?);
                if include_block_id {
                    blocks.push(u32::try_from(block_index)?);
                }
                for (column, value) in columns.iter_mut().zip(&row) {
                    column.push(*value);
                }
                if include_coupling_signature {
                    let (total_two_j, _) = decode_v3_into(
                        &row,
                        layout,
                        &reader.header().peel,
                        &mut occupied,
                        &mut couplings,
                    )?;
                    let mut printed_couplings = couplings.iter().peekable();
                    for (index, shell) in occupied
                        .iter()
                        .enumerate()
                        .take(occupied.len().saturating_sub(1))
                    {
                        let two_j = if let Some(step) = printed_couplings
                            .next_if(|step| usize::from(step.boundary) == index + 1)
                        {
                            step.two_j
                        } else {
                            shell.state.map_or(0, |state| state.two_j)
                        };
                        signatures.values().append_value(i32::from(two_j));
                    }
                    signatures.values().append_value(i32::from(total_two_j));
                    signatures.append(true);
                }
            }
            let mut fields = vec![Arc::new(Field::new("idx", DataType::UInt64, false))];
            fields.extend(
                crate::descriptor_schema::output_schema(layout)?
                    .fields()
                    .iter()
                    .cloned(),
            );
            let mut arrays: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from(indices))];
            arrays.extend(
                columns
                    .into_iter()
                    .map(|v| Arc::new(Int8Array::from(v)) as ArrayRef),
            );
            if include_block_id {
                fields.push(Arc::new(Field::new("block_id", DataType::UInt32, false)));
                arrays.push(Arc::new(UInt32Array::from(blocks)));
            }
            if include_coupling_signature {
                use arrow::array::Array;
                let array = signatures.finish();
                fields.push(Arc::new(Field::new(
                    "coupling_signature",
                    array.data_type().clone(),
                    false,
                )));
                arrays.push(Arc::new(array));
            }
            let header = reader.into_header().document;
            Ok((
                header,
                RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?,
            ))
        })
        .map_err(io_error)?;
    let schema = batch.schema();
    Ok((
        python_value(py, &header)?,
        PyRecordBatchReader::new(Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema))),
    ))
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, indices, header_path=None, output_header=None))]
fn select_csfs(
    py: Python<'_>,
    input_path: String,
    output_path: String,
    indices: Vec<u64>,
    header_path: Option<String>,
    output_header: Option<String>,
) -> PyResult<Py<PyAny>> {
    let source_header = companion(&input_path, header_path)?;
    let target_header = companion(&output_path, output_header)?;
    let stats = py
        .detach(|| {
            dataset::select(
                Path::new(&input_path),
                &source_header,
                Path::new(&output_path),
                &target_header,
                &indices,
            )
        })
        .map_err(io_error)?;
    python_value(py, &stats)
}

#[pyfunction]
#[pyo3(signature = (input_paths, output_path, header_paths=None, output_header=None))]
fn merge_csfs(
    py: Python<'_>,
    input_paths: Vec<String>,
    output_path: String,
    header_paths: Option<Vec<String>>,
    output_header: Option<String>,
) -> PyResult<Py<PyAny>> {
    if header_paths
        .as_ref()
        .is_some_and(|h| h.len() != input_paths.len())
    {
        return Err(PyValueError::new_err("input/header path counts differ"));
    }
    let inputs = input_paths
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let header = companion(&p, header_paths.as_ref().map(|v| v[i].clone()))?;
            Ok((PathBuf::from(p), header))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let header = companion(&output_path, output_header)?;
    let stats = py
        .detach(|| operations::merge(&inputs, Path::new(&output_path), &header))
        .map_err(io_error)?;
    python_value(py, &stats)
}

#[pyfunction]
#[pyo3(signature = (full_path, selected_path, full_header=None, selected_header=None, require_all=true))]
fn match_csfs(
    py: Python<'_>,
    full_path: String,
    selected_path: String,
    full_header: Option<String>,
    selected_header: Option<String>,
    require_all: bool,
) -> PyResult<Vec<Option<u64>>> {
    let full_header = companion(&full_path, full_header)?;
    let selected_header = companion(&selected_path, selected_header)?;
    py.detach(|| {
        operations::match_indices(
            Path::new(&full_path),
            &full_header,
            Path::new(&selected_path),
            &selected_header,
            require_all,
        )
    })
    .map_err(io_error)
}

#[pyfunction]
#[pyo3(signature = (descriptor_parquet, header_path, output, indices=None))]
fn restore_csfs_from_descriptors(
    py: Python<'_>,
    descriptor_parquet: String,
    header_path: String,
    output: String,
    indices: Option<Vec<u64>>,
) -> PyResult<Py<PyAny>> {
    let stats = py
        .detach(|| {
            dataset::restore_text_with_stats(
                Path::new(&descriptor_parquet),
                Path::new(&header_path),
                Path::new(&output),
                indices.as_deref(),
            )
        })
        .map_err(io_error)?;
    python_value(
        py,
        &json!({"success":true,"output_file":output,"record_count":stats.record_count,"output_bytes":stats.output_bytes}),
    )
}

#[pyfunction]
fn partition_csfs(
    py: Python<'_>,
    zero_parquet: String,
    zero_header: String,
    full_parquet: String,
    full_header: String,
    output_path: String,
) -> PyResult<Py<PyAny>> {
    let output_header = companion(&output_path, None)?;
    let stats = py
        .detach(|| {
            operations::zero_first(
                Path::new(&zero_parquet),
                Path::new(&zero_header),
                Path::new(&full_parquet),
                Path::new(&full_header),
                Path::new(&output_path),
                &output_header,
            )
        })
        .map_err(io_error)?;
    python_value(py, &stats)
}

#[pyfunction]
fn split_csfs_by_active_spaces(
    py: Python<'_>,
    input_parquet: String,
    header_path: String,
    targets: Vec<(String, String)>,
) -> PyResult<Py<PyAny>> {
    if targets.is_empty() {
        return Err(PyValueError::new_err("no active-space targets"));
    }
    let specs = targets
        .into_iter()
        .map(|(output, space)| {
            let output = PathBuf::from(output);
            Ok((output.clone(), dataset::header_path(&output)?, space))
        })
        .collect::<anyhow::Result<Vec<_>>>()
        .map_err(io_error)?;
    let outputs = py
        .detach(|| {
            operations::split_active_spaces(
                Path::new(&input_parquet),
                Path::new(&header_path),
                &specs,
            )
        })
        .map_err(io_error)?;
    python_value(py, &json!({"success":true,"outputs":outputs}))
}

#[pyfunction]
#[pyo3(signature = (input_path, output_dir=None, prefix=None, header_path=None))]
fn split_csfs_by_j(
    py: Python<'_>,
    input_path: String,
    output_dir: Option<String>,
    prefix: Option<String>,
    header_path: Option<String>,
) -> PyResult<Py<PyAny>> {
    let header = companion(&input_path, header_path)?;
    let input = Path::new(&input_path);
    let directory = output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| input.parent().unwrap_or(Path::new(".")).to_path_buf());
    let prefix = prefix.unwrap_or_else(|| {
        dataset::descriptor_stem(input.file_stem().and_then(|s| s.to_str()).unwrap_or("csfs"))
            .to_owned()
    });
    let outputs = py
        .detach(|| operations::split_blocks(input, &header, &directory, &prefix))
        .map_err(io_error)?;
    python_value(py, &json!({"success":true,"outputs":outputs}))
}

#[pyfunction]
#[pyo3(signature = (transcript, output_path, threads=None, storage="disk", memory_budget_mib=None, allow_unchecked_space=false, *, scratch_dir=None))]
#[allow(clippy::too_many_arguments)]
fn generate_csfs_from_transcript(
    py: Python<'_>,
    transcript: &Bound<'_, PyAny>,
    output_path: String,
    threads: Option<usize>,
    storage: &str,
    memory_budget_mib: Option<usize>,
    allow_unchecked_space: bool,
    scratch_dir: Option<PathBuf>,
) -> PyResult<Py<PyAny>> {
    use crate::configuration_generation;
    use crate::configuration_generation::csf_generation::disk_csf_generation_pipeline as disk;
    use crate::configuration_generation::shared_generation::GenerationOptions;
    let texts = if let Ok(text) = transcript.extract::<String>() {
        vec![text]
    } else {
        transcript.extract::<Vec<String>>()?
    };
    let output = Path::new(&output_path);
    let header = companion(&output_path, None)?;
    let result = py.detach(|| -> anyhow::Result<_> {
        anyhow::ensure!(threads != Some(0), "threads must be positive");
        crate::atomic_output::ensure_new_output(output)?;
        crate::atomic_output::ensure_new_output(&header)?;
        let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let work = tempfile::Builder::new().prefix(".rcsfs-generation-").tempdir_in(parent)?;
        let staged = work.path().join(output.file_name().ok_or_else(|| anyhow::anyhow!("missing output filename"))?);
        let staged_header = dataset::header_path(&staged)?;
        let result = if storage == "disk" {
            // The benchmark owns an explicit scratch path so it can time its removal.
            let scratch = scratch_dir.unwrap_or_else(|| work.path().join("scratch"));
            let options = GenerationOptions::from_api_with_space_policy(threads, memory_budget_mib, Some(scratch), allow_unchecked_space)?;
            let stats = disk::generate_descriptor_dataset(&texts, &staged, &staged_header, &options)?;
            json!({
                "success": true, "output_file": output_path, "header_file": header,
                "record_count": stats.unique_count, "block_count": stats.block_count,
                "blocks": stats.blocks, "unique_occupations": stats.unique_occupations,
                "descriptor_count": stats.unique_count, "descriptor_bytes": stats.descriptor_bytes,
                "generated_count": stats.generated_count, "duplicate_count": stats.duplicate_count,
                "segment_codec": stats.segment_codec, "deduplication": stats.deduplication,
                "stage_stats": stats.stage_stats, "resource_stats": stats.resource_stats,
                "plan_stats": stats.plan_stats,
            })
        } else {
            anyhow::ensure!(storage == "memory" && texts.len() == 1 && memory_budget_mib.is_none() && scratch_dir.is_none(), "memory generation requires one transcript and no disk memory budget or scratch directory");
            let stats = configuration_generation::generate_csfs_from_transcript(&texts[0], &staged, threads)?;
            json!({"success":true,"output_file":output_path,"header_file":header,"record_count":stats.record_count,"descriptor_count":stats.record_count,"block_count":stats.block_count,"blocks":stats.blocks,"unique_occupations":stats.unique_occupations})
        };
        crate::atomic_output::publish_staged_outputs(&[(&staged_header, &header), (&staged, output)], false)?;
        Ok(result)
    }).map_err(io_error)?;
    python_value(py, &result)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(convert_csfs, module)?)?;
    module.add_function(wrap_pyfunction!(read_csfs_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(read_zero_order_counts, module)?)?;
    module.add_function(wrap_pyfunction!(select_csfs, module)?)?;
    module.add_function(wrap_pyfunction!(merge_csfs, module)?)?;
    module.add_function(wrap_pyfunction!(match_csfs, module)?)?;
    module.add_function(wrap_pyfunction!(restore_csfs_from_descriptors, module)?)?;
    module.add_function(wrap_pyfunction!(partition_csfs, module)?)?;
    module.add_function(wrap_pyfunction!(split_csfs_by_active_spaces, module)?)?;
    module.add_function(wrap_pyfunction!(split_csfs_by_j, module)?)?;
    module.add_function(wrap_pyfunction!(generate_csfs_from_transcript, module)?)?;
    Ok(())
}
