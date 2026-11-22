//! Descriptor Parquet metadata and bounded writer accounting.

use anyhow::Result;

use crate::configuration_generation::shared_generation::{ResourceBudget, ResourcePermit};
use crate::descriptor_schema::DescriptorLayout;

/// Maximum rows buffered by the published Parquet writer.
pub(crate) const ROW_GROUP_ROWS: usize = 8_192;

pub(crate) mod descriptor {
    use std::path::Path;

    use anyhow::{Context, Result};
    use parquet::file::metadata::KeyValue;
    use parquet::file::properties::WriterProperties;

    use crate::descriptor_schema::{DescriptorLayout, hash_header_file, output_kv_metadata};

    use super::ROW_GROUP_ROWS;

    /// Bind published descriptors to the exact header body (including its block index).
    /// Header-less intermediate merges can omit the binding explicitly.
    pub(crate) fn metadata(
        layout: DescriptorLayout,
        peel_subshells: &[String],
        header_path: Option<&Path>,
    ) -> Result<Vec<KeyValue>> {
        let header_sha256 = header_path.map(hash_header_file).transpose()?;
        let header_filename = header_path
            .and_then(Path::file_name)
            .and_then(|name| name.to_str());
        Ok(output_kv_metadata(
            layout,
            peel_subshells,
            header_sha256.as_deref(),
            header_filename,
        ))
    }

    pub(crate) fn properties(
        layout: DescriptorLayout,
        peel_subshells: &[String],
        generated_count: usize,
        unique_count: usize,
        duplicate_count: usize,
        block_lengths: &[usize],
        header_path: Option<&Path>,
    ) -> Result<WriterProperties> {
        let mut metadata = metadata(layout, peel_subshells, header_path)?;
        metadata.extend([
            KeyValue::new(
                "generated_record_count".to_owned(),
                Some(generated_count.to_string()),
            ),
            KeyValue::new(
                "unique_record_count".to_owned(),
                Some(unique_count.to_string()),
            ),
            KeyValue::new(
                "duplicate_record_count".to_owned(),
                Some(duplicate_count.to_string()),
            ),
            KeyValue::new(
                "block_lengths".to_owned(),
                Some(format_usize_list(block_lengths)),
            ),
        ]);
        Ok(WriterProperties::builder()
            .set_compression(
                crate::csfs_descriptor::parquet_batch::parse_compression(None)
                    .expect("default compression is valid"),
            )
            .set_dictionary_enabled(true)
            .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
            .set_key_value_metadata(Some(metadata))
            .build())
    }

    pub(crate) fn writer_managed_bytes(layout: DescriptorLayout) -> Result<u64> {
        // Parquet INT(8,true) uses physical INT32 values. Keep the encoder
        // allowance wide even though Arrow/source descriptor buffers are i8.
        let bytes = layout
            .row_len()
            .checked_mul(ROW_GROUP_ROWS)
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(1 << 20))
            .context("Parquet writer managed byte count overflow")?;
        u64::try_from(bytes).context("Parquet writer bytes exceed u64")
    }

    fn format_usize_list(values: &[usize]) -> String {
        let body = values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!("[{body}]")
    }
}

pub(crate) fn descriptor_writer_permit(
    budget: &ResourceBudget,
    layout: DescriptorLayout,
    label: &str,
) -> Result<ResourcePermit> {
    budget.try_reserve(descriptor::writer_managed_bytes(layout)?, label)
}
