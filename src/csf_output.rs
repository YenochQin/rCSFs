//! Shared contracts for the two Parquet artifacts produced from generated CSFs.
//!
//! Both the reference two-pass path and the one-pass final encoder use these
//! definitions.  Keeping the row-group bound beside the writer-memory model is
//! important: the allowance is only true while the writer is prevented from
//! buffering more rows than it prices.

use anyhow::Result;

use crate::csf_generation::{ResourceBudget, ResourcePermit};
use crate::descriptor_schema::DescriptorLayout;

/// Maximum rows buffered by either published Parquet writer.
pub(crate) const ROW_GROUP_ROWS: usize = 8_192;

pub(crate) mod descriptor {
    use anyhow::{Context, Result};
    use parquet::file::metadata::KeyValue;
    use parquet::file::properties::WriterProperties;

    use crate::descriptor_schema::{DescriptorLayout, output_kv_metadata};

    use super::ROW_GROUP_ROWS;

    pub(crate) fn properties(
        layout: DescriptorLayout,
        peel_subshells: &[String],
        generated_count: usize,
        unique_count: usize,
        duplicate_count: usize,
        block_lengths: &[usize],
    ) -> Result<WriterProperties> {
        let mut metadata = output_kv_metadata(layout, peel_subshells, false, None, None);
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

pub(crate) mod csf_parquet {
    use std::fs::File;
    use std::sync::Arc;

    use anyhow::{Context, Result};
    use arrow::array::{ArrayRef, StringBuilder, UInt64Builder};
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use crate::descriptor_schema::DescriptorLayout;

    use super::ROW_GROUP_ROWS;

    pub(crate) fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("idx", DataType::UInt64, false),
            Field::new("line1", DataType::Utf8, false),
            Field::new("line2", DataType::Utf8, false),
            Field::new("line3", DataType::Utf8, false),
        ]))
    }

    pub(crate) fn properties() -> WriterProperties {
        WriterProperties::builder()
            .set_compression(parquet::basic::Compression::UNCOMPRESSED)
            .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
            .build()
    }

    pub(crate) fn writer(file: File) -> Result<ArrowWriter<File>> {
        ArrowWriter::try_new(file, schema(), Some(properties()))
            .context("failed to create CSF Parquet writer")
    }

    pub(crate) struct BatchBuilder {
        schema: SchemaRef,
        indices: UInt64Builder,
        lines: [StringBuilder; 3],
    }

    impl BatchBuilder {
        pub(crate) fn new(schema: SchemaRef, capacity: usize) -> Self {
            Self {
                schema,
                indices: UInt64Builder::with_capacity(capacity),
                lines: [
                    StringBuilder::new(),
                    StringBuilder::new(),
                    StringBuilder::new(),
                ],
            }
        }

        pub(crate) fn push(&mut self, index: u64, lines: &[String; 3]) {
            self.indices.append_value(index);
            for (builder, line) in self.lines.iter_mut().zip(lines) {
                builder.append_value(line);
            }
        }

        pub(crate) fn finish(mut self) -> Result<RecordBatch> {
            let arrays: Vec<ArrayRef> = vec![
                Arc::new(self.indices.finish()),
                Arc::new(self.lines[0].finish()),
                Arc::new(self.lines[1].finish()),
                Arc::new(self.lines[2].finish()),
            ];
            RecordBatch::try_new(self.schema, arrays)
                .context("failed to construct the CSF Parquet batch")
        }
    }

    pub(crate) fn writer_managed_bytes(layout: DescriptorLayout) -> Result<u64> {
        let fields = layout.row_len() / 4;
        let line_bytes = fields
            .checked_mul(crate::complete_csf::FIELD_WIDTH)
            .and_then(|bytes| bytes.checked_add(2))
            .context("CSF line width overflow")?;
        let per_row = line_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .context("CSF Parquet row width overflow")?;
        let rows_bytes = ROW_GROUP_ROWS
            .checked_mul(per_row)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_add(1 << 20))
            .context("CSF Parquet writer byte count overflow")?;
        u64::try_from(rows_bytes).context("CSF Parquet writer bytes exceed u64")
    }
}

pub(crate) fn descriptor_writer_permit(
    budget: &ResourceBudget,
    layout: DescriptorLayout,
    label: &str,
) -> Result<ResourcePermit> {
    budget.try_reserve(descriptor::writer_managed_bytes(layout)?, label)
}

pub(crate) fn csf_parquet_writer_permit(
    budget: &ResourceBudget,
    layout: DescriptorLayout,
    label: &str,
) -> Result<ResourcePermit> {
    budget.try_reserve(csf_parquet::writer_managed_bytes(layout)?, label)
}
