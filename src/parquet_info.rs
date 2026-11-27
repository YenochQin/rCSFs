//! Read-only Parquet metadata inspection.
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde::Serialize;
use std::fs::File;
use std::path::Path;

/// Runtime-independent Parquet metadata shared by Python and the native CLI.
#[derive(Debug, Serialize)]
pub struct ParquetInfo {
    /// Input path rendered as UTF-8, with invalid byte sequences replaced.
    pub file_path: String,
    /// Total size of the Parquet file in bytes.
    pub file_size: u64,
    /// Row count recorded in the Parquet footer.
    pub num_rows: i64,
    /// Number of top-level fields in the Parquet schema.
    pub num_columns: usize,
    /// Compression codec of the first column in the first row group, or `Unknown`.
    pub compression: String,
    /// Parquet writer identification from the footer, or `Unknown` when absent.
    pub created_by: String,
    /// Footer metadata keyed by entry name. `None` represents a key with no value.
    pub key_value_metadata: std::collections::BTreeMap<String, Option<String>>,
}

/// Read metadata without loading the Parquet rows.
pub fn read_parquet_info(
    parquet_path: &Path,
) -> Result<ParquetInfo, Box<dyn std::error::Error + Send + Sync>> {
    let file_size = std::fs::metadata(parquet_path)?.len();
    let reader = SerializedFileReader::new(File::open(parquet_path)?)?;
    let metadata = reader.metadata();
    Ok(ParquetInfo {
        file_path: parquet_path.to_string_lossy().into_owned(),
        file_size,
        num_rows: metadata.file_metadata().num_rows(),
        num_columns: metadata.file_metadata().schema().get_fields().len(),
        compression: metadata
            .row_groups()
            .first()
            .and_then(|r| r.columns().first())
            .map(|column| format!("{:?}", column.compression()))
            .unwrap_or_else(|| "Unknown".into()),
        created_by: metadata
            .file_metadata()
            .created_by()
            .unwrap_or("Unknown")
            .into(),
        key_value_metadata: metadata
            .file_metadata()
            .key_value_metadata()
            .into_iter()
            .flatten()
            .map(|entry| (entry.key.clone(), entry.value.clone()))
            .collect(),
    })
}
