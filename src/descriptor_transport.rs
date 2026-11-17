//! Shared descriptor transport contract: version, row layout and metadata.
//!
//! Shared verbatim by the producer crate and the lightweight native reader.

use anyhow::{Context, Result, ensure};
use parquet::file::metadata::KeyValue;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

/// Sentinel for unprinted J/coupling fields. Physical seniority is never missing.
///
/// Distinct from a printed `0` (`2J` or `2K` can both be
/// legitimately zero), so round-tripping preserves the "printed vs not"
/// distinction needed to restore the original CSF record.
pub const MISSING: i8 = -1;

/// GRASP stores angular momenta as 2J+1 in a signed byte.
pub const MAX_TWO_J: u16 = 126;
pub const DESCRIPTOR_DTYPE: &str = "int8";

/// Validate before narrowing; 127 fits i8 but violates GRASP's 2J+1 bound.
pub fn checked_angular_value(two_j: u16, field: &str) -> Result<i8> {
    ensure!(
        two_j <= MAX_TWO_J,
        "{field}={two_j} exceeds the descriptor angular bound 2J/2K <= {MAX_TWO_J} (GRASP |2J+1| <= 127)"
    );
    i8::try_from(two_j).context("descriptor angular value does not fit Int8")
}

pub fn validate_descriptor_storage(metadata: &[KeyValue]) -> Result<()> {
    let maximum = MAX_TWO_J.to_string();
    for (key, expected) in [
        ("descriptor_dtype", DESCRIPTOR_DTYPE),
        ("max_two_j", maximum.as_str()),
    ] {
        let mut entries = metadata.iter().filter(|entry| entry.key == key);
        ensure!(
            entries.next().and_then(|entry| entry.value.as_deref()) == Some(expected)
                && entries.next().is_none(),
            "descriptor file requires {key}={expected}; regenerate legacy Int32 descriptors from CSF text or CSF Parquet"
        );
    }
    Ok(())
}

/// Required marker separating physical-v descriptors from older printed-v files.
pub const SENIORITY_ENCODING: &str = "physical";

pub fn validate_seniority_encoding(metadata: &[KeyValue]) -> Result<()> {
    let mut entries = metadata
        .iter()
        .filter(|entry| entry.key == "seniority_encoding");
    ensure!(
        entries.next().and_then(|entry| entry.value.as_deref()) == Some(SENIORITY_ENCODING)
            && entries.next().is_none(),
        "descriptor file requires seniority_encoding=physical; regenerate legacy printed-seniority descriptors from CSF text or CSF Parquet"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DescriptorVersion {
    V2,
}

impl DescriptorVersion {
    /// Integer channels stored per peel subshell.
    pub const fn channels_per_subshell(self) -> usize {
        match self {
            Self::V2 => 4,
        }
    }

    /// The stable integer tag written to Parquet KV metadata and TOML sidecars.
    pub const fn tag(self) -> u8 {
        match self {
            Self::V2 => 2,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Err(anyhow::anyhow!(
                "V1 descriptors have been removed; regenerate V2 descriptors from the original CSF data"
            )),
            2 => Ok(Self::V2),
            other => Err(anyhow::anyhow!("unknown descriptor_version tag {other}")),
        }
    }
}

/// The fixed-width row shape for one descriptor version over `subshell_count`
/// peel subshells. Cheap to copy; construct it once per generator/export call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DescriptorLayout {
    version: DescriptorVersion,
    subshell_count: usize,
}

impl DescriptorLayout {
    pub fn new(version: DescriptorVersion, subshell_count: usize) -> Self {
        Self {
            version,
            subshell_count,
        }
    }

    pub fn version(self) -> DescriptorVersion {
        self.version
    }

    pub fn subshell_count(self) -> usize {
        self.subshell_count
    }

    pub fn channels_per_subshell(self) -> usize {
        self.version.channels_per_subshell()
    }

    /// Per-subshell feature columns: `channels_per_subshell * subshell_count`.
    pub fn feature_len(self) -> usize {
        self.channels_per_subshell() * self.subshell_count
    }

    /// Full row width including global columns (`4M+2`).
    pub fn row_len(self) -> usize {
        self.feature_len() + 2
    }

    /// Base offset of the feature block for peel subshell `index` (0-based).
    pub fn slot(self, index: usize) -> usize {
        index * self.channels_per_subshell()
    }

    pub fn n_offset(self) -> usize {
        0
    }

    pub fn two_j_offset(self) -> usize {
        1
    }

    /// V2-only: offset of the seniority channel within a subshell's slot.
    pub fn seniority_offset(self) -> usize {
        debug_assert_eq!(self.version, DescriptorVersion::V2);
        2
    }

    /// Offset of the printed-coupling (`2K`) channel within a subshell's slot.
    pub fn two_k_offset(self) -> usize {
        3
    }

    /// Index of the total angular momentum global column.
    pub fn total_two_j_index(self) -> Option<usize> {
        Some(self.feature_len())
    }

    /// Index of the parity global column.
    pub fn parity_index(self) -> Option<usize> {
        Some(self.feature_len() + 1)
    }

    /// Names of the per-subshell feature columns, in row order.
    ///
    /// Named feature columns let consumers exclude the global columns.
    pub fn feature_column_names(self) -> Vec<String> {
        (0..self.subshell_count)
            .flat_map(|index| {
                [
                    format!("sub{index}_n"),
                    format!("sub{index}_2j"),
                    format!("sub{index}_v"),
                    format!("sub{index}_2k"),
                ]
            })
            .collect()
    }

    /// Names of the global (non-per-subshell) columns, in row order.
    pub fn global_column_names(self) -> Vec<String> {
        vec!["total_two_j".to_owned(), "parity".to_owned()]
    }
}

/// Build the Arrow output schema for a descriptor Parquet file.
///
/// Descriptors retain raw integers; ML preprocessing belongs to the consumer.
pub fn output_schema(layout: DescriptorLayout) -> Result<Arc<arrow_schema::Schema>> {
    use arrow_schema::{DataType, Field, Schema};
    let fields = layout
        .feature_column_names()
        .into_iter()
        .chain(layout.global_column_names())
        .map(|name| Field::new(name, DataType::Int8, false))
        .collect::<Vec<_>>();
    Ok(Arc::new(Schema::new(fields)))
}

pub fn validate_output_schema(
    actual: &arrow_schema::Schema,
    layout: DescriptorLayout,
) -> Result<()> {
    let expected = output_schema(layout)?;
    ensure!(
        actual.fields().len() == expected.fields().len()
            && actual
                .fields()
                .iter()
                .zip(expected.fields())
                .all(|(actual, expected)| {
                    actual.name() == expected.name()
                        && actual.data_type() == expected.data_type()
                        && !actual.is_nullable()
                }),
        "descriptor schema must contain the ordered non-null Int8 V2 columns; regenerate legacy Int32 descriptors"
    );
    Ok(())
}

/// Lower-hex SHA-256 of a header TOML file's bytes.
///
/// This is the mechanism behind design doc §6's "header, blocks and
/// descriptors must match as a set" (plan D5): binding a descriptor file to
/// the exact header bytes it was generated from lets a restore path detect a
/// header that was regenerated or edited after the fact.
pub fn hash_header_file(header_path: &Path) -> Result<String> {
    let bytes = std::fs::read(header_path)
        .with_context(|| format!("failed to read header file {}", header_path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn escape_json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            _ => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

fn json_string_array(values: &[String]) -> String {
    let mut json = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&escape_json_string(value));
    }
    json.push(']');
    json
}

/// Build the Parquet key-value metadata pairs that make a descriptor file
/// format contract self-describing (design doc §6, plan D5 layer 2).
///
/// This metadata is immutable format fact only — never restoration data. The
/// optional `source_header_sha256`/`source_header_filename` bind a descriptor
/// file to the `{stem}_header.toml` it was generated from; omit them when no
/// such header exists yet (e.g. unit tests constructing a schema in isolation).
pub fn output_kv_metadata(
    layout: DescriptorLayout,
    peel_subshells: &[String],
    source_header_sha256: Option<&str>,
    source_header_filename: Option<&str>,
) -> Vec<KeyValue> {
    let mut entries = vec![
        KeyValue::new(
            "descriptor_dtype".to_owned(),
            Some(DESCRIPTOR_DTYPE.to_owned()),
        ),
        KeyValue::new("max_two_j".to_owned(), Some(MAX_TWO_J.to_string())),
        KeyValue::new(
            "descriptor_version".to_owned(),
            Some(layout.version().tag().to_string()),
        ),
        KeyValue::new(
            "channels_per_subshell".to_owned(),
            Some(layout.channels_per_subshell().to_string()),
        ),
        KeyValue::new(
            "subshell_count".to_owned(),
            Some(layout.subshell_count().to_string()),
        ),
        KeyValue::new(
            "peel_subshells".to_owned(),
            Some(json_string_array(peel_subshells)),
        ),
        KeyValue::new("missing_sentinel".to_owned(), Some(MISSING.to_string())),
        KeyValue::new(
            "seniority_encoding".to_owned(),
            Some(SENIORITY_ENCODING.to_owned()),
        ),
        KeyValue::new("normalized".to_owned(), Some("false".to_owned())),
        KeyValue::new(
            "feature_columns".to_owned(),
            Some(json_string_array(&layout.feature_column_names())),
        ),
        KeyValue::new(
            "global_columns".to_owned(),
            Some(json_string_array(&layout.global_column_names())),
        ),
    ];
    if let Some(hash) = source_header_sha256 {
        entries.push(KeyValue::new(
            "source_header_sha256".to_owned(),
            Some(hash.to_owned()),
        ));
    }
    if let Some(filename) = source_header_filename {
        entries.push(KeyValue::new(
            "source_header_filename".to_owned(),
            Some(filename.to_owned()),
        ));
    }
    entries
}
