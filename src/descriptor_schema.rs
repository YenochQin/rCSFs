//! Shared descriptor format contract: version, row layout and record validation.
//!
//! This is the single authority for descriptor shape and legality (design doc
//! `csf_descriptor_v2_ml_design.md` §3.1-§3.3). Both descriptor producers
//! (`complete_csf::CompleteCsfFile::descriptor_for` and
//! `csfs_descriptor::CSFDescriptorGenerator`) validate through
//! [`validate_record`] so a legality rule can never drift between the two.

use anyhow::{Context, Result, ensure};
use parquet::file::metadata::KeyValue;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

use crate::complete_csf::{IntermediateCoupling, OccupiedSubshell};
use crate::csf_generation::{Subshell, subshell_states};

/// Sentinel written to a V2 field that GRASP never printed for that record.
///
/// Distinct from a printed `0` (occupation, `2J`, seniority or `2K` can all be
/// legitimately zero), so round-tripping preserves the "printed vs not"
/// distinction that V1's dense zero-fill silently discarded.
pub const MISSING: i32 = -1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DescriptorVersion {
    V1,
    V2,
}

impl DescriptorVersion {
    /// Integer channels stored per peel subshell.
    pub const fn channels_per_subshell(self) -> usize {
        match self {
            Self::V1 => 3,
            Self::V2 => 4,
        }
    }

    /// The stable integer tag written to Parquet KV metadata and TOML sidecars.
    pub const fn tag(self) -> u8 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::V1),
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

    /// Full row width including global columns (`3M` for V1, `4M+2` for V2).
    pub fn row_len(self) -> usize {
        match self.version {
            DescriptorVersion::V1 => self.feature_len(),
            DescriptorVersion::V2 => self.feature_len() + 2,
        }
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
        match self.version {
            DescriptorVersion::V1 => 2,
            DescriptorVersion::V2 => 3,
        }
    }

    /// V2-only global column: absent (`None`) for V1, where the total is
    /// folded into the last occupied subshell's third field instead.
    pub fn total_two_j_index(self) -> Option<usize> {
        match self.version {
            DescriptorVersion::V1 => None,
            DescriptorVersion::V2 => Some(self.feature_len()),
        }
    }

    /// V2-only global column: absent (`None`) for V1, which never records parity.
    pub fn parity_index(self) -> Option<usize> {
        match self.version {
            DescriptorVersion::V1 => None,
            DescriptorVersion::V2 => Some(self.feature_len() + 1),
        }
    }

    /// Names of the per-subshell feature columns, in row order.
    ///
    /// V1 keeps its historical positional `col_{i}` names: nothing consumes
    /// them by name, and introducing named V1 columns would just add a second
    /// column-mapping surface. V2 uses named columns because
    /// `iter_indexed_descriptor_batches` must exclude the global columns
    /// without computing a stride.
    pub fn feature_column_names(self) -> Vec<String> {
        match self.version {
            DescriptorVersion::V1 => (0..self.feature_len())
                .map(|index| format!("col_{index}"))
                .collect(),
            DescriptorVersion::V2 => (0..self.subshell_count)
                .flat_map(|index| {
                    [
                        format!("sub{index}_n"),
                        format!("sub{index}_2j"),
                        format!("sub{index}_v"),
                        format!("sub{index}_2k"),
                    ]
                })
                .collect(),
        }
    }

    /// Names of the global (non-per-subshell) columns, in row order.
    pub fn global_column_names(self) -> Vec<String> {
        match self.version {
            DescriptorVersion::V1 => Vec::new(),
            DescriptorVersion::V2 => vec!["total_two_j".to_owned(), "parity".to_owned()],
        }
    }
}

/// Validate that `occupied`/`couplings` are a legal CSF record over `subshells`.
///
/// This checks legality against GRASP's physical state tables, not
/// generator reachability: an imported file may legally print a state the
/// generator's own visibility rule (`mod.rs`'s `FIRST`-flag suppression)
/// would never emit, and that must still validate. Checks performed:
///
/// - every occupied subshell index resolves into `subshells`;
/// - occupied subshells appear in ascending (Peel) order with no duplicates;
/// - `1 <= occupation <= capacity`;
/// - a printed `(2J, seniority)` state is a member of the subshell's legal
///   state table for that occupation;
/// - each coupling boundary satisfies `2 <= boundary < occupied.len()`;
/// - coupling boundaries are strictly increasing with no duplicates.
pub fn validate_record(
    subshells: &[String],
    occupied: &[OccupiedSubshell],
    couplings: &[IntermediateCoupling],
) -> Result<()> {
    ensure!(
        !occupied.is_empty(),
        "a CSF record must occupy at least one subshell"
    );

    let mut previous_index: Option<u16> = None;
    for shell in occupied {
        let index = usize::from(shell.subshell_index);
        let label = subshells
            .get(index)
            .with_context(|| format!("occupied subshell index {index} exceeds peel table"))?;
        ensure!(
            previous_index.is_none_or(|previous| shell.subshell_index > previous),
            "occupied subshells are out of Peel order or duplicated at index {index}"
        );
        previous_index = Some(shell.subshell_index);

        let subshell: Subshell = label
            .parse()
            .with_context(|| format!("invalid peel subshell label {label:?}"))?;
        let capacity = subshell.capacity();
        ensure!(
            shell.occupation >= 1 && shell.occupation <= capacity,
            "occupation {} for {label:?} is outside 1..={capacity}",
            shell.occupation
        );

        if let Some(state) = shell.state {
            let max_two_j = u16::from(capacity - 1);
            let legal_states = subshell_states(max_two_j, shell.occupation).with_context(|| {
                format!("no legal state table for {label:?}({})", shell.occupation)
            })?;
            ensure!(
                legal_states.contains(&state),
                "printed state {state:?} for {label:?}({}) is not a legal subshell state",
                shell.occupation
            );
        }
    }

    let occupied_len = u16::try_from(occupied.len()).context("too many occupied subshells")?;
    let mut previous_boundary: Option<u16> = None;
    for coupling in couplings {
        ensure!(
            (2..occupied_len).contains(&coupling.boundary),
            "coupling boundary {} is invalid for {} occupied subshells",
            coupling.boundary,
            occupied.len()
        );
        ensure!(
            previous_boundary.is_none_or(|previous| coupling.boundary > previous),
            "coupling boundaries are out of order or duplicated at boundary {}",
            coupling.boundary
        );
        previous_boundary = Some(coupling.boundary);
    }

    Ok(())
}

/// Build the Arrow output schema for a descriptor Parquet file.
///
/// V2 never normalizes (design doc §4.4, §5.1): the old per-subshell
/// normalization divides by physics-derived denominators keyed to a 3-wide
/// row and cannot be reused by just changing the stride to 4. Rejecting the
/// combination here means every V2 export path is Int32-only.
pub fn output_schema(
    layout: DescriptorLayout,
    normalize: bool,
) -> Result<Arc<arrow::datatypes::Schema>> {
    use arrow::datatypes::{DataType, Field, Schema};

    ensure!(
        !(layout.version() == DescriptorVersion::V2 && normalize),
        "V2 descriptors do not support normalize=true; normalization is P1 scope \
         (design doc §4.4/§5.1)"
    );

    let value_type = if normalize {
        DataType::Float32
    } else {
        DataType::Int32
    };
    let mut fields = Vec::with_capacity(layout.row_len());
    for name in layout.feature_column_names() {
        fields.push(Field::new(name, value_type.clone(), false));
    }
    for name in layout.global_column_names() {
        fields.push(Field::new(name, DataType::Int32, false));
    }
    Ok(Arc::new(Schema::new(fields)))
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
    normalized: bool,
    source_header_sha256: Option<&str>,
    source_header_filename: Option<&str>,
) -> Vec<KeyValue> {
    let mut entries = vec![
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
        KeyValue::new("normalized".to_owned(), Some(normalized.to_string())),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complete_csf::SubshellState;

    #[test]
    fn v1_layout_matches_legacy_dense_shape() {
        let layout = DescriptorLayout::new(DescriptorVersion::V1, 3);
        assert_eq!(layout.row_len(), 9);
        assert_eq!(layout.feature_len(), 9);
        assert_eq!(layout.total_two_j_index(), None);
        assert_eq!(layout.parity_index(), None);
        assert_eq!(
            layout.feature_column_names(),
            [
                "col_0", "col_1", "col_2", "col_3", "col_4", "col_5", "col_6", "col_7", "col_8"
            ]
        );
        assert!(layout.global_column_names().is_empty());
    }

    #[test]
    fn v2_layout_adds_two_global_columns() {
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 3);
        assert_eq!(layout.feature_len(), 12);
        assert_eq!(layout.row_len(), 14);
        assert_eq!(layout.total_two_j_index(), Some(12));
        assert_eq!(layout.parity_index(), Some(13));
        assert_eq!(
            layout.feature_column_names(),
            [
                "sub0_n", "sub0_2j", "sub0_v", "sub0_2k", "sub1_n", "sub1_2j", "sub1_v", "sub1_2k",
                "sub2_n", "sub2_2j", "sub2_v", "sub2_2k",
            ]
        );
        assert_eq!(layout.global_column_names(), ["total_two_j", "parity"]);
    }

    #[test]
    fn validate_record_accepts_a_legal_occupied_subshell() {
        let subshells = vec!["4f".to_owned()];
        let occupied = [OccupiedSubshell {
            subshell_index: 0,
            occupation: 4,
            state: Some(SubshellState {
                two_j: 4,
                seniority: Some(2),
            }),
        }];
        validate_record(&subshells, &occupied, &[]).unwrap();
    }

    #[test]
    fn validate_record_rejects_illegal_seniority_for_state() {
        let subshells = vec!["4f".to_owned()];
        // 4f(4) 2J=4 is only legal with seniority 2 or 4; seniority 9 does not exist.
        let occupied = [OccupiedSubshell {
            subshell_index: 0,
            occupation: 4,
            state: Some(SubshellState {
                two_j: 4,
                seniority: Some(9),
            }),
        }];
        let error = validate_record(&subshells, &occupied, &[]).unwrap_err();
        assert!(error.to_string().contains("not a legal subshell state"));
    }

    #[test]
    fn validate_record_rejects_occupation_above_capacity() {
        let subshells = vec!["4f".to_owned()];
        let occupied = [OccupiedSubshell {
            subshell_index: 0,
            occupation: 20,
            state: None,
        }];
        let error = validate_record(&subshells, &occupied, &[]).unwrap_err();
        assert!(error.to_string().contains("outside 1"));
    }

    #[test]
    fn validate_record_rejects_out_of_order_subshells() {
        let subshells = vec!["4f-".to_owned(), "4f".to_owned()];
        let occupied = [
            OccupiedSubshell {
                subshell_index: 1,
                occupation: 4,
                state: None,
            },
            OccupiedSubshell {
                subshell_index: 0,
                occupation: 3,
                state: None,
            },
        ];
        let error = validate_record(&subshells, &occupied, &[]).unwrap_err();
        assert!(error.to_string().contains("out of Peel order"));
    }

    #[test]
    fn validate_record_rejects_duplicate_subshell() {
        let subshells = vec!["4f".to_owned()];
        let occupied = [
            OccupiedSubshell {
                subshell_index: 0,
                occupation: 2,
                state: None,
            },
            OccupiedSubshell {
                subshell_index: 0,
                occupation: 2,
                state: None,
            },
        ];
        let error = validate_record(&subshells, &occupied, &[]).unwrap_err();
        assert!(error.to_string().contains("out of Peel order"));
    }

    #[test]
    fn validate_record_rejects_boundary_zero_one_or_last() {
        let subshells = vec!["1s".to_owned(), "2s".to_owned(), "3s".to_owned()];
        let occupied = [
            OccupiedSubshell {
                subshell_index: 0,
                occupation: 1,
                state: None,
            },
            OccupiedSubshell {
                subshell_index: 1,
                occupation: 1,
                state: None,
            },
            OccupiedSubshell {
                subshell_index: 2,
                occupation: 1,
                state: None,
            },
        ];
        for boundary in [0, 1, 3] {
            let couplings = [IntermediateCoupling { boundary, two_j: 0 }];
            let error = validate_record(&subshells, &occupied, &couplings).unwrap_err();
            assert!(error.to_string().contains("coupling boundary"));
        }
    }

    #[test]
    fn validate_record_rejects_nonincreasing_boundaries() {
        let subshells = vec![
            "1s".to_owned(),
            "2s".to_owned(),
            "3s".to_owned(),
            "4s".to_owned(),
        ];
        let occupied: Vec<_> = (0..4)
            .map(|index| OccupiedSubshell {
                subshell_index: index,
                occupation: 1,
                state: None,
            })
            .collect();
        let couplings = [
            IntermediateCoupling {
                boundary: 2,
                two_j: 0,
            },
            IntermediateCoupling {
                boundary: 2,
                two_j: 0,
            },
        ];
        let error = validate_record(&subshells, &occupied, &couplings).unwrap_err();
        assert!(error.to_string().contains("out of order or duplicated"));
    }

    #[test]
    fn output_schema_rejects_v2_normalize() {
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 2);
        assert!(output_schema(layout, true).is_err());
    }

    #[test]
    fn output_schema_v1_normalize_uses_float32() {
        use arrow::datatypes::DataType;
        let layout = DescriptorLayout::new(DescriptorVersion::V1, 2);
        let schema = output_schema(layout, true).unwrap();
        assert!(
            schema
                .fields()
                .iter()
                .all(|field| *field.data_type() == DataType::Float32)
        );
    }

    #[test]
    fn kv_metadata_round_trips_expected_keys() {
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 2);
        let peel_subshells = vec!["4f-".to_owned(), "4f".to_owned()];
        let entries = output_kv_metadata(
            layout,
            &peel_subshells,
            false,
            Some("abc123"),
            Some("x_header.toml"),
        );
        let get = |key: &str| {
            entries
                .iter()
                .find(|entry| entry.key == key)
                .and_then(|entry| entry.value.clone())
        };
        assert_eq!(get("descriptor_version"), Some("2".to_owned()));
        assert_eq!(get("channels_per_subshell"), Some("4".to_owned()));
        assert_eq!(get("subshell_count"), Some("2".to_owned()));
        assert_eq!(get("peel_subshells"), Some("[\"4f-\",\"4f\"]".to_owned()));
        assert_eq!(get("missing_sentinel"), Some("-1".to_owned()));
        assert_eq!(get("normalized"), Some("false".to_owned()));
        assert_eq!(get("source_header_sha256"), Some("abc123".to_owned()));
        assert_eq!(
            get("source_header_filename"),
            Some("x_header.toml".to_owned())
        );
    }
}
