//! Explicit CSF text import primitives and the V2 record encoder.
use crate::complete_csf::{IntermediateCoupling, OccupiedSubshell, Parity, SubshellState};
use crate::descriptor_schema::{DescriptorLayout, DescriptorVersion, MISSING, validate_record};
use crate::descriptor_v2::write_feature_row;
use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;

pub(crate) fn parse_peel_subshells_from_header_lines(lines: &[String]) -> Result<Vec<String>> {
    let lines: [String; 5] = lines
        .to_vec()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected five header lines"))?;
    Ok(crate::descriptor_dataset::DatasetHeader::from_lines(lines)?.peel)
}

pub(crate) mod parquet_batch {
    use anyhow::Result;
    pub(crate) fn parse_compression(
        compression: Option<&str>,
    ) -> Result<parquet::basic::Compression> {
        use parquet::basic::Compression;

        let Some(spec) = compression else {
            return Ok(Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(3).expect("zstd level 3 is always valid"),
            ));
        };

        let lower = spec.trim().to_ascii_lowercase();
        match lower.as_str() {
            "none" | "uncompressed" => Ok(Compression::UNCOMPRESSED),
            "snappy" => Ok(Compression::SNAPPY),
            "zstd" => Ok(Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(3).expect("zstd level 3 is always valid"),
            )),
            other if other.starts_with("zstd-") => {
                let level_str = &other["zstd-".len()..];
                let level: i32 = level_str.parse().map_err(|_| {
                    anyhow::anyhow!(
                        "invalid zstd level '{}': expected integer 1..=22",
                        level_str
                    )
                })?;
                let zstd_level = parquet::basic::ZstdLevel::try_new(level).map_err(|_| {
                    anyhow::anyhow!("zstd level {} out of range (expected 1..=22)", level)
                })?;
                Ok(Compression::ZSTD(zstd_level))
            }
            other => Err(anyhow::anyhow!(
                "unknown compression '{}': expected one of none/uncompressed/snappy/zstd/zstd-N",
                other
            )),
        }
    }
}

/// Convert a J-value string to its doubled integer representation (2J)
///
/// # Arguments
/// * `j_str` - J value as string, e.g., "3/2", "2", "5/2"
///
/// # Returns
/// * `Ok(i32)` - The doubled J value (2J)
/// * `Err(String)` - Error message if parsing fails
///
/// # Examples
/// ```text
/// // Fractional J values:
/// j_to_double_j("3/2") => Ok(3)
/// j_to_double_j("5/2") => Ok(5)
///
/// // Integer J values:
/// j_to_double_j("2")  => Ok(4)
/// j_to_double_j("4")  => Ok(8)
/// ```
pub fn j_to_double_j(j_str: &str) -> Result<i32> {
    let trimmed = j_str.trim();

    // Handle fractional J values (e.g., "3/2" -> 3)
    if let Some(slash_pos) = trimmed.find('/') {
        let numerator: i32 = trimmed[..slash_pos]
            .parse()
            .with_context(|| format!("Invalid J value numerator: {}", trimmed))?;
        return Ok(numerator);
    }

    // Handle integer J values (e.g., "2" -> 4, "4-" -> 8)
    // Remove trailing parity indicator if present
    let cleaned = trimmed.trim_end_matches('-').trim_end_matches('+');
    cleaned
        .parse::<i32>()
        .with_context(|| format!("Invalid J value: {}", trimmed))?
        .checked_mul(2)
        .with_context(|| format!("Doubled J overflow: {trimmed}"))
}

fn fixed_width_field(line: &str, start: usize, width: usize) -> &str {
    if start >= line.len() {
        return "";
    }
    let end = start.saturating_add(width).min(line.len());
    line.get(start..end).unwrap_or("")
}

fn fixed_width_trimmed_field(line: &str, start: usize, width: usize) -> &str {
    fixed_width_field(line, start, width).trim()
}

/// Parse a line2 state field, keeping the printed seniority label.
fn parse_v2_state_field(field: &str) -> Result<Option<SubshellState>> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let (seniority, j_value) = match trimmed.split_once(';') {
        Some((seniority, j_value)) => {
            ensure!(
                field.len() == 9
                    && field.as_bytes().get(3).is_some_and(u8::is_ascii_digit)
                    && field.as_bytes().get(4) == Some(&b';'),
                "seniority must occupy field offsets 3 and 4 as a single digit followed by ';'"
            );
            let seniority: u8 = seniority
                .trim()
                .parse()
                .with_context(|| format!("invalid seniority {seniority:?}"))?;
            ensure!(seniority <= 9, "seniority exceeds one GRASP output digit");
            (Some(seniority), j_value)
        }
        None => (None, trimmed),
    };
    let two_j =
        u16::try_from(j_to_double_j(j_value.trim())?).context("state 2J value is negative")?;
    Ok(Some(SubshellState { two_j, seniority }))
}

/// CSF Descriptor Generator
///
/// This struct maintains the state needed to convert CSF data into descriptor arrays.
pub struct CSFDescriptorGenerator {
    /// List of peel subshell names (e.g., ["5s", "4d-", "4d", ...])
    peel_subshells: Vec<String>,
    /// Map from subshell name to index for O(1) lookup
    orbital_index_map: HashMap<String, usize>,
    /// Number of orbitals (cached for performance)
    orbital_count: usize,
    /// Row shape for this generator's descriptor version (plan D8 seam 1).
    layout: DescriptorLayout,
}

impl CSFDescriptorGenerator {
    /// Create a V2 CSF descriptor generator
    ///
    /// # Arguments
    /// * `peel_subshells` - List of subshell names (e.g., ["5s", "4d-", "4d"])
    ///
    /// # Returns
    /// A new generator instance
    pub fn new(peel_subshells: Vec<String>) -> Self {
        Self::new_with_version(peel_subshells, DescriptorVersion::V2)
    }

    /// Create a new CSF descriptor generator for a specific descriptor version.
    pub fn new_with_version(peel_subshells: Vec<String>, version: DescriptorVersion) -> Self {
        let orbital_count = peel_subshells.len();
        let orbital_index_map: HashMap<_, _> = peel_subshells
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        Self {
            peel_subshells,
            orbital_index_map,
            orbital_count,
            layout: DescriptorLayout::new(version, orbital_count),
        }
    }

    /// Get the number of orbitals
    pub fn orbital_count(&self) -> usize {
        self.orbital_count
    }

    /// Get the peel subshells list
    pub fn peel_subshells(&self) -> &[String] {
        &self.peel_subshells
    }

    /// Get this generator's descriptor row layout.
    pub fn layout(&self) -> DescriptorLayout {
        self.layout
    }

    /// Parse one CSF's three lines into a reusable V2 row buffer.
    ///
    /// `row.len()` must equal `self.layout().row_len()`. This is the single
    /// entry point used by descriptor export code.
    pub fn parse_row_into(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
        row: &mut [i8],
    ) -> Result<()> {
        self.parse_csf_into(line1, line2, line3, row)
    }

    /// Parse a single CSF into a descriptor array
    ///
    /// # Arguments
    /// * `line1` - First line: subshell configurations and electron counts
    /// * `line2` - Second line: intermediate J coupling values
    /// * `line3` - Third line: final coupling and total J value
    ///
    /// # Returns
    /// A vector of i8 descriptor values
    ///
    /// # CSF Format Example
    /// ```text
    /// line1: "  2p ( 1)"
    /// line2: "      3/2"
    /// line3: "     3/2-"
    /// ```
    pub fn parse_csf(&self, line1: &str, line2: &str, line3: &str) -> Result<Vec<i8>> {
        let mut descriptor = vec![MISSING; self.layout.row_len()];
        self.parse_csf_into(line1, line2, line3, &mut descriptor)?;
        Ok(descriptor)
    }

    /// Parse a CSF into a reversible V2 row, preserving printed zeros and missing fields.
    pub fn parse_csf_into(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
        row: &mut [i8],
    ) -> Result<()> {
        ensure!(
            self.layout.version() == DescriptorVersion::V2,
            "parse_csf_into requires a V2-configured generator"
        );
        ensure!(
            row.len() == self.layout.row_len(),
            "row buffer length {} does not match expected {}",
            row.len(),
            self.layout.row_len()
        );
        ensure!(
            line1.is_ascii() && line2.is_ascii() && line3.is_ascii(),
            "CSF lines must be ASCII fixed-width text"
        );

        let subshells_line = line1.trim_end();
        let line_length = subshells_line.len();
        ensure!(
            line_length > 0 && line_length.is_multiple_of(9),
            "occupation line length {line_length} is not a positive multiple of 9"
        );
        let field_count = line_length / 9;

        let coupling_line_raw = line3.trim_end();
        ensure!(
            coupling_line_raw.len() > 5,
            "coupling line {coupling_line_raw:?} is too short to hold a total J and parity"
        );
        let coupling_line = coupling_line_raw
            .get(4..coupling_line_raw.len() - 5)
            .with_context(|| {
                format!(
                    "coupling line {coupling_line_raw:?} is too short for {field_count} subshells"
                )
            })?;

        let final_j_str =
            &coupling_line_raw[coupling_line_raw.len() - 5..coupling_line_raw.len() - 1];
        let final_two_j =
            u16::try_from(j_to_double_j(final_j_str)?).context("total 2J is negative")?;

        let parity_byte = coupling_line_raw.as_bytes()[coupling_line_raw.len() - 1];
        let parity = match parity_byte {
            b'+' => Parity::Even,
            b'-' => Parity::Odd,
            other => bail!(
                "invalid parity byte {:?} in coupling line {coupling_line_raw:?}",
                char::from(other)
            ),
        };

        let mut occupied = Vec::with_capacity(field_count);
        let mut couplings = Vec::new();
        let trimmed_line2 = line2.trim_end();

        for i in 0..field_count {
            let start = i * 9;
            let subshell_field = fixed_width_field(subshells_line, start, 9);
            ensure!(
                subshell_field.len() == 9,
                "occupation field at position {i} is truncated"
            );

            let subshell = fixed_width_trimmed_field(subshell_field, 0, 5);
            ensure!(!subshell.is_empty(), "empty subshell field at position {i}");
            ensure!(
                subshell_field.as_bytes()[5] == b'(' && subshell_field.as_bytes()[8] == b')',
                "malformed occupation field {subshell_field:?} at position {i}"
            );

            let electron_field = fixed_width_trimmed_field(subshell_field, 6, 2);
            let electrons: u8 = electron_field.parse().with_context(|| {
                format!("invalid occupation {electron_field:?} at position {i}")
            })?;

            let &orbital_index = self.orbital_index_map.get(subshell).with_context(|| {
                format!("{subshell:?} at position {i} is absent from peel subshells")
            })?;

            let middle_field = fixed_width_field(trimmed_line2, start, 9);
            let state = parse_v2_state_field(middle_field)
                .with_context(|| format!("invalid state field {middle_field:?} at position {i}"))?;

            occupied.push(OccupiedSubshell {
                subshell_index: u16::try_from(orbital_index)?,
                occupation: electrons,
                state,
            });

            let coupling_field = fixed_width_field(coupling_line, start, 9);
            let trimmed_coupling = coupling_field.trim();
            if !trimmed_coupling.is_empty() {
                let boundary = u16::try_from(i + 1)?;
                let two_j = u16::try_from(j_to_double_j(trimmed_coupling)?)
                    .context("coupling 2J is negative")?;
                couplings.push(IntermediateCoupling { boundary, two_j });
            }
        }

        validate_record(&self.peel_subshells, &occupied, &couplings)?;
        write_feature_row(
            self.layout,
            &self.peel_subshells,
            &occupied,
            &couplings,
            final_two_j,
            parity,
            row,
        )
    }

    /// Derive the coupling-family view from V2; it is not a CSF identity key.
    pub(crate) fn parse_coupling_signature_into(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
        descriptor: &mut [i8],
        signature: &mut Vec<i32>,
    ) -> Result<()> {
        self.parse_csf_into(line1, line2, line3, descriptor)?;
        signature.clear();
        for slot in descriptor[..self.layout.feature_len()].as_chunks::<4>().0 {
            if slot[0] > 0 {
                let value = if slot[3] != MISSING {
                    slot[3]
                } else if slot[1] != MISSING {
                    slot[1]
                } else {
                    0
                };
                signature.push(i32::from(value));
            }
        }
        if let Some(last) = signature.last_mut() {
            *last = i32::from(descriptor[self.layout.total_two_j_index().unwrap()]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j_to_double_j_fractional() {
        assert_eq!(j_to_double_j("3/2").unwrap(), 3);
        assert_eq!(j_to_double_j("5/2").unwrap(), 5);
        assert_eq!(j_to_double_j("1/2").unwrap(), 1);
    }

    #[test]
    fn test_j_to_double_j_integer() {
        assert_eq!(j_to_double_j("2").unwrap(), 4);
        assert_eq!(j_to_double_j("3").unwrap(), 6);
        assert_eq!(j_to_double_j("4").unwrap(), 8);
    }

    #[test]
    fn test_j_to_double_j_with_parity() {
        assert_eq!(j_to_double_j("4-").unwrap(), 8);
        assert_eq!(j_to_double_j("3+").unwrap(), 6);
    }

    #[test]
    fn test_j_to_double_j_invalid() {
        assert!(j_to_double_j("invalid").is_err());
        assert!(j_to_double_j("abc/def").is_err());
    }

    #[test]
    fn test_descriptor_generator_creation() {
        let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
        let generator = CSFDescriptorGenerator::new(subshells.clone());

        assert_eq!(generator.orbital_count(), 3);
        assert_eq!(generator.peel_subshells(), &subshells);
    }

    #[test]
    fn parse_csf_into_rejects_wrong_buffer_size() {
        let generator = CSFDescriptorGenerator::new(vec!["5s".to_string()]);
        let mut too_short = vec![0i8; 2];

        let result = generator.parse_csf_into("  5s ( 2)", "", "      0  ", &mut too_short);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("row buffer length"),
            "error should explain buffer length mismatch"
        );
    }

    #[test]
    fn descriptor_pipeline_channel_capacity_limits_high_worker_memory_pressure() {
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(1), 1);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(2), 2);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(8), 8);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(48), 8);
    }

    ////////////////////////////////////////////////////////////////////////////
    // V2 text-path parser tests
    ////////////////////////////////////////////////////////////////////////////

    fn v2_generator(subshells: Vec<String>) -> CSFDescriptorGenerator {
        CSFDescriptorGenerator::new_with_version(subshells, DescriptorVersion::V2)
    }

    /// Generate one physically legal CSF record and return its three
    /// canonical text lines, so V2 text-parser tests never rely on
    /// hand-typed field literals that might not satisfy the real subshell
    /// state tables (as `tests/fixtures/complete.csf` predates
    /// `validate_record` and does not).
    fn generate_one_record_text(
        configuration: Vec<crate::configuration_generation::SubshellOccupation>,
        two_j: u16,
    ) -> (Vec<String>, String, String, String) {
        use crate::configuration_generation::{GenerationRequest, generate_csfs};

        let request = GenerationRequest {
            core_subshells: Vec::new(),
            configuration,
            min_two_j: two_j,
            max_two_j: two_j,
        };
        let generated = generate_csfs(&request).unwrap();
        let record = &generated.records[0];
        let mut buffer = Vec::new();
        generated.write_record_to(record, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        let mut lines = text.lines();
        (
            generated.subshells.clone(),
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
        )
    }

    #[test]
    fn parse_csf_into_basic_six_orbitals() {
        use crate::configuration_generation::SubshellOccupation;

        let (subshells, line1, line2, line3) = generate_one_record_text(
            vec![
                SubshellOccupation {
                    subshell: "5s".parse().unwrap(),
                    electrons: 2,
                },
                SubshellOccupation {
                    subshell: "4d-".parse().unwrap(),
                    electrons: 4,
                },
                SubshellOccupation {
                    subshell: "4d".parse().unwrap(),
                    electrons: 3,
                },
            ],
            3,
        );
        let generator = v2_generator(subshells.clone());

        let mut row = vec![0i8; generator.layout().row_len()];
        generator
            .parse_csf_into(&line1, &line2, &line3, &mut row)
            .unwrap();

        // Row length is 4*3+2 = 14.
        assert_eq!(row.len(), 14);
        assert_eq!(subshells.len(), 3);

        // 4d- (index 1, base 4): closed subshell (occupation == capacity),
        // so the generator never prints its state — MISSING rather than a
        // line2 back-fill of the neighboring coupling.
        assert_eq!(row[4], 4); // occupation
        assert_eq!(row[5], MISSING); // no printed 2J
        assert_eq!(row[6], 0); // closed subshell has physical seniority zero

        // 4d (index 2, base 8, last occupied position): total 2J is a
        // separate global column, never folded into this subshell's slot.
        assert_eq!(row[8], 3); // occupation
        assert_eq!(row[10], 3); // 4d(3), J=3/2 is a seniority-three state
        assert_eq!(row[11], MISSING); // last position: 2K boundary never printed

        let total_index = generator.layout().total_two_j_index().unwrap();
        assert_eq!(row[total_index], 3);
    }

    #[test]
    fn parse_csf_into_reads_parity_byte() {
        let subshells = vec!["5s".to_string()];
        let generator = v2_generator(subshells);
        let mut row = vec![0i8; generator.layout().row_len()];

        generator
            .parse_csf_into("  5s ( 1)", "", "        3+", &mut row)
            .unwrap();
        assert_eq!(row[row.len() - 1], 1);

        let error = generator
            .parse_csf_into("  5s ( 1)", "", "        3x", &mut row)
            .unwrap_err();
        assert!(error.to_string().contains("invalid parity byte"));
    }

    #[test]
    fn parse_csf_into_keeps_printed_seniority() {
        // 4f(4) at 2J=4 (doubled: 8) is only reachable with seniority 2 or 4
        // (the design doc §2.2 collision sample); seniority occupies the
        // fixed field offsets 3-4 (`kopp1`'s "s;"), J right-aligned in the
        // remaining 9-wide field.
        let subshells = vec!["4f".to_string()];
        let generator = v2_generator(subshells);
        let mut row = vec![0i8; generator.layout().row_len()];

        generator
            .parse_csf_into("  4f ( 4)", "   2;   4", "        4+", &mut row)
            .unwrap();

        assert_eq!(row[0], 4); // occupation
        assert_eq!(row[1], 8); // 2J = 4 -> 8
        assert_eq!(row[2], 2); // printed seniority retained
    }

    #[test]
    fn parse_csf_into_rejects_invalid_fields() {
        let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
        let generator = v2_generator(subshells);
        let mut row = vec![0i8; generator.layout().row_len()];

        // (a) Coupling line too short to slice with the fixed 4/5 offsets.
        assert!(
            generator
                .parse_csf_into("  5s ( 2)  4d-( 4)  4d ( 6)", "", "  4-", &mut row)
                .is_err()
        );

        // (b) Unparsable line2 J value must propagate, not become printed 0.
        assert!(
            generator
                .parse_csf_into(
                    "  5s ( 2)  4d-( 4)  4d ( 6)",
                    "         garbage         ",
                    "                        4-  ",
                    &mut row
                )
                .is_err()
        );

        // (c) Unparsable line3 coupling value must propagate.
        assert!(
            generator
                .parse_csf_into(
                    "  5s ( 2)  4d-( 4)  4d ( 6)",
                    "                   3/2      ",
                    "                garbage4-  ",
                    &mut row
                )
                .is_err()
        );

        // (d) Orbital missing from the peel map must error, not warn-and-skip.
        let missing_generator = v2_generator(vec!["5s".to_string()]);
        let mut missing_row = vec![0i8; missing_generator.layout().row_len()];
        assert!(
            missing_generator
                .parse_csf_into("  6p ( 2)", "", "        0+", &mut missing_row)
                .is_err()
        );

        // (e) Truncated/unparsable electron count must error, not become 0.
        assert!(
            generator
                .parse_csf_into("  5s (2  ", "", "                        0+", &mut row)
                .is_err()
        );

        // (f) Out-of-Peel-order occupied subshells must error via validate_record.
        let reordered_generator = v2_generator(vec!["4d-".to_string(), "5s".to_string()]);
        let mut reordered_row = vec![0i8; reordered_generator.layout().row_len()];
        assert!(
            reordered_generator
                .parse_csf_into(
                    "  5s ( 2)  4d-( 4)",
                    "",
                    "           0+",
                    &mut reordered_row
                )
                .is_err()
        );
    }

    #[test]
    fn parse_csf_into_rejects_zero_electron_count() {
        // V2 requires validate_record's occupation >= 1 instead.
        let generator = v2_generator(vec!["5s".to_string()]);
        let mut row = vec![0i8; generator.layout().row_len()];
        let error = generator
            .parse_csf_into("  5s ( 0)", "", "        0+", &mut row)
            .unwrap_err();
        assert!(error.to_string().contains("outside 1"));
    }

    #[test]
    fn text_and_complete_producers_agree_on_v2() {
        use crate::complete_csf::CompleteCsfFile;
        use crate::descriptor_v2::encode_v2;
        use std::io::Cursor;

        // The shared fixture (`tests/fixtures/complete.csf`) prints a state
        // outside GRASP's real state tables (see descriptor_v2.rs test
        // comments), so build a physically legal multi-record file with the
        // generator instead, then compare the two V2 producers row for row.
        use crate::configuration_generation::{
            GenerationRequest, SubshellOccupation, generate_csfs,
        };
        let request = GenerationRequest {
            core_subshells: Vec::new(),
            configuration: vec![
                SubshellOccupation {
                    subshell: "4f-".parse().unwrap(),
                    electrons: 3,
                },
                SubshellOccupation {
                    subshell: "4f".parse().unwrap(),
                    electrons: 4,
                },
            ],
            min_two_j: 1,
            max_two_j: 5,
        };
        let generated = generate_csfs(&request).unwrap();
        let mut text = Vec::new();
        generated.write_to(&mut text).unwrap();
        let parsed = CompleteCsfFile::parse_reader(Cursor::new(text)).unwrap();

        let generator = v2_generator(parsed.subshells.clone());
        for record in &parsed.records {
            let (line1, line2, line3) = format_record_for_test(&parsed, record);
            let mut text_row = vec![0i8; generator.layout().row_len()];
            generator
                .parse_csf_into(&line1, &line2, &line3, &mut text_row)
                .unwrap();

            let mut complete_row = vec![0i8; generator.layout().row_len()];
            encode_v2(&parsed, record, &mut complete_row).unwrap();

            assert_eq!(text_row, complete_row);
        }
    }

    fn format_record_for_test(
        file: &crate::complete_csf::CompleteCsfFile,
        record: &crate::complete_csf::CsfRecord,
    ) -> (String, String, String) {
        let mut buffer = Vec::new();
        file.write_record_to(record, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        let mut lines = text.lines();
        (
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
        )
    }
}
