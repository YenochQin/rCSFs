//! Lossless integer representation of GRASP CSF files.
//!
//! The ML descriptor intentionally discards some state information. This module
//! keeps every integer needed to reproduce the three fixed-width CSF lines,
//! including optional seniority labels and sparse intermediate couplings.

use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::mem::size_of;
use std::path::Path;

const HEADER_LINE_COUNT: usize = 5;
const FIELD_WIDTH: usize = 9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Parity {
    Even,
    Odd,
}

impl Parity {
    fn parse(value: u8) -> Result<Self> {
        match value {
            b'+' => Ok(Self::Even),
            b'-' => Ok(Self::Odd),
            _ => bail!("invalid parity character {:?}", char::from(value)),
        }
    }

    fn as_byte(self) -> u8 {
        match self {
            Self::Even => b'+',
            Self::Odd => b'-',
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubshellState {
    pub two_j: u16,
    pub seniority: Option<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OccupiedSubshell {
    pub subshell_index: u16,
    pub occupation: u8,
    pub state: Option<SubshellState>,
}

/// An explicitly printed intermediate coupling at a GRASP field boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntermediateCoupling {
    /// Zero-based 9-character field boundary. The first printable boundary is 2.
    pub boundary: u16,
    pub two_j: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsfRecord {
    occupied_start: u64,
    occupied_len: u16,
    coupling_start: u64,
    coupling_len: u16,
    pub total_two_j: u16,
    pub parity: Parity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymmetryBlock {
    pub record_start: u64,
    pub record_len: u64,
    pub total_two_j: u16,
    pub parity: Parity,
}

#[derive(Debug, Eq, PartialEq)]
pub struct CompleteCsfFile {
    pub header_lines: [String; HEADER_LINE_COUNT],
    pub subshells: Vec<String>,
    pub records: Vec<CsfRecord>,
    pub occupied_subshells: Vec<OccupiedSubshell>,
    pub intermediate_couplings: Vec<IntermediateCoupling>,
    pub blocks: Vec<SymmetryBlock>,
}

impl CompleteCsfFile {
    pub fn parse_path(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open CSF file {}", path.display()))?;
        Self::parse_reader(BufReader::new(file))
            .with_context(|| format!("failed to parse CSF file {}", path.display()))
    }

    pub fn parse_reader(reader: impl BufRead) -> Result<Self> {
        let mut lines = reader.lines().enumerate();
        let mut header = Vec::with_capacity(HEADER_LINE_COUNT);
        for header_index in 0..HEADER_LINE_COUNT {
            let (line_index, line) = lines
                .next()
                .with_context(|| format!("missing CSF header line {}", header_index + 1))?;
            let line = line.with_context(|| format!("failed to read line {}", line_index + 1))?;
            ensure_ascii(&line, line_index + 1)?;
            header.push(line);
        }
        let header_lines: [String; HEADER_LINE_COUNT] = header
            .try_into()
            .expect("exactly five header lines were collected");
        let subshells = parse_peel_subshells(&header_lines[3])?;
        let subshell_index = subshells
            .iter()
            .enumerate()
            .map(|(index, label)| {
                let index = u16::try_from(index).context("too many peel subshells")?;
                Ok((label.as_str(), index))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let mut data_lines = Vec::with_capacity(3);
        let mut records = Vec::new();
        let mut occupied_subshells = Vec::new();
        let mut intermediate_couplings = Vec::new();
        let mut blocks = Vec::new();
        let mut block_start = 0usize;

        for (line_index, line) in lines {
            let line_number = line_index + 1;
            let line = line.with_context(|| format!("failed to read line {line_number}"))?;
            ensure_ascii(&line, line_number)?;
            if line.trim() == "*" {
                ensure!(
                    data_lines.is_empty(),
                    "block separator at line {line_number} interrupts a CSF record"
                );
                finish_block(&records, &mut blocks, &mut block_start)?;
                continue;
            }

            data_lines.push((line_number, line));
            if data_lines.len() == 3 {
                let record = parse_record(
                    &data_lines,
                    &subshell_index,
                    &mut occupied_subshells,
                    &mut intermediate_couplings,
                )?;
                records.push(record);
                data_lines.clear();
            }
        }

        ensure!(
            data_lines.is_empty(),
            "final CSF record has {} of 3 lines",
            data_lines.len()
        );
        finish_block(&records, &mut blocks, &mut block_start)?;

        Ok(Self {
            header_lines,
            subshells,
            records,
            occupied_subshells,
            intermediate_couplings,
            blocks,
        })
    }

    pub fn write_path(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .with_context(|| format!("failed to create CSF file {}", path.display()))?;
        self.write_to(BufWriter::new(file))
            .with_context(|| format!("failed to write CSF file {}", path.display()))
    }

    pub fn write_to(&self, mut writer: impl Write) -> Result<()> {
        self.validate_layout()?;
        for line in &self.header_lines {
            writeln!(writer, "{line}")?;
        }
        for (block_index, block) in self.blocks.iter().enumerate() {
            if block_index > 0 {
                writeln!(writer, " *")?;
            }
            let start = usize::try_from(block.record_start)?;
            let len = usize::try_from(block.record_len)?;
            for record in &self.records[start..start + len] {
                let (line1, line2, line3) = self.format_record(record)?;
                writeln!(writer, "{line1}")?;
                writeln!(writer, "{line2}")?;
                writeln!(writer, "{line3}")?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    /// Approximate heap capacity owned by the compact representation.
    ///
    /// Allocator bookkeeping and temporary parser/formatter buffers are not included.
    pub fn allocated_bytes(&self) -> usize {
        let header_bytes = self
            .header_lines
            .iter()
            .map(String::capacity)
            .sum::<usize>();
        let subshell_text_bytes = self.subshells.iter().map(String::capacity).sum::<usize>();
        self.subshells.capacity() * size_of::<String>()
            + subshell_text_bytes
            + self.records.capacity() * size_of::<CsfRecord>()
            + self.occupied_subshells.capacity() * size_of::<OccupiedSubshell>()
            + self.intermediate_couplings.capacity() * size_of::<IntermediateCoupling>()
            + self.blocks.capacity() * size_of::<SymmetryBlock>()
            + header_bytes
    }

    pub fn occupied(&self, record: &CsfRecord) -> Result<&[OccupiedSubshell]> {
        checked_slice(
            &self.occupied_subshells,
            record.occupied_start,
            record.occupied_len,
            "occupied subshell",
        )
    }

    pub fn couplings(&self, record: &CsfRecord) -> Result<&[IntermediateCoupling]> {
        checked_slice(
            &self.intermediate_couplings,
            record.coupling_start,
            record.coupling_len,
            "intermediate coupling",
        )
    }

    fn format_record(&self, record: &CsfRecord) -> Result<(String, String, String)> {
        let occupied = self.occupied(record)?;
        ensure!(
            !occupied.is_empty(),
            "cannot format a CSF without occupied subshells"
        );
        let field_count = occupied.len();
        let mut line1 = vec![b' '; field_count * FIELD_WIDTH];
        let mut line2 = vec![b' '; field_count * FIELD_WIDTH];

        for (field_index, entry) in occupied.iter().enumerate() {
            let label = self
                .subshells
                .get(usize::from(entry.subshell_index))
                .with_context(|| format!("invalid subshell index {}", entry.subshell_index))?;
            ensure!(label.is_ascii(), "subshell label {label:?} is not ASCII");
            ensure!(
                label.len() <= 5,
                "subshell label {label:?} exceeds five characters"
            );
            ensure!(
                entry.occupation <= 99,
                "occupation exceeds two output digits"
            );
            let start = field_index * FIELD_WIDTH;
            let display_label = if label.ends_with('-') {
                label.to_owned()
            } else {
                format!("{label} ")
            };
            ensure!(
                display_label.len() <= 5,
                "formatted subshell label {display_label:?} exceeds five characters"
            );
            let label_start = start + 5 - display_label.len();
            line1[label_start..start + 5].copy_from_slice(display_label.as_bytes());
            line1[start + 5] = b'(';
            let occupation = entry.occupation.to_string();
            let occupation_start = start + 8 - occupation.len();
            line1[occupation_start..start + 8].copy_from_slice(occupation.as_bytes());
            line1[start + 8] = b')';

            if let Some(state) = entry.state {
                write_state_field(&mut line2[start..start + FIELD_WIDTH], state)?;
            }
        }

        let mut line3 = vec![b' '; field_count * FIELD_WIDTH + 2];
        for coupling in self.couplings(record)? {
            let boundary = usize::from(coupling.boundary);
            ensure!(
                (2..field_count).contains(&boundary),
                "coupling boundary {boundary} is invalid for {field_count} subshells"
            );
            let end = boundary * FIELD_WIDTH + 2;
            write_two_j_ending_at(&mut line3, end, coupling.two_j)?;
        }
        let total_end = field_count * FIELD_WIDTH;
        write_two_j_ending_at(&mut line3, total_end, record.total_two_j)?;
        line3[total_end + 1] = record.parity.as_byte();

        Ok((
            String::from_utf8(line1).expect("formatter emits ASCII"),
            String::from_utf8(line2)
                .expect("formatter emits ASCII")
                .trim_end()
                .to_owned(),
            String::from_utf8(line3).expect("formatter emits ASCII"),
        ))
    }

    fn validate_layout(&self) -> Result<()> {
        let mut expected_start = 0u64;
        for block in &self.blocks {
            ensure!(
                block.record_start == expected_start,
                "non-contiguous symmetry block layout"
            );
            let end = block
                .record_start
                .checked_add(block.record_len)
                .context("symmetry block record range overflow")?;
            ensure!(
                end <= self.records.len() as u64,
                "symmetry block exceeds records"
            );
            for record in &self.records[usize::try_from(block.record_start)?..usize::try_from(end)?]
            {
                ensure!(
                    record.total_two_j == block.total_two_j && record.parity == block.parity,
                    "record symmetry differs from its block"
                );
            }
            expected_start = end;
        }
        ensure!(
            expected_start == self.records.len() as u64,
            "symmetry blocks do not cover all records"
        );
        Ok(())
    }
}

fn parse_record(
    lines: &[(usize, String)],
    subshell_index: &HashMap<&str, u16>,
    occupied_arena: &mut Vec<OccupiedSubshell>,
    coupling_arena: &mut Vec<IntermediateCoupling>,
) -> Result<CsfRecord> {
    let (line1_number, line1) = (&lines[0].0, &lines[0].1);
    let (line2_number, line2) = (&lines[1].0, &lines[1].1);
    let (line3_number, line3) = (&lines[2].0, &lines[2].1);
    ensure!(
        !line1.is_empty() && line1.len().is_multiple_of(FIELD_WIDTH),
        "line {line1_number}: occupation line length {} is not a positive multiple of 9",
        line1.len()
    );
    let field_count = line1.len() / FIELD_WIDTH;
    ensure!(
        line2.len() <= line1.len(),
        "line {line2_number}: state line length {} exceeds occupation line length {}",
        line2.len(),
        line1.len()
    );
    ensure!(
        line3.len() == line1.len() + 2,
        "line {line3_number}: coupling line length {} must be occupation length + 2 ({})",
        line3.len(),
        line1.len() + 2
    );

    let occupied_start = u64::try_from(occupied_arena.len())?;
    for field_index in 0..field_count {
        let start = field_index * FIELD_WIDTH;
        let field1 = &line1[start..start + FIELD_WIDTH];
        let label = field1[0..5].trim();
        ensure!(
            !label.is_empty(),
            "line {line1_number}: empty subshell field"
        );
        ensure!(
            field1.as_bytes()[5] == b'(' && field1.as_bytes()[8] == b')',
            "line {line1_number}: malformed occupation field {field1:?}"
        );
        let occupation = field1[6..8].trim().parse::<u8>().with_context(|| {
            format!("line {line1_number}: invalid occupation in field {field1:?}")
        })?;
        let &subshell_index = subshell_index.get(label).with_context(|| {
            format!("line {line1_number}: subshell {label:?} is absent from peel subshells")
        })?;
        let field2 = fixed_width_field(line2, start, FIELD_WIDTH);
        let state = parse_state_field(field2)
            .with_context(|| format!("line {line2_number}: invalid state field {field2:?}"))?;
        occupied_arena.push(OccupiedSubshell {
            subshell_index,
            occupation,
            state,
        });
    }
    let occupied_len = u16::try_from(field_count).context("too many occupied subshells")?;

    let coupling_start = u64::try_from(coupling_arena.len())?;
    for boundary in 2..field_count {
        let start = boundary * FIELD_WIDTH - 1;
        let end = boundary * FIELD_WIDTH + 3;
        let value = line3[start..end].trim();
        if !value.is_empty() {
            coupling_arena.push(IntermediateCoupling {
                boundary: u16::try_from(boundary)?,
                two_j: parse_two_j(value).with_context(|| {
                    format!("line {line3_number}: invalid coupling value {value:?}")
                })?,
            });
        }
    }
    let coupling_len = u16::try_from(coupling_arena.len() as u64 - coupling_start)?;

    let bytes = line3.as_bytes();
    let parity = Parity::parse(bytes[line3.len() - 1])
        .with_context(|| format!("line {line3_number}: invalid final parity"))?;
    let total_field = line3[field_count * FIELD_WIDTH - 3..field_count * FIELD_WIDTH + 1].trim();
    let total_two_j = parse_two_j(total_field)
        .with_context(|| format!("line {line3_number}: invalid total J {total_field:?}"))?;

    Ok(CsfRecord {
        occupied_start,
        occupied_len,
        coupling_start,
        coupling_len,
        total_two_j,
        parity,
    })
}

fn parse_peel_subshells(line: &str) -> Result<Vec<String>> {
    let subshells = line
        .split_ascii_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(!subshells.is_empty(), "peel subshell header is empty");
    ensure!(
        subshells.len() <= usize::from(u16::MAX),
        "too many peel subshells"
    );
    let mut seen = HashMap::with_capacity(subshells.len());
    for label in &subshells {
        ensure!(
            label.len() <= 5,
            "subshell label {label:?} exceeds five characters"
        );
        ensure!(
            seen.insert(label, ()).is_none(),
            "duplicate peel subshell {label:?}"
        );
    }
    Ok(subshells)
}

fn parse_state_field(field: &str) -> Result<Option<SubshellState>> {
    let field = field.trim();
    if field.is_empty() {
        return Ok(None);
    }
    let (seniority, j_value) = match field.split_once(';') {
        Some((seniority, j_value)) => (
            Some(
                seniority
                    .trim()
                    .parse::<u8>()
                    .with_context(|| format!("invalid seniority {seniority:?}"))?,
            ),
            j_value,
        ),
        None => (None, field),
    };
    Ok(Some(SubshellState {
        two_j: parse_two_j(j_value.trim())?,
        seniority,
    }))
}

fn fixed_width_field(line: &str, start: usize, width: usize) -> &str {
    if start >= line.len() {
        return "";
    }
    &line[start..start.saturating_add(width).min(line.len())]
}

fn parse_two_j(value: &str) -> Result<u16> {
    if let Some((numerator, denominator)) = value.split_once('/') {
        ensure!(denominator == "2", "unsupported J denominator in {value:?}");
        return numerator
            .parse::<u16>()
            .with_context(|| format!("invalid half-integer J {value:?}"));
    }
    value
        .parse::<u16>()
        .with_context(|| format!("invalid integer J {value:?}"))?
        .checked_mul(2)
        .with_context(|| format!("2J overflow for {value:?}"))
}

fn format_two_j(two_j: u16) -> String {
    if two_j.is_multiple_of(2) {
        (two_j / 2).to_string()
    } else {
        format!("{two_j}/2")
    }
}

fn write_state_field(field: &mut [u8], state: SubshellState) -> Result<()> {
    debug_assert_eq!(field.len(), FIELD_WIDTH);
    let value = format_two_j(state.two_j);
    ensure!(
        value.len() <= 4,
        "J value {value:?} exceeds GRASP field width"
    );
    let value_start = FIELD_WIDTH - value.len();
    field[value_start..].copy_from_slice(value.as_bytes());
    if let Some(seniority) = state.seniority {
        ensure!(seniority <= 9, "seniority exceeds one GRASP output digit");
        ensure!(value_start > 4, "seniority overlaps J value {value:?}");
        field[3] = b'0' + seniority;
        field[4] = b';';
    }
    Ok(())
}

fn write_two_j_ending_at(line: &mut [u8], end: usize, two_j: u16) -> Result<()> {
    let value = format_two_j(two_j);
    ensure!(
        value.len() <= 4,
        "J value {value:?} exceeds GRASP field width"
    );
    let start = end + 1 - value.len();
    line[start..=end].copy_from_slice(value.as_bytes());
    Ok(())
}

fn finish_block(
    records: &[CsfRecord],
    blocks: &mut Vec<SymmetryBlock>,
    block_start: &mut usize,
) -> Result<()> {
    if *block_start == records.len() {
        return Ok(());
    }
    let first = records[*block_start];
    for record in &records[*block_start..] {
        ensure!(
            record.total_two_j == first.total_two_j && record.parity == first.parity,
            "CSF block mixes different total J/parity values"
        );
    }
    blocks.push(SymmetryBlock {
        record_start: u64::try_from(*block_start)?,
        record_len: u64::try_from(records.len() - *block_start)?,
        total_two_j: first.total_two_j,
        parity: first.parity,
    });
    *block_start = records.len();
    Ok(())
}

fn checked_slice<'a, T>(values: &'a [T], start: u64, len: u16, kind: &str) -> Result<&'a [T]> {
    let start = usize::try_from(start)?;
    let end = start
        .checked_add(usize::from(len))
        .with_context(|| format!("{kind} range overflow"))?;
    values
        .get(start..end)
        .with_context(|| format!("{kind} range exceeds arena"))
}

fn ensure_ascii(line: &str, line_number: usize) -> Result<()> {
    ensure!(
        line.is_ascii(),
        "line {line_number} contains non-ASCII text; GRASP CSF is fixed-width ASCII"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const CSF: &str = concat!(
        "Core subshells:\n",
        "  1s   2s\n",
        "Peel subshells:\n",
        "  4f-  4f   5d-  5d\n",
        "CSF(s):\n",
        "  4f-( 3)  4f ( 4)  5d-( 1)\n",
        "      3/2   4;   4      3/2\n",
        "                  7/2      4-\n",
        " *\n",
        "  4f-( 2)  4f ( 5)  5d ( 1)\n",
        "        2   2;11/2      5/2\n",
        "                    3    5/2+\n",
    );

    #[test]
    fn complete_integer_representation_round_trips_fixed_width_text() {
        let parsed = CompleteCsfFile::parse_reader(Cursor::new(CSF)).unwrap();
        assert_eq!(parsed.records.len(), 2);
        assert_eq!(parsed.blocks.len(), 2);
        assert_eq!(parsed.records[0].total_two_j, 8);
        assert_eq!(parsed.records[0].parity, Parity::Odd);
        let occupied = parsed.occupied(&parsed.records[0]).unwrap();
        assert_eq!(occupied[0].state.unwrap().two_j, 3);
        assert_eq!(occupied[1].state.unwrap().seniority, Some(4));
        assert_eq!(occupied[1].state.unwrap().two_j, 8);
        assert_eq!(parsed.couplings(&parsed.records[0]).unwrap()[0].two_j, 7);

        let mut output = Vec::new();
        parsed.write_to(&mut output).unwrap();
        assert_eq!(output, CSF.as_bytes());
    }

    #[test]
    fn blank_state_is_distinct_from_explicit_zero_state() {
        let mut input = CSF.to_owned();
        input = input.replacen("      3/2", "        0", 1);
        let parsed = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap();
        let occupied = parsed.occupied(&parsed.records[0]).unwrap();
        assert_eq!(
            occupied[0].state,
            Some(SubshellState {
                two_j: 0,
                seniority: None
            })
        );
    }

    #[test]
    fn rejects_subshell_missing_from_header() {
        let input = CSF.replacen("4f-( 3)", "5g-( 3)", 1);
        let error = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap_err();
        assert!(error.to_string().contains("absent from peel subshells"));
    }
}
