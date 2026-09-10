use _rcsfs::complete_csf::{CompleteCsfFile, Parity, SubshellState};
use std::io::Cursor;

const CSF: &str = include_str!("fixtures/complete.csf");

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

#[test]
fn rejects_separator_without_grasp_leading_space() {
    let input = CSF.replacen(" *", "*", 1);
    let error = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("block separator must be exactly")
    );
}

#[test]
fn rejects_empty_symmetry_block() {
    let input = CSF.replacen(" *\n", " *\n *\n", 1);
    let error = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap_err();
    assert!(error.to_string().contains("empty symmetry block"));
}

#[test]
fn rejects_trailing_separator() {
    let input = format!("{CSF} *\n");
    let error = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap_err();
    assert!(error.to_string().contains("empty symmetry block"));
}

#[test]
fn rejects_unexpected_header_label() {
    let input = CSF.replacen("Peel subshells:", "Peel subshell:", 1);
    let error = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected CSF header label \"Peel subshells:\"")
    );
}

#[test]
fn rejects_file_without_records() {
    let header = CSF.split_inclusive('\n').take(5).collect::<String>();
    let error = CompleteCsfFile::parse_reader(Cursor::new(header)).unwrap_err();
    assert!(error.to_string().contains("contains no CSF records"));
}

/// Replace one 1-based line of `CSF`, so tests never depend on hand-counted
/// padding in the fixed-width literal above.
fn with_line(line_number: usize, replacement: &str) -> String {
    let mut lines = CSF.lines().map(str::to_owned).collect::<Vec<_>>();
    lines[line_number - 1] = replacement.to_owned();
    lines
        .iter()
        .map(|line| format!("{line}\n"))
        .collect::<String>()
}

/// `kopp2` never writes an even numerator over `/2`; accepting it would
/// silently rewrite the field to reduced form on output.
#[test]
fn rejects_unreduced_half_integer_j() {
    let input = with_line(8, "                  8/2      4-");
    let error = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap_err();
    assert!(format!("{error:#}").contains("is unreduced"));
}

/// `u16::from_str` accepts a leading `+`, but no GRASP writer emits one.
#[test]
fn rejects_sign_prefixed_j() {
    let input = with_line(8, "                  7/2     +4-");
    let error = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap_err();
    assert!(format!("{error:#}").contains("not a bare decimal number"));
}

/// `kopp1` fixes the seniority separator at field offset 4.
#[test]
fn rejects_misaligned_seniority_separator() {
    let input = with_line(7, "      3/2  4 ;   4      3/2");
    let error = CompleteCsfFile::parse_reader(Cursor::new(input)).unwrap_err();
    assert!(format!("{error:#}").contains("seniority must occupy field offsets 3 and 4"));
}

/// The suppression rule in `kopp2` leaves interior boundaries blank, so
/// `couplings()` is sparse and indexed by `boundary`, not dense.
#[test]
fn printed_couplings_are_sparse() {
    let parsed = CompleteCsfFile::parse_reader(Cursor::new(CSF)).unwrap();
    let record = &parsed.records[0];
    let occupied = parsed.occupied(record).unwrap();
    let couplings = parsed.couplings(record).unwrap();
    assert_eq!(occupied.len(), 3);
    assert_eq!(couplings.len(), 1);
    assert_eq!(couplings[0].boundary, 2);
}
#[test]
fn rejects_text_that_would_be_silently_rewritten() {
    let cases = [
        CSF.replacen("( 3)", "(+3)", 1),
        CSF.replacen("( 3)", "(03)", 1),
        CSF.replacen("      4-", "     04-", 1),
        CSF.replacen("                  7/2", "X                 7/2", 1),
        CSF.replacen("      3/2", "     3/2 ", 1),
        CSF.replacen("  4f-", "4f-  ", 2),
        CSF.replacen("      3/2\n", "      3/2 \n", 1),
    ];
    for input in cases {
        assert!(
            CompleteCsfFile::parse_reader(Cursor::new(&input)).is_err(),
            "accepted {input:?}"
        );
    }
}

#[test]
fn rejects_noncanonical_line_endings() {
    for input in [
        CSF.replace('\n', "\r\n"),
        CSF.trim_end_matches('\n').to_owned(),
    ] {
        assert!(CompleteCsfFile::parse_reader(Cursor::new(input)).is_err());
    }
}

#[test]
fn individual_records_export_without_headers_or_block_separators() {
    let parsed = CompleteCsfFile::parse_reader(Cursor::new(CSF)).unwrap();
    let expected = CSF
        .lines()
        .skip(5)
        .filter(|line| *line != " *")
        .collect::<Vec<_>>();
    for (record, lines) in parsed.records.iter().zip(expected.chunks_exact(3)) {
        let mut output = Vec::new();
        parsed.write_record_to(record, &mut output).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            format!("{}\n", lines.join("\n"))
        );
    }
    let mut record = parsed.records[0].clone();
    record.total_two_j = u16::MAX;
    assert!(parsed.write_record_to(&record, Vec::new()).is_err());
}
