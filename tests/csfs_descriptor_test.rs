//! Reversible V3 descriptors through the Rust text parsing API.

use _rcsfs::complete_csf::CompleteCsfFile;
use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;
use _rcsfs::descriptor_schema::DescriptorVersion;
use _rcsfs::descriptor_v3::{encode_v3, restore_file};
use std::path::Path;

#[test]
fn descriptor_generator_defaults_to_v3_and_matches_integer_records() {
    let file = CompleteCsfFile::parse_path(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample.csf"),
    )
    .unwrap();
    let generator = CSFDescriptorGenerator::new(file.subshells.clone());
    assert_eq!(generator.layout().version(), DescriptorVersion::V3);
    assert_eq!(generator.layout().channels_per_subshell(), 4);
    let mut expected = vec![0; generator.layout().row_len()];
    let mut reused = vec![99; expected.len()];
    for record in &file.records {
        let mut bytes = Vec::new();
        file.write_record_to(record, &mut bytes).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let lines = text.lines().collect::<Vec<_>>();
        encode_v3(&file, record, &mut expected).unwrap();
        assert_eq!(
            generator.parse_csf(lines[0], lines[1], lines[2]).unwrap(),
            expected
        );
        generator
            .parse_csf_into(lines[0], lines[1], lines[2], &mut reused)
            .unwrap();
        assert_eq!(reused, expected);
    }
    assert_eq!(generator.orbital_count(), file.subshells.len());
    assert_eq!(generator.peel_subshells(), &file.subshells);
}

#[test]
fn indexed_restore_keeps_source_blocks_even_when_symmetries_match() {
    let file = CompleteCsfFile::parse_reader(include_bytes!("fixtures/descriptor_v3.c").as_slice())
        .unwrap();
    let layout = CSFDescriptorGenerator::new(file.subshells.clone()).layout();
    let rows: Vec<_> = file
        .records
        .iter()
        .map(|record| {
            let mut row = vec![0; layout.row_len()];
            encode_v3(&file, record, &mut row).unwrap();
            row
        })
        .collect();
    for (indices, expected) in [
        (vec![0, 1, 2], vec![2, 1]),
        (vec![1, 2], vec![1, 1]),
        (vec![2, 0, 2], vec![1, 1, 1]),
    ] {
        let restored = restore_file(
            &rows,
            layout,
            file.header_lines.clone(),
            Some(&[2, 1]),
            Some(&indices),
        )
        .unwrap();
        assert_eq!(
            restored
                .blocks
                .iter()
                .map(|b| b.record_len)
                .collect::<Vec<_>>(),
            expected
        );
        for (record, index) in restored.records.iter().zip(indices) {
            let mut row = vec![0; layout.row_len()];
            encode_v3(&restored, record, &mut row).unwrap();
            assert_eq!(row, rows[index as usize]);
        }
    }
    for invalid in [&[2][..], &[2, 0, 1], &[usize::MAX, 1]] {
        assert!(
            restore_file(
                &rows,
                layout,
                file.header_lines.clone(),
                Some(invalid),
                Some(&[0])
            )
            .is_err()
        );
    }
}

#[test]
fn row_parser_accepts_closed_core_with_or_without_empty_peel_slots() {
    for (peel, expected) in [
        (vec![], vec![1, 1]),
        (vec!["2s".to_owned()], vec![0, 0, 1, 1, 1, 1]),
    ] {
        let parser = CSFDescriptorGenerator::new(peel);
        let mut row = vec![99; expected.len()];
        parser.parse_row_into("", "", "0+", &mut row).unwrap();
        assert_eq!(row, expected);
        assert_eq!(parser.parse_csf("  ", " ", " 0+ ").unwrap(), expected);
        for (state, coupling) in [("1/2", "0+"), ("", "1+"), ("", "0-")] {
            assert!(parser.parse_csf("", state, coupling).is_err());
        }
    }
}

#[test]
fn seniority_keeps_distinct_states_in_a_single_j_block() {
    use _rcsfs::csf_generation::{GenerationRequest, SubshellOccupation, generate_csfs};
    let file = generate_csfs(&GenerationRequest {
        core_subshells: Vec::new(),
        configuration: vec![SubshellOccupation {
            subshell: "4f".parse().unwrap(),
            electrons: 4,
        }],
        min_two_j: 4,
        max_two_j: 4,
    })
    .unwrap();
    let generator = CSFDescriptorGenerator::new(file.subshells.clone());
    let rows = file
        .records
        .iter()
        .map(|record| {
            let mut bytes = Vec::new();
            file.write_record_to(record, &mut bytes).unwrap();
            let text = String::from_utf8(bytes).unwrap();
            let lines = text.lines().collect::<Vec<_>>();
            generator.parse_csf(lines[0], lines[1], lines[2]).unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(rows, [[4, 2, 5, 5, 5, 1], [4, 4, 5, 5, 5, 1]]);
}

#[test]
fn descriptor_generator_rejects_non_ascii_and_wrong_buffers() {
    let generator = CSFDescriptorGenerator::new(vec!["5s".to_owned()]);
    assert!(
        generator
            .parse_csf("  测试 ( 2)", "", "        0+")
            .is_err()
    );
    let mut short = [0; 3];
    assert!(
        generator
            .parse_csf_into("  5s ( 2)", "", "        0+", &mut short)
            .is_err()
    );
}

#[test]
fn retired_descriptor_versions_require_regeneration() {
    let error = DescriptorVersion::from_tag(1).unwrap_err();
    assert!(error.to_string().contains("regenerate V3"));
    assert!(DescriptorVersion::from_tag(2).is_err());
    assert_eq!(
        DescriptorVersion::from_tag(3).unwrap(),
        DescriptorVersion::V3
    );
}
