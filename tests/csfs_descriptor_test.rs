//! Reversible V2 descriptors through the Rust text parsing API.

use _rcsfs::complete_csf::CompleteCsfFile;
use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;
use _rcsfs::descriptor_schema::{DescriptorVersion, MISSING};
use _rcsfs::descriptor_v2::encode_v2;
use std::path::Path;

#[test]
fn descriptor_generator_defaults_to_v2_and_matches_integer_records() {
    let file = CompleteCsfFile::parse_path(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample.csf"),
    )
    .unwrap();
    let generator = CSFDescriptorGenerator::new(file.subshells.clone());
    assert_eq!(generator.layout().version(), DescriptorVersion::V2);
    assert_eq!(generator.layout().channels_per_subshell(), 4);
    let mut expected = vec![0; generator.layout().row_len()];
    let mut reused = vec![99; expected.len()];
    for record in &file.records {
        let mut bytes = Vec::new();
        file.write_record_to(record, &mut bytes).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let lines = text.lines().collect::<Vec<_>>();
        encode_v2(&file, record, &mut expected).unwrap();
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
    assert_eq!(rows, [[4, 4, 2, MISSING, 4, 1], [4, 4, 4, MISSING, 4, 1]]);
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
    assert!(error.to_string().contains("regenerate V2"));
    assert!(DescriptorVersion::from_tag(3).is_err());
}
