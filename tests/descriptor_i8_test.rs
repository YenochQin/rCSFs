use _rcsfs::configuration_generation::{GenerationRequest, SubshellOccupation, generate_csfs};
use _rcsfs::csfs_descriptor::{CSFDescriptorGenerator, j_to_double_j};
use _rcsfs::descriptor_schema::{DescriptorLayout, DescriptorVersion, encode_angular};
use _rcsfs::descriptor_v3::{decode_v3_into, encode_v3};

#[test]
fn angular_narrowing_checks_the_grasp_bound_before_casting() {
    assert_eq!(encode_angular(0, "test").unwrap(), 1);
    assert_eq!(encode_angular(126, "test").unwrap(), 127);
    for value in [127, 128, 255, 256, u16::MAX] {
        assert!(encode_angular(value, "test").is_err());
    }
}

#[test]
fn text_parser_rejects_large_values_without_wrapping() {
    let generator = CSFDescriptorGenerator::new(vec!["5s".to_owned()]);
    // Exercise numeric range checking, not the physical coupling of this row.
    for last_line in ["        64+", "      9999+"] {
        let error = generator
            .parse_csf("  5s ( 1)", "      1/2", last_line)
            .unwrap_err();
        assert!(error.to_string().contains("angular bound"), "{error:#}");
    }
    assert!(j_to_double_j("2147483647").is_err());
}

#[test]
fn integer_encoder_and_decoder_validate_each_angular_channel() {
    let mut file = generate_csfs(&GenerationRequest {
        core_subshells: vec![],
        configuration: ["1s", "2s", "3s"]
            .into_iter()
            .map(|label| SubshellOccupation {
                subshell: label.parse().unwrap(),
                electrons: 1,
            })
            .collect(),
        min_two_j: 1,
        max_two_j: 1,
    })
    .unwrap();
    let layout = DescriptorLayout::new(DescriptorVersion::V3, file.subshells.len());
    let mut row = vec![0i8; layout.row_len()];
    encode_v3(&file, &file.records[0], &mut row).unwrap();
    let valid = row.clone();
    for column in [
        layout.two_j_offset(),
        layout.slot(1) + layout.two_k_offset(),
        layout.total_two_j_index().unwrap(),
    ] {
        for value in [0, 127, i8::MIN] {
            row.copy_from_slice(&valid);
            row[column] = value;
            assert!(
                decode_v3_into(&row, layout, &file.subshells, &mut vec![], &mut vec![]).is_err()
            );
        }
    }
    file.records[0].total_two_j = 128;
    assert!(
        encode_v3(&file, &file.records[0], &mut row)
            .unwrap_err()
            .to_string()
            .contains("total 2J")
    );
    let mut text = Vec::new();
    file.records[0].total_two_j = 1;
    file.write_to(&mut text).unwrap();
    let mut file = _rcsfs::complete_csf::CompleteCsfFile::parse_reader(text.as_slice()).unwrap();
    file.intermediate_couplings[0].two_j = 128;
    assert!(
        encode_v3(&file, &file.records[0], &mut row)
            .unwrap_err()
            .to_string()
            .contains("subshell 2s")
    );
}

#[test]
fn edited_generated_coupling_agrees_with_reimported_text() {
    let mut file = generate_csfs(&GenerationRequest {
        core_subshells: vec![],
        configuration: ["1s", "2s", "3s"]
            .map(|label| SubshellOccupation {
                subshell: label.parse().unwrap(),
                electrons: 1,
            })
            .to_vec(),
        min_two_j: 1,
        max_two_j: 1,
    })
    .unwrap();
    let parser = CSFDescriptorGenerator::new(file.subshells.clone());
    let mut row = vec![0; parser.layout().row_len()];
    let coupling = &mut file.intermediate_couplings[0];
    assert_eq!((coupling.boundary, coupling.two_j), (2, 0));
    coupling.two_j = 2;
    let record = &file.records[0];
    encode_v3(&file, record, &mut row).unwrap();
    let mut text = Vec::new();
    file.write_record_to(record, &mut text).unwrap();
    let text = std::str::from_utf8(&text).unwrap();
    let lines = text.lines().collect::<Vec<_>>();
    assert_eq!(row, parser.parse_csf(lines[0], lines[1], lines[2]).unwrap());
    assert_eq!(
        row[parser.layout().slot(1) + parser.layout().two_k_offset()],
        3
    );

    file.intermediate_couplings[0].two_j = 4;
    assert!(encode_v3(&file, record, &mut row).is_err());
}

#[test]
fn maximum_code_and_wide_intermediate_round_trip() {
    let file = generate_csfs(&GenerationRequest {
        core_subshells: vec![],
        configuration: ["1s", "2s", "11n-", "11n", "12n", "13n", "14n", "15n"]
            .into_iter()
            .map(|label| SubshellOccupation {
                subshell: label.parse().unwrap(),
                electrons: 1,
            })
            .collect(),
        min_two_j: 126,
        max_two_j: 126,
    })
    .unwrap();
    assert_eq!(file.records.len(), 1);
    let generator = CSFDescriptorGenerator::new(file.subshells.clone());
    let mut row = vec![0; generator.layout().row_len()];
    encode_v3(&file, &file.records[0], &mut row).unwrap();
    assert_eq!(&row[row.len() - 2..], &[127, 1]);
    assert_eq!(row[4 * 6 + 3], 106);
    let mut text = Vec::new();
    file.write_record_to(&file.records[0], &mut text).unwrap();
    let text = std::str::from_utf8(&text).unwrap();
    assert!(text.contains("105/2"));
    let lines = text.lines().collect::<Vec<_>>();
    assert_eq!(
        generator.parse_csf(lines[0], lines[1], lines[2]).unwrap(),
        row
    );
}
