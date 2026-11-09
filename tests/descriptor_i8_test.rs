use _rcsfs::configuration_generation::{GenerationRequest, SubshellOccupation, generate_csfs};
use _rcsfs::csfs_descriptor::{CSFDescriptorGenerator, j_to_double_j};
use _rcsfs::descriptor_schema::{DescriptorLayout, DescriptorVersion, checked_angular_value};
use _rcsfs::descriptor_v2::{decode_v2_into, encode_v2};

#[test]
fn angular_narrowing_checks_the_grasp_bound_before_casting() {
    assert_eq!(checked_angular_value(0, "test").unwrap(), 0);
    assert_eq!(checked_angular_value(126, "test").unwrap(), 126);
    for value in [127, 128, 255, 256, u16::MAX] {
        assert!(checked_angular_value(value, "test").is_err());
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
    let layout = DescriptorLayout::new(DescriptorVersion::V2, file.subshells.len());
    let mut row = vec![0i8; layout.row_len()];
    encode_v2(&file, &file.records[0], &mut row).unwrap();
    let valid = row.clone();
    for column in [
        layout.two_j_offset(),
        layout.slot(1) + layout.two_k_offset(),
        layout.total_two_j_index().unwrap(),
    ] {
        for value in [127, i8::MIN] {
            row.copy_from_slice(&valid);
            row[column] = value;
            assert!(
                decode_v2_into(&row, layout, &file.subshells, &mut vec![], &mut vec![]).is_err()
            );
        }
    }
    file.records[0].total_two_j = 128;
    assert!(
        encode_v2(&file, &file.records[0], &mut row)
            .unwrap_err()
            .to_string()
            .contains("total 2J")
    );
    file.records[0].total_two_j = 1;
    file.intermediate_couplings[0].two_j = 128;
    assert!(
        encode_v2(&file, &file.records[0], &mut row)
            .unwrap_err()
            .to_string()
            .contains("intermediate 2K")
    );
}
