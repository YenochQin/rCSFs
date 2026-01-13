//! Integration tests for CSF descriptor generation module
//!
//! These tests verify the public API of the CSF descriptor generator.

//////////////////////////////////////////////////////////////////////////////
// CSFDescriptorGenerator Public API Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_descriptor_generator_creation() {
    use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;

    let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
    let generator = CSFDescriptorGenerator::new(subshells.clone());

    assert_eq!(generator.orbital_count(), 3);
    assert_eq!(generator.peel_subshells(), &subshells);
}

#[test]
fn test_descriptor_generator_parse_csf_basic() {
    use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;

    let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string(),
                         "5p-".to_string(), "5p".to_string(), "6s".to_string()];
    let generator = CSFDescriptorGenerator::new(subshells);

    let line1 = "  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)";
    let line2 = "                   3/2               2        ";
    let line3 = "                                           4-  ";

    let result = generator.parse_csf(line1, line2, line3);

    assert!(result.is_ok());
    let descriptor = result.unwrap();

    // Expected descriptor size: 6 orbitals * 3 values per orbital = 18
    assert_eq!(descriptor.len(), 18);

    // Verify first orbital (5s): [2, 0, 0]
    assert_eq!(descriptor[0], 2);  // 2 electrons
    assert_eq!(descriptor[1], 0);  // J_middle = 0
    assert_eq!(descriptor[2], 0);  // J_coupling = 0

    // Verify second orbital (4d-): [4, 0, 0]
    assert_eq!(descriptor[3], 4);  // 4 electrons
    assert_eq!(descriptor[4], 0);  // J_middle = 0
    assert_eq!(descriptor[5], 0);  // J_coupling = 0

    // Verify third orbital (4d): [6, 3, 2]
    assert_eq!(descriptor[6], 6);  // 6 electrons
    assert_eq!(descriptor[7], 3);  // J_middle = 3/2 -> 3
    assert_eq!(descriptor[8], 2);  // J_coupling = 2

    // Verify last orbital (6s): [2, 0, 8] (last orbital gets final J)
    assert_eq!(descriptor[15], 2);  // 2 electrons
    assert_eq!(descriptor[16], 0);  // J_middle = 0
    assert_eq!(descriptor[17], 8);  // J_coupling = 4- -> 8
}

#[test]
fn test_descriptor_generator_parse_csf_empty_orbital() {
    use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;

    let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
    let generator = CSFDescriptorGenerator::new(subshells);

    // CSF with empty 5s orbital
    let line1 = "  5s ( 0)  4d-( 4)  4d ( 6)";
    let line2 = "                   3/2      ";
    let line3 = "                        4-  ";

    let result = generator.parse_csf(line1, line2, line3);

    assert!(result.is_ok());
    let descriptor = result.unwrap();

    // First orbital (5s) with 0 electrons: [0, 0, 0]
    assert_eq!(descriptor[0], 0);
    assert_eq!(descriptor[1], 0);
    assert_eq!(descriptor[2], 0);
}

#[test]
fn test_descriptor_generator_parse_csf_fractional_j() {
    use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;

    let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
    let generator = CSFDescriptorGenerator::new(subshells);

    // CSF with fractional J values
    let line1 = "  5s ( 2)  4d-( 4)  4d ( 6)";
    let line2 = "                   5/2      ";
    let line3 = "                        4-  ";

    let result = generator.parse_csf(line1, line2, line3);

    assert!(result.is_ok());
    let descriptor = result.unwrap();

    // Third orbital (4d): [6, 5, 8] (5/2 -> 5, 4- -> 8)
    assert_eq!(descriptor[6], 6);
    assert_eq!(descriptor[7], 5);  // J_middle = 5/2 -> 5
    assert_eq!(descriptor[8], 8);  // J_coupling = 4- -> 8
}

#[test]
fn test_descriptor_generator_orbital_count() {
    use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;

    // Test with different numbers of orbitals
    let subshells1 = vec!["5s".to_string()];
    let gen1 = CSFDescriptorGenerator::new(subshells1);
    assert_eq!(gen1.orbital_count(), 1);

    let subshells2 = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
    let gen2 = CSFDescriptorGenerator::new(subshells2);
    assert_eq!(gen2.orbital_count(), 3);

    let subshells3 = vec![
        "5s".to_string(), "4d-".to_string(), "4d".to_string(),
        "5p-".to_string(), "5p".to_string(), "6s".to_string()
    ];
    let gen3 = CSFDescriptorGenerator::new(subshells3);
    assert_eq!(gen3.orbital_count(), 6);
}

#[test]
fn test_descriptor_generator_get_config() {
    use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;

    let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
    let generator = CSFDescriptorGenerator::new(subshells.clone());

    // Test get_config() method
    let config = generator.get_config();
    assert_eq!(config.get("orbital_count"), Some(&3));
    assert_eq!(config.get("peel_subshells"), Some(&subshells));
}

#[test]
fn test_descriptor_generator_descriptor_size() {
    use _rcsfs::csfs_descriptor::CSFDescriptorGenerator;

    let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
    let generator = CSFDescriptorGenerator::new(subshells);

    let line1 = "  5s ( 2)  4d-( 4)  4d ( 6)";
    let line2 = "                   3/2      ";
    let line3 = "                        4-  ";

    let result = generator.parse_csf(line1, line2, line3);

    assert!(result.is_ok());
    let descriptor = result.unwrap();

    // Descriptor size should be 3 * orbital_count
    assert_eq!(descriptor.len(), 3 * generator.orbital_count());
}
