//! Integration tests for descriptor normalization module
//!
//! These tests verify the normalization functions for CSF descriptors.

//////////////////////////////////////////////////////////////////////////////
// Format Conversion Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_convert_full_to_angular() {
    use _rcsfs::descriptor_normalization::convert_full_to_angular;

    // Test basic conversions
    assert_eq!(convert_full_to_angular("2s"), "s ");
    assert_eq!(convert_full_to_angular("2p-"), "p-");
    assert_eq!(convert_full_to_angular("2p"), "p ");
    assert_eq!(convert_full_to_angular("3d-"), "d-");
    assert_eq!(convert_full_to_angular("3d"), "d ");
    assert_eq!(convert_full_to_angular("4f-"), "f-");
    assert_eq!(convert_full_to_angular("4f"), "f ");

    // Test with whitespace
    assert_eq!(convert_full_to_angular(" 2s "), "s ");
    assert_eq!(convert_full_to_angular(" 3p-"), "p-");

    // Test higher principal quantum numbers
    assert_eq!(convert_full_to_angular("5s"), "s ");
    assert_eq!(convert_full_to_angular("6g-"), "g-");
    assert_eq!(convert_full_to_angular("7h"), "h ");
}

#[test]
fn test_convert_full_to_angular_list() {
    use _rcsfs::descriptor_normalization::convert_full_to_angular_list;

    // Test list conversion
    let full = vec!["2s".to_string(), "2p-".to_string(), "2p".to_string(), "3s".to_string()];
    let angular = convert_full_to_angular_list(&full);
    assert_eq!(angular, vec!["s ", "p-", "p ", "s "]);

    // Test with more complex list
    let full = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string(), "5p-".to_string()];
    let angular = convert_full_to_angular_list(&full);
    assert_eq!(angular, vec!["s ", "d-", "d ", "p-"]);
}

//////////////////////////////////////////////////////////////////////////////
// Subshell Properties Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_get_max_subshell_electrons() {
    use _rcsfs::descriptor_normalization::get_max_subshell_electrons;

    assert_eq!(get_max_subshell_electrons("s "), Some(2.0));
    assert_eq!(get_max_subshell_electrons("p-"), Some(2.0));
    assert_eq!(get_max_subshell_electrons("p "), Some(4.0));
    assert_eq!(get_max_subshell_electrons("d-"), Some(4.0));
    assert_eq!(get_max_subshell_electrons("d "), Some(6.0));
    assert_eq!(get_max_subshell_electrons("f-"), Some(6.0));
    assert_eq!(get_max_subshell_electrons("f "), Some(8.0));
    assert_eq!(get_max_subshell_electrons("g-"), Some(8.0));
    assert_eq!(get_max_subshell_electrons("g "), Some(10.0));
    assert_eq!(get_max_subshell_electrons("h-"), Some(10.0));
    assert_eq!(get_max_subshell_electrons("h "), Some(12.0));
    assert_eq!(get_max_subshell_electrons("i-"), Some(12.0));
    assert_eq!(get_max_subshell_electrons("i "), Some(14.0));
    assert_eq!(get_max_subshell_electrons("xyz"), None);
}

#[test]
fn test_get_kappa_squared() {
    use _rcsfs::descriptor_normalization::get_kappa_squared;

    assert_eq!(get_kappa_squared("s "), Some(1));   // kappa = -1
    assert_eq!(get_kappa_squared("p-"), Some(1));   // kappa = 1
    assert_eq!(get_kappa_squared("p "), Some(4));   // kappa = -2
    assert_eq!(get_kappa_squared("d-"), Some(4));   // kappa = 2
    assert_eq!(get_kappa_squared("d "), Some(9));   // kappa = -3
    assert_eq!(get_kappa_squared("f-"), Some(9));   // kappa = 3
    assert_eq!(get_kappa_squared("f "), Some(16));  // kappa = -4
    assert_eq!(get_kappa_squared("g-"), Some(16));  // kappa = 4
    assert_eq!(get_kappa_squared("g "), Some(25));  // kappa = -5
    assert_eq!(get_kappa_squared("h-"), Some(25));  // kappa = 5
    assert_eq!(get_kappa_squared("h "), Some(36));  // kappa = -6
    assert_eq!(get_kappa_squared("i-"), Some(36));  // kappa = 6
    assert_eq!(get_kappa_squared("i "), Some(49));  // kappa = -7
    assert_eq!(get_kappa_squared("xyz"), None);
}

#[test]
fn test_get_subshell_properties() {
    use _rcsfs::descriptor_normalization::get_subshell_properties;

    // s orbital: max_electrons=2, kappa_sq=1
    let result = get_subshell_properties("s ", 10).unwrap();
    assert_eq!(result[0], 2);
    assert_eq!(result[1], 1);
    assert_eq!(result[2], 10);

    // p orbital: max_electrons=4, kappa_sq=4
    let result = get_subshell_properties("p ", 20).unwrap();
    assert_eq!(result[0], 4);
    assert_eq!(result[1], 4);
    assert_eq!(result[2], 20);

    // d orbital: max_electrons=6, kappa_sq=9
    let result = get_subshell_properties("d ", 30).unwrap();
    assert_eq!(result[0], 6);
    assert_eq!(result[1], 9);
    assert_eq!(result[2], 30);

    // Unknown subshell should return error
    assert!(get_subshell_properties("xyz", 10).is_err());
}

#[test]
fn test_get_subshells_properties() {
    use _rcsfs::descriptor_normalization::get_subshells_properties;

    let subshells = vec!["s ".to_string(), "p ".to_string(), "d ".to_string()];
    let result = get_subshells_properties(&subshells, 10).unwrap();

    // Expected: [2, 1, 10,  4, 4, 10,  6, 9, 10]
    assert_eq!(result.len(), 9);

    // s orbital
    assert_eq!(result[0], 2);
    assert_eq!(result[1], 1);
    assert_eq!(result[2], 10);

    // p orbital
    assert_eq!(result[3], 4);
    assert_eq!(result[4], 4);
    assert_eq!(result[5], 10);

    // d orbital
    assert_eq!(result[6], 6);
    assert_eq!(result[7], 9);
    assert_eq!(result[8], 10);

    // Test order preservation
    let subshells_reversed = vec!["d ".to_string(), "p ".to_string(), "s ".to_string()];
    let result_reversed = get_subshells_properties(&subshells_reversed, 5).unwrap();

    // Should be in reversed order: d first, then p, then s
    assert_eq!(result_reversed[0], 6);  // d max
    assert_eq!(result_reversed[3], 4);  // p max
    assert_eq!(result_reversed[6], 2);  // s max

    // Unknown subshell should return error
    let invalid_subshells = vec!["s ".to_string(), "xyz".to_string()];
    assert!(get_subshells_properties(&invalid_subshells, 10).is_err());
}

#[test]
fn test_get_subshells_properties_with_full_notation() {
    use _rcsfs::descriptor_normalization::get_subshells_properties;

    // Test that get_subshells_properties works with full notation
    let full_notation = vec!["2s".to_string(), "2p".to_string(), "3d".to_string()];
    let result = get_subshells_properties(&full_notation, 10).unwrap();

    // Expected: [2, 1, 10,  4, 4, 10,  6, 9, 10]
    assert_eq!(result.len(), 9);

    // s orbital
    assert_eq!(result[0], 2);
    assert_eq!(result[1], 1);
    assert_eq!(result[2], 10);

    // p orbital
    assert_eq!(result[3], 4);
    assert_eq!(result[4], 4);
    assert_eq!(result[5], 10);

    // d orbital
    assert_eq!(result[6], 6);
    assert_eq!(result[7], 9);
    assert_eq!(result[8], 10);
}

//////////////////////////////////////////////////////////////////////////////
// Reciprocal Computation Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_compute_properties_reciprocals() {
    use _rcsfs::descriptor_normalization::compute_properties_reciprocals;

    // Test basic reciprocal calculation
    let properties = vec![2, 1, 10, 4, 4, 10];
    let result = compute_properties_reciprocals(&properties).unwrap();

    assert_eq!(result.len(), 6);
    assert!((result[0] - 0.5).abs() < 0.001);   // 1/2
    assert!((result[1] - 1.0).abs() < 0.001);   // 1/1
    assert!((result[2] - 0.1).abs() < 0.001);   // 1/10
    assert!((result[3] - 0.25).abs() < 0.001);  // 1/4
    assert!((result[4] - 0.25).abs() < 0.001);  // 1/4
    assert!((result[5] - 0.1).abs() < 0.001);   // 1/10

    // Test with single element
    let single = vec![5];
    let result_single = compute_properties_reciprocals(&single).unwrap();
    assert_eq!(result_single.len(), 1);
    assert!((result_single[0] - 0.2).abs() < 0.001);  // 1/5

    // Test division by zero error
    let with_zero = vec![2, 0, 4];
    assert!(compute_properties_reciprocals(&with_zero).is_err());

    // Test combined with get_subshells_properties
    let subshells = vec!["s ".to_string(), "p ".to_string()];
    let props = get_subshells_properties(&subshells, 10).unwrap();
    let reciprocals = compute_properties_reciprocals(&props).unwrap();

    // props = [2, 1, 10, 4, 4, 10]
    // expected = [0.5, 1.0, 0.1, 0.25, 0.25, 0.1]
    assert!((reciprocals[0] - 0.5).abs() < 0.001);
    assert!((reciprocals[1] - 1.0).abs() < 0.001);
    assert!((reciprocals[2] - 0.1).abs() < 0.001);
    assert!((reciprocals[3] - 0.25).abs() < 0.001);
    assert!((reciprocals[4] - 0.25).abs() < 0.001);
    assert!((reciprocals[5] - 0.1).abs() < 0.001);
}

//////////////////////////////////////////////////////////////////////////////
// Normalization Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_normalize_electron_count() {
    use _rcsfs::descriptor_normalization::normalize_electron_count;

    // s orbital: 2/2 = 1.0
    assert_eq!(normalize_electron_count(2, "s "), Ok(1.0));

    // d orbital: 6/6 = 1.0
    assert_eq!(normalize_electron_count(6, "d "), Ok(1.0));

    // p- orbital: 2/2 = 1.0 (full)
    assert_eq!(normalize_electron_count(2, "p-"), Ok(1.0));

    // p orbital: 3/4 = 0.75 (partially filled)
    let result = normalize_electron_count(3, "p ").unwrap();
    assert!((result - 0.75).abs() < 0.01);

    // Unknown subshell
    assert!(normalize_electron_count(5, "xyz").is_err());
}

#[test]
fn test_normalize_descriptor() {
    use _rcsfs::descriptor_normalization::normalize_descriptor;

    let descriptor = vec![2, 3, 4, 6, 3, 8]; // 2 orbitals
    let subshells = vec!["s ".to_string(), "d ".to_string()];
    let max_cumulative_doubled_j = 10;

    let result = normalize_descriptor(&descriptor, &subshells, max_cumulative_doubled_j).unwrap();

    // get_subshells_properties => [2, 1, 10, 6, 9, 10]
    // reciprocals => [0.5, 1.0, 0.1, 0.167, 0.111, 0.1]
    // result: [2*0.5, 3*1.0, 4*0.1, 6*0.167, 3*0.111, 8*0.1]
    //       = [1.0, 3.0, 0.4, 1.0, 0.333, 0.8]

    // First orbital (s): 2*0.5=1.0, 3*1.0=3.0, 4*0.1=0.4
    assert!((result[0] - 1.0).abs() < 0.01);
    assert!((result[1] - 3.0).abs() < 0.01);
    assert!((result[2] - 0.4).abs() < 0.01);

    // Second orbital (d): 6*0.167=1.0, 3*0.111=0.333, 8*0.1=0.8
    assert!((result[3] - 1.0).abs() < 0.01);
    assert!((result[4] - 0.333).abs() < 0.01);
    assert!((result[5] - 0.8).abs() < 0.01);
}

#[test]
fn test_normalize_descriptor_length_mismatch() {
    use _rcsfs::descriptor_normalization::normalize_descriptor;

    let descriptor = vec![2, 3, 4, 6]; // Wrong length (should be 6 for 2 orbitals)
    let subshells = vec!["s ".to_string(), "d ".to_string()];
    let max_cumulative_doubled_j = 10;

    assert!(normalize_descriptor(&descriptor, &subshells, max_cumulative_doubled_j).is_err());
}

#[test]
fn test_batch_normalize_descriptors() {
    use _rcsfs::descriptor_normalization::batch_normalize_descriptors;

    let descriptors = vec![
        vec![1, 3, 4, 3, 3, 8],
        vec![2, 3, 4, 6, 3, 8],
    ];
    let subshells = vec!["s ".to_string(), "d ".to_string()];
    let max_cumulative_doubled_j = 10;

    let results = batch_normalize_descriptors(&descriptors, &subshells, max_cumulative_doubled_j).unwrap();

    assert_eq!(results.len(), 2);

    // reciprocals: [0.5, 1.0, 0.1, 0.167, 0.111, 0.1]

    // First descriptor: [1, 3, 4, 3, 3, 8] * reciprocals
    assert!((results[0][0] - 0.5).abs() < 0.01);  // 1*0.5
    assert!((results[0][3] - 0.5).abs() < 0.01);  // 3*0.167

    // Second descriptor: [2, 3, 4, 6, 3, 8] * reciprocals
    assert!((results[1][0] - 1.0).abs() < 0.01);  // 2*0.5
    assert!((results[1][3] - 1.0).abs() < 0.01);  // 6*0.167
}

#[test]
fn test_get_all_subshell_limits() {
    use _rcsfs::descriptor_normalization::get_all_subshell_limits;

    let limits = get_all_subshell_limits();

    assert_eq!(limits.get("s "), Some(&2.0));
    assert_eq!(limits.get("p-"), Some(&2.0));
    assert_eq!(limits.get("d "), Some(&6.0));
    assert_eq!(limits.get("f-"), Some(&6.0));
}
