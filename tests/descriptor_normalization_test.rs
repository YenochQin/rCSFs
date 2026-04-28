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
    let full = vec![
        "2s".to_string(),
        "2p-".to_string(),
        "2p".to_string(),
        "3s".to_string(),
    ];
    let angular = convert_full_to_angular_list(&full);
    assert_eq!(angular, vec!["s ", "p-", "p ", "s "]);

    // Test with more complex list
    let full = vec![
        "5s".to_string(),
        "4d-".to_string(),
        "4d".to_string(),
        "5p-".to_string(),
    ];
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

    assert_eq!(get_kappa_squared("s "), Some(1)); // kappa = -1
    assert_eq!(get_kappa_squared("p-"), Some(1)); // kappa = 1
    assert_eq!(get_kappa_squared("p "), Some(4)); // kappa = -2
    assert_eq!(get_kappa_squared("d-"), Some(4)); // kappa = 2
    assert_eq!(get_kappa_squared("d "), Some(9)); // kappa = -3
    assert_eq!(get_kappa_squared("f-"), Some(9)); // kappa = 3
    assert_eq!(get_kappa_squared("f "), Some(16)); // kappa = -4
    assert_eq!(get_kappa_squared("g-"), Some(16)); // kappa = 4
    assert_eq!(get_kappa_squared("g "), Some(25)); // kappa = -5
    assert_eq!(get_kappa_squared("h-"), Some(25)); // kappa = 5
    assert_eq!(get_kappa_squared("h "), Some(36)); // kappa = -6
    assert_eq!(get_kappa_squared("i-"), Some(36)); // kappa = 6
    assert_eq!(get_kappa_squared("i "), Some(49)); // kappa = -7
    assert_eq!(get_kappa_squared("xyz"), None);
}

//////////////////////////////////////////////////////////////////////////////
// Normalization Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_normalize_electron_count() {
    use _rcsfs::descriptor_normalization::normalize_electron_count;

    // s orbital: 2/2 = 1.0
    assert!((normalize_electron_count(2, "s ").unwrap() - 1.0).abs() < 1e-5);

    // d orbital: 6/6 = 1.0
    assert!((normalize_electron_count(6, "d ").unwrap() - 1.0).abs() < 1e-5);

    // p- orbital: 2/2 = 1.0 (full)
    assert!((normalize_electron_count(2, "p-").unwrap() - 1.0).abs() < 1e-5);

    // p orbital: 3/4 = 0.75 (partially filled)
    let result = normalize_electron_count(3, "p ").unwrap();
    assert!((result - 0.75).abs() < 0.01);

    // Unknown subshell
    assert!(normalize_electron_count(5, "xyz").is_err());
}

#[test]
fn test_batch_normalize_descriptors() {
    use _rcsfs::descriptor_normalization::batch_normalize_descriptors;

    let descriptors = vec![vec![1, 1, 1, 3, 9, 6], vec![1, 1, 1, 1, 5, 6]];
    let subshells = vec!["s ".to_string(), "d ".to_string()];
    let two_j_target = 6;

    let results = batch_normalize_descriptors(&descriptors, &subshells, two_j_target).unwrap();

    assert_eq!(results.len(), 2);

    assert!((results[0][0] - 0.5).abs() < 1e-5);
    assert!((results[0][5] - 1.0).abs() < 1e-5);
    assert!((results[1][0] - 0.5).abs() < 1e-5);
    assert!((results[1][4] - 1.0).abs() < 1e-5);
    assert!((results[1][5] - 1.0).abs() < 1e-5);
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

//////////////////////////////////////////////////////////////////////////////
// Per-CSF Physics-Correct Normalization Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_normalize_descriptor_per_csf_s_orbital() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // s orbital: g=2, n=1, u=1*(2-1)=1
    // prefix[0]=1, suffix[0]=0
    // U_0 = min(1, 1+0) = 1
    // n_i/g=0.5, 2Q/u=1/1=1.0, 2J/U=1/1=1.0
    let descriptor = vec![1, 1, 1];
    let subshells = vec!["s ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 1).unwrap();

    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_normalize_descriptor_per_csf_empty_subshell() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // n=0: u=0 → 2Q and 2J_cum fields must output 0.0
    let descriptor = vec![0, 0, 0];
    let subshells = vec!["p ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 0).unwrap();

    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 0.0).abs() < 1e-5);
    assert!((result[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_normalize_descriptor_per_csf_full_subshell() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // d orbital full: n=6=g, u=0 → n_i/g=1.0, others 0.0
    let descriptor = vec![6, 0, 0];
    let subshells = vec!["d ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 0).unwrap();

    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 0.0).abs() < 1e-5);
    assert!((result[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_normalize_descriptor_per_csf_d_half_filled() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // d orbital half-filled: n=3, g=6, u=3*(6-3)=9
    // At half-filling u == kappa^2, so 2Q_max/u = 9/9 = 1.0
    // prefix[0]=9, suffix[0]=0, U_0=min(9,9+0)=9
    let descriptor = vec![3, 9, 9];
    let subshells = vec!["d ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 9).unwrap();

    assert!((result[0] - 0.5).abs() < 1e-5); // 3/6
    assert!((result[1] - 1.0).abs() < 1e-5); // 9/9
    assert!((result[2] - 1.0).abs() < 1e-5); // 9/9
}

#[test]
fn test_normalize_descriptor_per_csf_d_non_half_filled() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // d orbital with n=1: u=1*(6-1)=5  (kappa^2=9 would be loose; u=5 is tighter)
    // 2Q_max for n=1 in d orbital is 5, so 5/5 = 1.0 (not 5/9 as with kappa^2)
    let descriptor = vec![1, 5, 5];
    let subshells = vec!["d ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 5).unwrap();

    assert!((result[0] - (1.0 / 6.0)).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5); // 5/5, not 5/9
    assert!((result[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_normalize_descriptor_per_csf_two_subshells_prefix_suffix() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // Two s orbitals: g=[2,2], n=[1,1], u=[1,1]
    // prefix=[1,2], suffix=[1,0]
    // 2J_target=2
    // i=0: U=min(1, 2+1)=1  → 2J_cum[0]/1
    // i=1: U=min(2, 2+0)=2  → 2J_cum[1]/2
    let descriptor = vec![1, 1, 1, 1, 1, 2];
    let subshells = vec!["s ".to_string(), "s ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 2).unwrap();

    // i=0
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] - 1.0).abs() < 1e-5); // 1/min(1,3)=1/1
    // i=1
    assert!((result[3] - 0.5).abs() < 1e-5);
    assert!((result[4] - 1.0).abs() < 1e-5);
    assert!((result[5] - 1.0).abs() < 1e-5); // 2/min(2,2)=1.0
}

#[test]
fn test_normalize_descriptor_per_csf_last_position_is_one() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // When prefix[N-1] >= 2J_target, the last coupling always normalizes to 1.0
    // p orbital: n=2, g=4, u=2*(4-2)=4; 2J_target=4
    // prefix[0]=4, suffix[0]=0, U_0=min(4,4)=4
    // last coupling = 2J_target = 4 → 4/4=1.0
    let descriptor = vec![2, 4, 4];
    let subshells = vec!["p ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 4).unwrap();

    assert!((result[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_normalize_descriptor_per_csf_length_mismatch() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    let descriptor = vec![1, 1]; // too short for 1 orbital (needs 3)
    let subshells = vec!["s ".to_string()];
    assert!(normalize_descriptor_per_csf(&descriptor, &subshells, 1).is_err());
}

#[test]
fn test_normalize_descriptor_per_csf_empty_descriptor() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    let descriptor = vec![];
    let subshells = vec![];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_normalize_descriptor_per_csf_full_notation_subshells() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // Full notation "2s" and angular notation "s " must produce identical results
    let descriptor = vec![1, 1, 1];
    let result_angular =
        normalize_descriptor_per_csf(&descriptor, &vec!["s ".to_string()], 1).unwrap();
    let result_full =
        normalize_descriptor_per_csf(&descriptor, &vec!["2s".to_string()], 1).unwrap();

    for (a, b) in result_angular.iter().zip(result_full.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_normalize_descriptor_per_csf_unknown_subshell() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    let descriptor = vec![1, 1, 1];
    let subshells = vec!["xyz".to_string()];
    assert!(normalize_descriptor_per_csf(&descriptor, &subshells, 1).is_err());
}

#[test]
fn test_normalize_descriptor_per_csf_rejects_invalid_electron_count() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    let too_many = vec![3, 0, 0];
    let negative = vec![-1, 0, 0];
    let subshells = vec!["s ".to_string()];

    assert!(normalize_descriptor_per_csf(&too_many, &subshells, 0).is_err());
    assert!(normalize_descriptor_per_csf(&negative, &subshells, 0).is_err());
}

#[test]
fn test_normalize_descriptor_per_csf_rejects_negative_target_j() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    let descriptor = vec![1, 1, 1];
    let subshells = vec!["s ".to_string()];

    assert!(normalize_descriptor_per_csf(&descriptor, &subshells, -1).is_err());
}

#[test]
fn test_normalize_descriptor_per_csf_rear_constraint_binding() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // subshells = ["f ", "s "]  g=[8, 2]
    // n=[4, 1], u=[4*(8-4)=16, 1*(2-1)=1]
    // prefix=[16, 17], suffix=[1, 0]
    // two_j_target=2
    // i=0 (f): U=min(16, 2+1)=3  ← rear constraint is binding (3 << prefix=16)
    // i=1 (s): U=min(17, 2+0)=2  ← rear constraint is binding
    let descriptor = vec![4, 12, 3, 1, 1, 2];
    let subshells = vec!["f ".to_string(), "s ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 2).unwrap();

    // i=0 (f): n/g=4/8=0.5, 2Q/u=12/16=0.75, 2J/U=3/3=1.0
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 0.75).abs() < 1e-5);
    assert!((result[2] - 1.0).abs() < 1e-5);
    // i=1 (s): n/g=1/2=0.5, 2Q/u=1/1=1.0, 2J/U=2/2=1.0
    assert!((result[3] - 0.5).abs() < 1e-5);
    assert!((result[4] - 1.0).abs() < 1e-5);
    assert!((result[5] - 1.0).abs() < 1e-5);
}

#[test]
fn test_normalize_descriptor_per_csf_heterogeneous_partial_occupation() {
    use _rcsfs::descriptor_normalization::normalize_descriptor_per_csf;

    // subshells = ["p-", "d "]  g=[2, 6]
    // n=[1, 2], u=[1*(2-1)=1, 2*(6-2)=8]
    // prefix=[1, 9], suffix=[8, 0]
    // two_j_target=5
    // i=0 (p-): U=min(1, 5+8)=1  ← front constraint binding
    // i=1 (d ): U=min(9, 5+0)=5  ← rear constraint binding (5 < prefix=9)
    let descriptor = vec![1, 1, 1, 2, 6, 5];
    let subshells = vec!["p-".to_string(), "d ".to_string()];
    let result = normalize_descriptor_per_csf(&descriptor, &subshells, 5).unwrap();

    // i=0 (p-): n/g=1/2=0.5, 2Q/u=1/1=1.0, 2J/U=1/1=1.0
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] - 1.0).abs() < 1e-5);
    // i=1 (d ): n/g=2/6=1/3, 2Q/u=6/8=0.75, 2J/U=5/5=1.0
    assert!((result[3] - (1.0 / 3.0)).abs() < 1e-5);
    assert!((result[4] - 0.75).abs() < 1e-5);
    assert!((result[5] - 1.0).abs() < 1e-5);
}
