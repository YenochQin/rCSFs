//! CSF Descriptor Normalization Module
//!
//! This module provides functions for normalizing CSF descriptor values.
//! Normalization helps improve machine learning model performance by scaling
//! values to a consistent range.

use anyhow::{Context, Result};
use std::collections::HashMap;

/// Convert full subshell notation (e.g., "2s", "3p-") to angular notation (e.g., "s ", "p-")
///
/// This function strips the principal quantum number (n) from subshell identifiers,
/// keeping only the angular momentum quantum number (l) and parity indicator.
///
/// # Arguments
/// * `full_subshell` - Full subshell identifier with principal quantum number (e.g., "2s", "3p-")
///
/// # Returns
/// Angular momentum notation (e.g., "s " for s-orbitals, "p-" for p- orbitals)
///
/// # Conversion Rules
/// - "2s" → "s " (s-orbitals have a trailing space for consistency)
/// - "2p-" → "p-" (negative parity p-orbitals keep the minus sign)
/// - "2p" → "p " (positive parity p-orbitals have a trailing space)
/// - "3d-" → "d-" (negative parity d-orbitals)
/// - "3d" → "d " (positive parity d-orbitals)
/// - "4f-" → "f-" (negative parity f-orbitals)
/// - "4f" → "f " (positive parity f-orbitals)
///
/// # Examples
/// ```text
/// convert_full_to_angular("2s")  => "s "
/// convert_full_to_angular("2p-") => "p-"
/// convert_full_to_angular("3p")  => "p "
/// convert_full_to_angular("3d-") => "d-"
/// convert_full_to_angular("4d")  => "d "
/// convert_full_to_angular("5f-") => "f-"
/// ```
pub fn convert_full_to_angular(full_subshell: &str) -> String {
    let trimmed = full_subshell.trim();

    // Find the first alphabetic character (start of angular momentum label)
    let alpha_start = trimmed
        .chars()
        .position(|c| c.is_alphabetic())
        .unwrap_or(0);

    // Get everything from the first alphabetic character onward
    let angular_part = &trimmed[alpha_start..];

    // The normalization module expects specific format:
    // - Single letter orbitals (s, p, d, f, g, h, i) need trailing space: "s ", "p ", "d ", etc.
    // - Orbitals with minus sign keep the minus without trailing space: "p-", "d-", etc.
    if angular_part.len() == 1 {
        // Single letter orbital (positive parity) - add trailing space
        format!("{} ", angular_part)
    } else {
        // Has minus sign or other characters - keep as-is
        angular_part.to_string()
    }
}

/// Convert a list of full subshell notations to angular notations
///
/// # Arguments
/// * `full_subshells` - List of full subshell identifiers (e.g., ["2s", "2p-", "2p"])
///
/// # Returns
/// List of angular momentum notations (e.g., ["s ", "p-", "p "])
///
/// # Examples
/// ```text
/// convert_full_to_angular_list(&["2s", "2p-", "2p", "3s"])
/// => ["s ", "p-", "p ", "s "]
/// ```
pub fn convert_full_to_angular_list(full_subshells: &[String]) -> Vec<String> {
    full_subshells
        .iter()
        .map(|s| convert_full_to_angular(s))
        .collect()
}

/// Get the maximum electron capacity for a given subshell type
///
/// Subshell strings must match exactly (including whitespace):
/// - "s " for s-orbital
/// - "p-" for p-orbital with negative parity
/// - "p " for p-orbital
/// - "d-" for d-orbital with negative parity
/// - etc.
///
/// # Arguments
/// * `subshell` - Subshell identifier string (exact match required)
///
/// # Returns
/// * `Some(f32)` - Maximum electron capacity for the subshell
/// * `None` - Unknown subshell type
///
/// # Examples
/// ```text
/// get_max_subshell_electrons("s ")  => Some(2.0)
/// get_max_subshell_electrons("p-")  => Some(2.0)
/// get_max_subshell_electrons("d ")  => Some(6.0)
/// get_max_subshell_electrons("xyz") => None
/// ```
pub fn get_max_subshell_electrons(subshell: &str) -> Option<f32> {
    // Subshell to max electron mapping (exact string match required)
    // For relativistic orbitals: max electrons = 2j + 1 where j = l ± 1/2
    let limits: HashMap<&str, f32> = HashMap::from([
        ("s ", 2.0),   // s-orbital (j=1/2): 2j+1 = 2
        ("p-", 2.0),   // p- orbital (l=1, j=1/2): 2j+1 = 2
        ("p ", 4.0),   // p-orbital (l=1, j=3/2): 2j+1 = 4
        ("d-", 4.0),   // d- orbital (l=2, j=3/2): 2j+1 = 4
        ("d ", 6.0),   // d-orbital (l=2, j=5/2): 2j+1 = 6
        ("f-", 6.0),   // f- orbital (l=3, j=5/2): 2j+1 = 6
        ("f ", 8.0),   // f-orbital (l=3, j=7/2): 2j+1 = 8
        ("g-", 8.0),   // g- orbital (l=4, j=7/2): 2j+1 = 8
        ("g ", 10.0),  // g-orbital (l=4, j=9/2): 2j+1 = 10
        ("h-", 10.0),  // h- orbital (l=5, j=9/2): 2j+1 = 10
        ("h ", 12.0),  // h-orbital (l=5, j=11/2): 2j+1 = 12
        ("i-", 12.0),  // i- orbital (l=6, j=11/2): 2j+1 = 12
        ("i ", 14.0),  // i-orbital (l=6, j=13/2): 2j+1 = 14
    ]);

    limits.get(subshell).copied()
}

/// Get the kappa squared value for a given subshell type
///
/// In relativistic atomic physics, kappa is the quantum number related to
/// the total angular momentum j and orbital angular momentum l:
/// - kappa = -(l + 1) for j = l - 1/2 (negative parity orbitals like "p-", "d-")
/// - kappa = l for j = l + 1/2 (positive parity orbitals like "s ", "p ", "d ")
///
/// This function returns kappa², which is always positive.
///
/// # Arguments
/// * `subshell` - Subshell identifier string (exact match required)
///
/// # Returns
/// * `Some(i32)` - Kappa squared value for the subshell
/// * `None` - Unknown subshell type
///
/// # Examples
/// ```text
/// get_kappa_squared("s ")  => Some(1)   // kappa = -1, kappa² = 1
/// get_kappa_squared("p-")  => Some(1)   // kappa = 1, kappa² = 1
/// get_kappa_squared("p ")  => Some(4)   // kappa = -2, kappa² = 4
/// get_kappa_squared("d ")  => Some(9)   // kappa = -3, kappa² = 9
/// get_kappa_squared("f ")  => Some(16)  // kappa = -4, kappa² = 16
/// get_kappa_squared("xyz") => None
/// ```
pub fn get_kappa_squared(subshell: &str) -> Option<i32> {
    let kappa: HashMap<&str, i32> = HashMap::from([
        ("s ", -1),
        ("p-", 1),
        ("p ", -2),
        ("d-", 2),
        ("d ", -3),
        ("f-", 3),
        ("f ", -4),
        ("g-", 4),
        ("g ", -5),
        ("h-", 5),
        ("h ", -6),
        ("i-", 6),
        ("i ", -7),
    ]);

    kappa.get(subshell).map(|&k| k * k)
}

/// Normalize electron count for a subshell
///
/// Computes the normalized value as: `num_electrons / max_subshell_electrons`
///
/// # Arguments
/// * `num_electrons` - Number of electrons in the subshell
/// * `subshell` - Subshell identifier string (exact match required)
///
/// # Returns
/// * `Ok(f32)` - Normalized value (may exceed 1.0 if electrons exceed max)
/// * `Err(String)` - Error message if subshell is unknown
///
/// # Examples
/// ```text
/// normalize_electron_count(2, "s ")  => Ok(1.0)   // 2/2 = 1.0
/// normalize_electron_count(6, "d ")  => Ok(1.0)   // 6/6 = 1.0
/// normalize_electron_count(3, "p ")  => Ok(0.75)  // 3/4 < 1 (partially filled)
/// normalize_electron_count(5, "xyz") => Err("Unknown subshell: xyz")
/// ```
pub fn normalize_electron_count(num_electrons: i32, subshell: &str) -> Result<f32> {
    let max_electrons = get_max_subshell_electrons(subshell)
        .ok_or_else(|| anyhow::anyhow!("Unknown subshell: {}", subshell))?;

    if max_electrons <= 0.0 {
        return Err(anyhow::anyhow!("Invalid max electrons for subshell {}: {}", subshell, max_electrons));
    }

    Ok(num_electrons as f32 / max_electrons)
}


/// Normalize a descriptor array using per-CSF physics-correct denominators
///
/// This implements the recommended normalization scheme from the normalization analysis:
///
/// - `n_i`       (descriptor[3*i])   → divided by `g_i`                    (unchanged)
/// - `2Q_i`      (descriptor[3*i+1]) → divided by `n_i * (g_i - n_i)`      (tighter, per-CSF)
/// - `2J_cum,i`  (descriptor[3*i+2]) → divided by `min(prefix_i, 2J_target + suffix_i)` (position-dependent)
///
/// Where:
/// - `g_i = 2|kappa_i|` is the subshell capacity (maximum electrons)
/// - `u_i = n_i * (g_i - n_i)` is the occupation-dependent upper bound on `2Q_i`
/// - `prefix_i = sum_{k=0}^{i} u_k` (cumulative bound from the front)
/// - `suffix_i = sum_{k=i+1}^{N-1} u_k` (cumulative bound from the rear)
///
/// Zero-division is handled safely: when the denominator is 0 the corresponding
/// field is physically forced to 0 and `0.0` is output directly.
///
/// # Arguments
/// * `descriptor`    - Raw descriptor array of length `3 * peel_subshells.len()`
/// * `peel_subshells` - Ordered list of subshell names (full or angular notation)
/// * `two_j_target`  - Doubled total angular momentum `2J` of the target state;
///                     for a single-Jpi block this is the block's fixed J value;
///                     call with `descriptor[descriptor.len() - 1]` when reading
///                     directly from a parsed CSF (the value is always stored there
///                     by `parse_csf`).
///
/// # Returns
/// * `Ok(Vec<f32>)` - Normalized descriptor (same length as `descriptor`, values in `[0, 1]`)
/// * `Err(anyhow::Error)` - If length mismatch or unknown subshell
pub fn normalize_descriptor_per_csf(
    descriptor: &[i32],
    peel_subshells: &[String],
    two_j_target: i32,
) -> Result<Vec<f32>> {
    let n = peel_subshells.len();
    if descriptor.len() != 3 * n {
        return Err(anyhow::anyhow!(
            "Descriptor length mismatch: expected {}, got {}",
            3 * n,
            descriptor.len()
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Step 1: Compute g_i for each subshell (static, depends only on subshell type)
    let g: Vec<i32> = peel_subshells
        .iter()
        .map(|s| {
            let angular = convert_full_to_angular(s);
            get_max_subshell_electrons(&angular)
                .ok_or_else(|| anyhow::anyhow!("Unknown subshell: {}", s))
                .map(|v| v as i32)
        })
        .collect::<Result<Vec<_>>>()?;

    // Step 2: Extract n_i from descriptor and compute u_i = n_i * (g_i - n_i)
    let n_elec: Vec<i32> = (0..n).map(|i| descriptor[3 * i]).collect();
    let u: Vec<i32> = (0..n).map(|i| n_elec[i] * (g[i] - n_elec[i])).collect();

    // Step 3: Prefix sums  prefix[i] = sum_{k=0}^{i} u[k]
    let mut prefix = vec![0i32; n];
    prefix[0] = u[0];
    for i in 1..n {
        prefix[i] = prefix[i - 1] + u[i];
    }

    // Step 4: Suffix sums  suffix[i] = sum_{k=i+1}^{n-1} u[k]   (suffix[n-1] = 0)
    let mut suffix = vec![0i32; n];
    if n >= 2 {
        suffix[n - 2] = u[n - 1];
        for i in (0..n - 2).rev() {
            suffix[i] = suffix[i + 1] + u[i + 1];
        }
    }

    // Step 5: Build normalized output
    let mut result = vec![0.0f32; 3 * n];
    for i in 0..n {
        let gi = g[i];
        let ui = u[i];

        // n_i / g_i  (g_i is always > 0 for any known subshell)
        result[3 * i] = n_elec[i] as f32 / gi as f32;

        // 2Q_i / (n_i * (g_i - n_i))  — 0 when subshell is empty or full
        result[3 * i + 1] = if ui == 0 {
            0.0
        } else {
            descriptor[3 * i + 1] as f32 / ui as f32
        };

        // 2J_cum,i / U_i_occ   where U_i_occ = min(prefix_i, 2J_target + suffix_i)
        let u_i_occ = prefix[i].min(two_j_target + suffix[i]);
        result[3 * i + 2] = if u_i_occ == 0 {
            0.0
        } else {
            descriptor[3 * i + 2] as f32 / u_i_occ as f32
        };
    }

    Ok(result)
}

/// Batch normalize multiple descriptor arrays
///
/// # Arguments
/// * `descriptors` - Vector of descriptor arrays
/// * `peel_subshells` - List of subshell names in order
/// * `two_j_target` - Doubled target angular momentum applied to **every** descriptor
///
/// # Returns
/// * `Ok(Vec<Vec<f32>>)` - Vector of normalized descriptor arrays
/// * `Err(anyhow::Error)` - Error if any normalization fails
///
/// # Warning
/// A single `two_j_target` is applied to all descriptors in the batch. This is
/// correct for single-Jpi datasets but **wrong for mixed-Jpi data**, where
/// different CSFs have different target J values. For mixed-Jpi datasets call
/// `normalize_descriptor_per_csf` per CSF, reading
/// `two_j_target = descriptor[descriptor.len() - 1]` from each individual CSF.
pub fn batch_normalize_descriptors(
    descriptors: &[Vec<i32>],
    peel_subshells: &[String],
    two_j_target: i32,
) -> Result<Vec<Vec<f32>>> {
    descriptors
        .iter()
        .enumerate()
        .map(|(idx, desc)| {
            normalize_descriptor_per_csf(desc, peel_subshells, two_j_target)
                .with_context(|| format!("Failed to normalize descriptor at index {}", idx))
        })
        .collect()
}

/// Get all supported subshell types and their max electron capacities
///
/// # Returns
/// HashMap mapping subshell identifiers to maximum electron capacities
pub fn get_all_subshell_limits() -> HashMap<String, f32> {
    HashMap::from([
        ("s ".to_string(), 2.0),
        ("p-".to_string(), 2.0),
        ("p ".to_string(), 4.0),
        ("d-".to_string(), 4.0),
        ("d ".to_string(), 6.0),
        ("f-".to_string(), 6.0),
        ("f ".to_string(), 8.0),
        ("g-".to_string(), 8.0),
        ("g ".to_string(), 10.0),
        ("h-".to_string(), 10.0),
        ("h ".to_string(), 12.0),
        ("i-".to_string(), 12.0),
        ("i ".to_string(), 14.0),
    ])
}

//////////////////////////////////////////////////////////////////////////////
/// Rust Tests
//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_full_to_angular() {
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
        // Test list conversion
        let full = vec!["2s".to_string(), "2p-".to_string(), "2p".to_string(), "3s".to_string()];
        let angular = convert_full_to_angular_list(&full);
        assert_eq!(angular, vec!["s ", "p-", "p ", "s "]);

        // Test with more complex list
        let full = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string(), "5p-".to_string()];
        let angular = convert_full_to_angular_list(&full);
        assert_eq!(angular, vec!["s ", "d-", "d ", "p-"]);
    }

    #[test]
    fn test_get_max_subshell_electrons() {
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
    fn test_normalize_electron_count() {
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
        let descriptors = vec![
            vec![1, 1, 1, 3, 9, 6],
            vec![1, 1, 1, 1, 5, 6],
        ];
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
        let limits = get_all_subshell_limits();

        assert_eq!(limits.get("s "), Some(&2.0));
        assert_eq!(limits.get("p-"), Some(&2.0));
        assert_eq!(limits.get("d "), Some(&6.0));
        assert_eq!(limits.get("f-"), Some(&6.0));
    }

    // -----------------------------------------------------------------------
    // Tests for normalize_descriptor_per_csf
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_descriptor_per_csf_s_orbital() {
        // s orbital: g=2, n=1, u=1*(2-1)=1
        // prefix[0]=1, suffix[0]=0
        // U_0 = min(1, 1+0) = 1
        // n_i/g = 0.5, 2Q/u = 1/1 = 1.0, 2J/U = 1/1 = 1.0
        let descriptor = vec![1, 1, 1];
        let subshells = vec!["s ".to_string()];
        let result = normalize_descriptor_per_csf(&descriptor, &subshells, 1).unwrap();
        assert!((result[0] - 0.5).abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-5);
        assert!((result[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_descriptor_per_csf_empty_subshell() {
        // n=0: u=0 → 2Q and 2J_cum fields must be 0.0
        let descriptor = vec![0, 0, 0];
        let subshells = vec!["p ".to_string()];
        let result = normalize_descriptor_per_csf(&descriptor, &subshells, 0).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_descriptor_per_csf_full_subshell() {
        // d orbital full: n=6, g=6, u=6*(6-6)=0 → 2Q=0 and 2J_cum=0
        let descriptor = vec![6, 0, 0];
        let subshells = vec!["d ".to_string()];
        let result = normalize_descriptor_per_csf(&descriptor, &subshells, 0).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-5); // n/g = 6/6
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_descriptor_per_csf_d_half_filled() {
        // d orbital half-filled: n=3, g=6, u=3*(6-3)=9
        // At half-filling u_i == kappa_i^2, so 2Q_max/u = 9/9 = 1.0
        // prefix[0]=9, suffix[0]=0, 2J_target=9
        // U_0 = min(9, 9+0)=9
        let descriptor = vec![3, 9, 9];
        let subshells = vec!["d ".to_string()];
        let result = normalize_descriptor_per_csf(&descriptor, &subshells, 9).unwrap();
        assert!((result[0] - 0.5).abs() < 1e-5); // 3/6
        assert!((result[1] - 1.0).abs() < 1e-5); // 9/9
        assert!((result[2] - 1.0).abs() < 1e-5); // 9/9
    }

    #[test]
    fn test_normalize_descriptor_per_csf_two_subshells() {
        // Two s orbitals: g=[2,2], n=[1,1], u=[1,1]
        // prefix=[1,2], suffix=[1,0]
        // 2J_target=2
        // i=0: U=min(1, 2+1)=1  → 2J_cum[0]/1
        // i=1: U=min(2, 2+0)=2  → 2J_cum[1]/2
        let descriptor = vec![1, 1, 1,  1, 1, 2];
        let subshells = vec!["s ".to_string(), "s ".to_string()];
        let result = normalize_descriptor_per_csf(&descriptor, &subshells, 2).unwrap();
        // i=0
        assert!((result[0] - 0.5).abs() < 1e-5); // 1/2
        assert!((result[1] - 1.0).abs() < 1e-5); // 1/1
        assert!((result[2] - 1.0).abs() < 1e-5); // 1/min(1,3)=1/1
        // i=1
        assert!((result[3] - 0.5).abs() < 1e-5); // 1/2
        assert!((result[4] - 1.0).abs() < 1e-5); // 1/1
        assert!((result[5] - 1.0).abs() < 1e-5); // 2/min(2,2)=2/2
    }

    #[test]
    fn test_normalize_descriptor_per_csf_last_position_is_one() {
        // When prefix[N-1] >= 2J_target, the last coupling always normalizes to 1.0
        // p orbital: n=2, g=4, u=2*(4-2)=4; 2J_target=4
        // prefix[0]=4, suffix[0]=0
        // U_0 = min(4, 4) = 4
        // last coupling = 2J_target = 4 → 4/4 = 1.0
        let descriptor = vec![2, 4, 4];
        let subshells = vec!["p ".to_string()];
        let result = normalize_descriptor_per_csf(&descriptor, &subshells, 4).unwrap();
        assert!((result[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_descriptor_per_csf_length_mismatch() {
        let descriptor = vec![1, 1]; // too short
        let subshells = vec!["s ".to_string()];
        assert!(normalize_descriptor_per_csf(&descriptor, &subshells, 1).is_err());
    }

    #[test]
    fn test_normalize_descriptor_per_csf_empty_descriptor() {
        let descriptor = vec![];
        let subshells = vec![];
        let result = normalize_descriptor_per_csf(&descriptor, &subshells, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_normalize_descriptor_per_csf_full_notation_subshells() {
        // Full notation "2s" should be auto-converted and produce identical results
        let descriptor = vec![1, 1, 1];
        let result_angular = normalize_descriptor_per_csf(&descriptor, &vec!["s ".to_string()], 1).unwrap();
        let result_full    = normalize_descriptor_per_csf(&descriptor, &vec!["2s".to_string()], 1).unwrap();
        for (a, b) in result_angular.iter().zip(result_full.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_descriptor_per_csf_rear_constraint_binding() {
        // subshells = ["f ", "s "]  g=[8, 2]
        // n=[4, 1], u=[4*(8-4)=16, 1*(2-1)=1]
        // prefix=[16, 17], suffix=[1, 0]
        // two_j_target=2
        // i=0 (f): U=min(16, 2+1)=3  ← rear constraint is binding (3 << prefix=16)
        // i=1 (s): U=min(17, 2+0)=2  ← rear constraint is binding
        let descriptor = vec![4, 12, 3,  1, 1, 2];
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
        // subshells = ["p-", "d "]  g=[2, 6]
        // n=[1, 2], u=[1*(2-1)=1, 2*(6-2)=8]
        // prefix=[1, 9], suffix=[8, 0]
        // two_j_target=5
        // i=0 (p-): U=min(1, 5+8)=1  ← front constraint binding
        // i=1 (d ): U=min(9, 5+0)=5  ← rear constraint binding (5 < prefix=9)
        let descriptor = vec![1, 1, 1,  2, 6, 5];
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
}
