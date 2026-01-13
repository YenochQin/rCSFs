//! CSF Descriptor Normalization Module
//!
//! This module provides functions for normalizing CSF descriptor values.
//! Normalization helps improve machine learning model performance by scaling
//! values to a consistent range.

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
pub fn normalize_electron_count(num_electrons: i32, subshell: &str) -> Result<f32, String> {
    let max_electrons = get_max_subshell_electrons(subshell)
        .ok_or_else(|| format!("Unknown subshell: {}", subshell))?;

    if max_electrons <= 0.0 {
        return Err(format!("Invalid max electrons for subshell {}: {}", subshell, max_electrons));
    }

    Ok(num_electrons as f32 / max_electrons)
}

/// Get subshell properties as an array: [max_electrons, kappa_squared, max_cumulative_doubled_j]
///
/// This function combines multiple subshell properties into a single array for convenience.
///
/// # Arguments
/// * `subshell` - Subshell identifier string (exact match required)
/// * `max_cumulative_doubled_j` - Maximum cumulative 2J value (typically an integer)
///
/// # Returns
/// * `Ok([i32; 3])` - Array containing [max_electrons, kappa_squared, max_cumulative_doubled_j]
/// * `Err(String)` - Error message if subshell is unknown
///
/// # Examples
/// ```text
/// get_subshell_properties("s ", 10)  => Ok([2, 1, 10])
/// get_subshell_properties("p ", 20)  => Ok([4, 4, 20])
/// get_subshell_properties("d ", 30)  => Ok([6, 9, 30])
/// get_subshell_properties("xyz", 10) => Err("Unknown subshell: xyz")
/// ```
pub fn get_subshell_properties(subshell: &str, max_cumulative_doubled_j: i32) -> Result<[i32; 3], String> {
    let max_electrons = get_max_subshell_electrons(subshell)
        .ok_or_else(|| format!("Unknown subshell: {}", subshell))?;
    let kappa_sq = get_kappa_squared(subshell)
        .ok_or_else(|| format!("Unknown subshell: {}", subshell))?;

    Ok([max_electrons as i32, kappa_sq, max_cumulative_doubled_j])
}

/// Get subshell properties for multiple subshells in order
///
/// This function iterates through the subshells list in order and concatenates
/// the results from get_subshell_properties for each subshell.
///
/// The function automatically converts full notation (e.g., "2s", "3p-") to
/// angular notation (e.g., "s ", "p-") before looking up properties.
///
/// # Arguments
/// * `subshells` - List of subshell identifier strings (order is preserved)
///                 Can be in full notation (e.g., "2s", "2p-") or angular notation (e.g., "s ", "p-")
/// * `max_cumulative_doubled_j` - Maximum cumulative 2J value (applied to all subshells)
///
/// # Returns
/// * `Ok(Vec<i32>)` - Flattened array containing [max_e1, kappa1, 2J1, max_e2, kappa2, 2J2, ...]
/// * `Err(String)` - Error message if any subshell is unknown
///
/// # Examples
/// ```text
/// // Using angular notation (original format)
/// subshells = ["s ", "p ", "d "]
/// get_subshells_properties(subshells, 10)
/// => Ok([2, 1, 10,  // s: max=2, kappa_sq=1, 2J=10
///       4, 4, 10,   // p: max=4, kappa_sq=4, 2J=10
///       6, 9, 10])  // d: max=6, kappa_sq=9, 2J=10
///
/// // Using full notation (auto-converted)
/// subshells = ["2s", "2p", "3d"]
/// get_subshells_properties(subshells, 10)
/// => Ok([2, 1, 10,  // s: max=2, kappa_sq=1, 2J=10
///       4, 4, 10,   // p: max=4, kappa_sq=4, 2J=10
///       6, 9, 10])  // d: max=6, kappa_sq=9, 2J=10
/// ```
pub fn get_subshells_properties(
    subshells: &[String],
    max_cumulative_doubled_j: i32,
) -> Result<Vec<i32>, String> {
    let mut result = Vec::with_capacity(subshells.len() * 3);

    for subshell in subshells {
        // Convert full notation to angular notation if needed
        // This handles both "2s" -> "s " and "s " -> "s " cases
        let angular_subshell = convert_full_to_angular(subshell);
        let props = get_subshell_properties(&angular_subshell, max_cumulative_doubled_j)?;
        result.extend_from_slice(&props);
    }

    Ok(result)
}

/// Calculate reciprocal of each element in the subshell properties array
///
/// This function takes the output from get_subshells_properties and computes
/// the reciprocal (1/x) of each element.
///
/// # Arguments
/// * `properties` - Array of subshell properties (output from get_subshells_properties)
///
/// # Returns
/// * `Ok(Vec<f32>)` - Array containing reciprocals [1/x1, 1/x2, ...]
/// * `Err(String)` - Error message if any element is zero (division by zero)
///
/// # Examples
/// ```text
/// properties = [2, 1, 10, 4, 4, 10]
///
/// compute_properties_reciprocals(properties)
/// => Ok([0.5, 1.0, 0.1,   // 1/2, 1/1, 1/10
///       0.25, 0.25, 0.1]) // 1/4, 1/4, 1/10
/// ```
pub fn compute_properties_reciprocals(properties: &[i32]) -> Result<Vec<f32>, String> {
    let mut result = Vec::with_capacity(properties.len());

    for (idx, &val) in properties.iter().enumerate() {
        if val == 0 {
            return Err(format!("Cannot compute reciprocal: element at index {} is zero", idx));
        }
        result.push(1.0 / val as f32);
    }

    Ok(result)
}

/// Normalize a descriptor array using subshell properties
///
/// Each descriptor contains triplets of [n_electrons, J_middle, J_coupling] for each orbital.
/// This function normalizes all values by multiplying with the reciprocal of subshell properties.
///
/// # Arguments
/// * `descriptor` - Descriptor array to normalize
/// * `peel_subshells` - List of subshell names in order (must match descriptor length)
/// * `max_cumulative_doubled_j` - Maximum cumulative 2J value for normalization
///
/// # Returns
/// * `Ok(Vec<f32>)` - Normalized descriptor array (same size as input)
/// * `Err(String)` - Error message if normalization fails
///
/// # Examples
/// ```text
/// descriptor = [2, 3, 4, 6, 3, 8]  // 2 orbitals: [e1, J1, Jc1, e2, J2, Jc2]
/// subshells = ["s ", "d "]
/// max_cumulative_doubled_j = 10
///
/// // get_subshells_properties => [2, 1, 10, 6, 9, 10]
/// // reciprocals => [0.5, 1.0, 0.1, 0.167, 0.111, 0.1]
///
/// normalize_descriptor(descriptor, subshells, 10)
/// => [1.0, 3.0, 0.4, 1.0, 0.333, 0.8]  // element-wise multiplication
/// ```
pub fn normalize_descriptor(
    descriptor: &[i32],
    peel_subshells: &[String],
    max_cumulative_doubled_j: i32,
) -> Result<Vec<f32>, String> {
    if descriptor.len() != 3 * peel_subshells.len() {
        return Err(format!(
            "Descriptor length mismatch: expected {}, got {}",
            3 * peel_subshells.len(),
            descriptor.len()
        ));
    }

    // Get normalization denominators from subshell properties
    let denominators = get_subshells_properties(peel_subshells, max_cumulative_doubled_j)?;

    // Compute reciprocals for normalization
    let reciprocals = compute_properties_reciprocals(&denominators)?;

    // Element-wise multiplication: descriptor * reciprocals
    let normalized: Vec<f32> = descriptor
        .iter()
        .zip(reciprocals.iter())
        .map(|(&d, &r)| d as f32 * r)
        .collect();

    Ok(normalized)
}

/// Batch normalize multiple descriptor arrays
///
/// # Arguments
/// * `descriptors` - Vector of descriptor arrays
/// * `peel_subshells` - List of subshell names in order
/// * `max_cumulative_doubled_j` - Maximum cumulative 2J value for normalization
///
/// # Returns
/// * `Ok(Vec<Vec<f32>>)` - Vector of normalized descriptor arrays
/// * `Err(String)` - Error message if any normalization fails
pub fn batch_normalize_descriptors(
    descriptors: &[Vec<i32>],
    peel_subshells: &[String],
    max_cumulative_doubled_j: i32,
) -> Result<Vec<Vec<f32>>, String> {
    descriptors
        .iter()
        .enumerate()
        .map(|(idx, desc)| {
            normalize_descriptor(desc, peel_subshells, max_cumulative_doubled_j)
                .map_err(|e| format!("Failed to normalize descriptor at index {}: {}", idx, e))
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
    fn test_get_subshells_properties_with_full_notation() {
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
    fn test_get_subshell_properties() {
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
    fn test_compute_properties_reciprocals() {
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

    #[test]
    fn test_normalize_electron_count() {
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
        let descriptor = vec![2, 3, 4, 6]; // Wrong length (should be 6 for 2 orbitals)
        let subshells = vec!["s ".to_string(), "d ".to_string()];
        let max_cumulative_doubled_j = 10;

        assert!(normalize_descriptor(&descriptor, &subshells, max_cumulative_doubled_j).is_err());
    }

    #[test]
    fn test_batch_normalize_descriptors() {
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
        let limits = get_all_subshell_limits();

        assert_eq!(limits.get("s "), Some(&2.0));
        assert_eq!(limits.get("p-"), Some(&2.0));
        assert_eq!(limits.get("d "), Some(&6.0));
        assert_eq!(limits.get("f-"), Some(&6.0));
    }
}
