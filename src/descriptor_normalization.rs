//! CSF Descriptor Normalization Module
//!
//! This module provides functions for normalizing CSF descriptor values.
//! Normalization helps improve machine learning model performance by scaling
//! values to a consistent range.

use std::collections::HashMap;

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

/// Get the half-filled electron capacity for a given subshell type
///
/// For relativistic orbitals, this returns half of the maximum (2j+1) capacity.
/// Half-filled configurations are often particularly stable in atomic physics.
///
/// # Arguments
/// * `subshell` - Subshell identifier string (exact match required)
///
/// # Returns
/// * `Some(f32)` - Half-filled electron capacity for the subshell
/// * `None` - Unknown subshell type
///
/// # Examples
/// ```text
/// get_half_filled_electrons("s ")  => Some(1.0)  // 2/2 = 1
/// get_half_filled_electrons("p-")  => Some(1.0)  // 2/2 = 1
/// get_half_filled_electrons("p ")  => Some(2.0)  // 4/2 = 2
/// get_half_filled_electrons("d ")  => Some(3.0)  // 6/2 = 3
/// get_half_filled_electrons("f ")  => Some(4.0)  // 8/2 = 4
/// get_half_filled_electrons("xyz") => None
/// ```
pub fn get_half_filled_electrons(subshell: &str) -> Option<f32> {
    get_max_subshell_electrons(subshell).map(|max| max / 2.0)
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

/// Normalize a descriptor array using subshell information
///
/// Each descriptor contains triplets of [n_electrons, J_middle, J_coupling] for each orbital.
/// This function normalizes only the electron count values (every 3rd element starting at 0).
///
/// # Arguments
/// * `descriptor` - Descriptor array to normalize
/// * `peel_subshells` - List of subshell names in order (must match descriptor length)
///
/// # Returns
/// * `Ok(Vec<f32>)` - Normalized descriptor array (same size as input)
/// * `Err(String)` - Error message if normalization fails
///
/// # Examples
/// ```text
/// descriptor = [2, 3, 4, 6, 3, 8]  // 2 orbitals: [e1, J1, Jc1, e2, J2, Jc2]
/// subshells = ["s ", "d "]
///
/// normalize_descriptor(descriptor, subshells)
/// => [1.0, 3.0, 4.0, 1.0, 3.0, 8.0]  // 2/2=1.0, 6/6=1.0 (only e values normalized)
/// ```
pub fn normalize_descriptor(
    descriptor: &[i32],
    peel_subshells: &[String],
) -> Result<Vec<f32>, String> {
    if descriptor.len() != 3 * peel_subshells.len() {
        return Err(format!(
            "Descriptor length mismatch: expected {}, got {}",
            3 * peel_subshells.len(),
            descriptor.len()
        ));
    }

    let mut normalized = Vec::with_capacity(descriptor.len());

    for (orbital_idx, subshell) in peel_subshells.iter().enumerate() {
        let base_idx = orbital_idx * 3;

        // Normalize electron count (position 0 in each triplet)
        let num_electrons = descriptor[base_idx];
        let normalized_electrons = normalize_electron_count(num_electrons, subshell)?;

        // Copy J_middle and J_coupling as-is (positions 1 and 2)
        let j_middle = descriptor[base_idx + 1] as f32;
        let j_coupling = descriptor[base_idx + 2] as f32;

        normalized.push(normalized_electrons);
        normalized.push(j_middle);
        normalized.push(j_coupling);
    }

    Ok(normalized)
}

/// Batch normalize multiple descriptor arrays
///
/// # Arguments
/// * `descriptors` - Vector of descriptor arrays
/// * `peel_subshells` - List of subshell names in order
///
/// # Returns
/// * `Ok(Vec<Vec<f32>>)` - Vector of normalized descriptor arrays
/// * `Err(String)` - Error message if any normalization fails
pub fn batch_normalize_descriptors(
    descriptors: &[Vec<i32>],
    peel_subshells: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    descriptors
        .iter()
        .enumerate()
        .map(|(idx, desc)| {
            normalize_descriptor(desc, peel_subshells)
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
/// Python Bindings (PyO3)
//////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-exposed function to get max electrons for a subshell
#[cfg(feature = "python")]
#[pyfunction]
fn py_get_max_subshell_electrons(subshell: String) -> PyResult<f32> {
    get_max_subshell_electrons(&subshell)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Unknown subshell: {}", subshell)))
}

/// Python-exposed function to normalize electron count
///
/// Args:
///     num_electrons: Number of electrons in the subshell
///     subshell: Subshell identifier string (e.g., "s ", "p-", "d ")
///
/// Returns:
///     Normalized value (num_electrons / max_subshell_electrons)
#[cfg(feature = "python")]
#[pyfunction]
fn py_normalize_electron_count(num_electrons: i32, subshell: String) -> PyResult<f32> {
    normalize_electron_count(num_electrons, &subshell)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Python-exposed function to normalize a descriptor array
///
/// Args:
///     descriptor: Descriptor array to normalize
///     peel_subshells: List of subshell names in order
///
/// Returns:
///     Normalized descriptor array (electron counts normalized)
#[cfg(feature = "python")]
#[pyfunction]
fn py_normalize_descriptor(descriptor: Vec<i32>, peel_subshells: Vec<String>) -> PyResult<Vec<f32>> {
    normalize_descriptor(&descriptor, &peel_subshells)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Python-exposed function to batch normalize descriptors
///
/// Args:
///     descriptors: List of descriptor arrays
///     peel_subshells: List of subshell names in order
///
/// Returns:
///     List of normalized descriptor arrays
#[cfg(feature = "python")]
#[pyfunction]
fn py_batch_normalize_descriptors(
    descriptors: Vec<Vec<i32>>,
    peel_subshells: Vec<String>,
) -> PyResult<Vec<Vec<f32>>> {
    batch_normalize_descriptors(&descriptors, &peel_subshells)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Python-exposed function to get all subshell limits
///
/// Returns:
///     Dictionary mapping subshell identifiers to max electron capacities
#[cfg(feature = "python")]
#[pyfunction]
fn py_get_all_subshell_limits(py: Python) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    let limits = get_all_subshell_limits();
    let dict = pyo3::types::PyDict::new(py);
    for (key, value) in limits {
        dict.set_item(key, value)?;
    }
    Ok(dict.into())
}

/// Register the normalization module functions
#[cfg(feature = "python")]
pub fn register_normalization_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_get_max_subshell_electrons, module)?)?;
    module.add_function(wrap_pyfunction!(py_normalize_electron_count, module)?)?;
    module.add_function(wrap_pyfunction!(py_normalize_descriptor, module)?)?;
    module.add_function(wrap_pyfunction!(py_batch_normalize_descriptors, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_all_subshell_limits, module)?)?;
    Ok(())
}

//////////////////////////////////////////////////////////////////////////////
/// Rust Tests
//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_get_half_filled_electrons() {
        assert_eq!(get_half_filled_electrons("s "), Some(1.0));  // 2/2
        assert_eq!(get_half_filled_electrons("p-"), Some(1.0));  // 2/2
        assert_eq!(get_half_filled_electrons("p "), Some(2.0));  // 4/2
        assert_eq!(get_half_filled_electrons("d-"), Some(2.0));  // 4/2
        assert_eq!(get_half_filled_electrons("d "), Some(3.0));  // 6/2
        assert_eq!(get_half_filled_electrons("f-"), Some(3.0));  // 6/2
        assert_eq!(get_half_filled_electrons("f "), Some(4.0));  // 8/2
        assert_eq!(get_half_filled_electrons("g-"), Some(4.0));  // 8/2
        assert_eq!(get_half_filled_electrons("g "), Some(5.0));  // 10/2
        assert_eq!(get_half_filled_electrons("h-"), Some(5.0));  // 10/2
        assert_eq!(get_half_filled_electrons("h "), Some(6.0));  // 12/2
        assert_eq!(get_half_filled_electrons("i-"), Some(6.0));  // 12/2
        assert_eq!(get_half_filled_electrons("i "), Some(7.0));  // 14/2
        assert_eq!(get_half_filled_electrons("xyz"), None);
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

        let result = normalize_descriptor(&descriptor, &subshells).unwrap();

        // First orbital (s): 2/2 = 1.0, J values unchanged
        assert!((result[0] - 1.0).abs() < 0.01);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);

        // Second orbital (d): 6/6 = 1.0, J values unchanged
        assert!((result[3] - 1.0).abs() < 0.01);
        assert_eq!(result[4], 3.0);
        assert_eq!(result[5], 8.0);
    }

    #[test]
    fn test_normalize_descriptor_length_mismatch() {
        let descriptor = vec![2, 3, 4, 6]; // Wrong length (should be 6 for 2 orbitals)
        let subshells = vec!["s ".to_string(), "d ".to_string()];

        assert!(normalize_descriptor(&descriptor, &subshells).is_err());
    }

    #[test]
    fn test_batch_normalize_descriptors() {
        let descriptors = vec![
            vec![1, 3, 4, 3, 3, 8],
            vec![2, 3, 4, 6, 3, 8],
        ];
        let subshells = vec!["s ".to_string(), "d ".to_string()];

        let results = batch_normalize_descriptors(&descriptors, &subshells).unwrap();

        assert_eq!(results.len(), 2);

        // First descriptor: 1/2=0.5, 3/6=0.5
        assert!((results[0][0] - 0.5).abs() < 0.01);
        assert!((results[0][3] - 0.5).abs() < 0.01);

        // Second descriptor: 2/2=1.0, 6/6=1.0
        assert!((results[1][0] - 1.0).abs() < 0.01);
        assert!((results[1][3] - 1.0).abs() < 0.01);
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
