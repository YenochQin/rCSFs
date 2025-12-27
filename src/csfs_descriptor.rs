//! CSF Descriptor Generation Module
//!
//! This module converts Configuration State Function (CSF) data into descriptor arrays
//! for machine learning applications. Each CSF is parsed into a fixed-length array
//! containing electron counts and angular momentum coupling values.

use std::collections::HashMap;

/// Represents the result of a CSF parsing operation
#[derive(Debug, Clone)]
pub struct CSFDescriptor {
    /// The descriptor array: [e_count, middle_J, coupling_J, ...] for each orbital
    pub values: Vec<f32>,
    /// Metadata about the parsing operation
    pub metadata: DescriptorMetadata,
}

#[derive(Debug, Clone)]
pub struct DescriptorMetadata {
    pub orbital_count: usize,
    pub occupied_orbitals: Vec<usize>,
    pub final_double_j: i32,
}

/// Convert a J-value string to its doubled integer representation (2J)
///
/// # Arguments
/// * `j_str` - J value as string, e.g., "3/2", "2", "5/2"
///
/// # Returns
/// * `Ok(i32)` - The doubled J value (2J)
/// * `Err(String)` - Error message if parsing fails
///
/// # Examples
/// ```text
/// // Fractional J values:
/// j_to_double_j("3/2") => Ok(3)
/// j_to_double_j("5/2") => Ok(5)
///
/// // Integer J values:
/// j_to_double_j("2")  => Ok(4)
/// j_to_double_j("4")  => Ok(8)
/// ```
pub fn j_to_double_j(j_str: &str) -> Result<i32, String> {
    let trimmed = j_str.trim();

    if trimmed.is_empty() {
        return Ok(0); // Empty string represents 0
    }

    // Handle fractional J values (e.g., "3/2" -> 3)
    if let Some(slash_pos) = trimmed.find('/') {
        let numerator: i32 = trimmed[..slash_pos]
            .parse()
            .map_err(|_| format!("Invalid J value numerator: {}", trimmed))?;
        return Ok(numerator);
    }

    // Handle integer J values (e.g., "2" -> 4, "4-" -> 8)
    // Remove trailing parity indicator if present
    let cleaned = trimmed.trim_end_matches('-').trim_end_matches('+');
    cleaned
        .parse::<i32>()
        .map(|j| j * 2)
        .map_err(|_| format!("Invalid J value: {}", trimmed))
}

/// Chunk a string into fixed-size pieces
///
/// # Arguments
/// * `s` - The string to chunk
/// * `chunk_size` - Size of each chunk
///
/// # Returns
/// Vector of string chunks
fn chunk_string(s: &str, chunk_size: usize) -> Vec<&str> {
    s.as_bytes()
        .chunks(chunk_size)
        .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
        .collect()
}

/// CSF Descriptor Generator
///
/// This struct maintains the state needed to convert CSF data into descriptor arrays.
pub struct CSFDescriptorGenerator {
    /// List of peel subshell names (e.g., ["5s", "4d-", "4d", ...])
    peel_subshells: Vec<String>,
    /// Map from subshell name to index for O(1) lookup
    orbital_index_map: HashMap<String, usize>,
    /// Number of orbitals (cached for performance)
    orbital_count: usize,
}

impl CSFDescriptorGenerator {
    /// Create a new CSF descriptor generator
    ///
    /// # Arguments
    /// * `peel_subshells` - List of subshell names (e.g., ["5s", "4d-", "4d"])
    ///
    /// # Returns
    /// A new generator instance
    pub fn new(peel_subshells: Vec<String>) -> Self {
        let orbital_count = peel_subshells.len();
        let orbital_index_map: HashMap<_, _> = peel_subshells
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        Self {
            peel_subshells,
            orbital_index_map,
            orbital_count,
        }
    }

    /// Get the number of orbitals
    pub fn orbital_count(&self) -> usize {
        self.orbital_count
    }

    /// Get the peel subshells list
    pub fn peel_subshells(&self) -> &[String] {
        &self.peel_subshells
    }

    /// Parse a single CSF into a descriptor array
    ///
    /// # Arguments
    /// * `line1` - First line: subshell configurations and electron counts
    /// * `line2` - Second line: intermediate J coupling values
    /// * `line3` - Third line: final coupling and total J value
    ///
    /// # Returns
    /// A CSFDescriptor containing the values and metadata
    ///
    /// # CSF Format Example
    /// ```text
    /// line1: "  5s ( 2)  4d-( 4)  4d ( 6)"
    /// line2: "                   3/2      "
    /// line3: "                        4-  "
    /// ```
    pub fn parse_csf(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
    ) -> Result<CSFDescriptor, String> {
        // Initialize descriptor array with zeros
        let mut descriptor = vec![0.0f32; 3 * self.orbital_count];
        let mut occupied_orbitals = Vec::new();

        // Step 1: Preprocess the three lines
        let subshells_line = line1.trim_end();
        let line_length = subshells_line.len();

        // Pad middle line to match line1 length
        let middle_line = format!("{:<width$}", line2.trim_end(), width = line_length);

        // Extract coupling line (remove first 4 and last 5 characters)
        let coupling_line_raw = line3.trim_end();
        let coupling_start = if coupling_line_raw.len() > 5 { 4 } else { 0 };
        let coupling_end = if coupling_line_raw.len() > 5 { coupling_line_raw.len() - 5 } else { coupling_line_raw.len() };
        let coupling_line = format!("{:<width$}",
            &coupling_line_raw[coupling_start..coupling_end],
            width = line_length
        );

        // Step 2: Extract final J value from the end of line3
        let final_j_start = if coupling_line_raw.len() > 5 { coupling_line_raw.len() - 5 } else { 0 };
        let final_j_end = if coupling_line_raw.len() > 1 { coupling_line_raw.len() - 1 } else { coupling_line_raw.len() };
        let final_j_str = &coupling_line_raw[final_j_start..final_j_end];
        let final_double_j = j_to_double_j(final_j_str)?;

        // Step 3: Chunk lines into 9-character blocks
        let subshell_list = chunk_string(subshells_line, 9);
        let middle_list = chunk_string(&middle_line, 9);
        let coupling_list = chunk_string(&coupling_line, 9);

        // Step 4: Process each subshell block
        for (i, ((subshell_charges, middle_item), coupling_item)) in
            subshell_list.iter().zip(middle_list.iter()).zip(coupling_list.iter()).enumerate()
        {
            // Extract subshell name (first 5 characters, trimmed)
            let subshell = subshell_charges.get(0..5)
                .map(|s: &str| s.trim())
                .unwrap_or("")
                .to_string();

            // Extract electron number (characters 6-8, i.e., indices 6 and 7)
            let subshell_electron_num: f32 = if subshell_charges.len() >= 8 {
                subshell_charges[6..8].trim().parse()
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            // Check if this is the last subshell
            let is_last = i == subshell_list.len() - 1;

            // Process middle J coupling value (line 2)
            let mut temp_middle_item: i32 = 0;
            if !middle_item.trim().is_empty() {
                // If semicolon separated, take the last value
                let middle_value = if let Some(semi_pos) = middle_item.find(';') {
                    &middle_item[semi_pos + 1..]
                } else {
                    middle_item
                };
                temp_middle_item = j_to_double_j(middle_value).unwrap_or(0);
            }

            // Process coupling J value (line 3)
            let mut temp_coupling_item: i32 = 0;
            if !coupling_item.trim().is_empty() {
                temp_coupling_item = j_to_double_j(coupling_item).unwrap_or(0);
            } else if !middle_item.trim().is_empty() {
                // If line 3 is empty but line 2 has a value, use line 2's value
                temp_coupling_item = temp_middle_item;
            }

            // Special handling: last subshell uses final J value
            if is_last {
                temp_coupling_item = final_double_j;
            }

            // Step 5: Find orbital index in the peel subshells list
            if let Some(&orbs_idx) = self.orbital_index_map.get(&subshell) {
                let descriptor_idx = orbs_idx * 3;

                // Ensure we don't go out of bounds
                if descriptor_idx + 3 <= descriptor.len() {
                    descriptor[descriptor_idx] = subshell_electron_num;
                    descriptor[descriptor_idx + 1] = temp_middle_item as f32;
                    descriptor[descriptor_idx + 2] = temp_coupling_item as f32;

                    occupied_orbitals.push(orbs_idx);
                }
            } else {
                // Warning: subshell not in list (skip as Python does)
                eprintln!("Warning: {} not found in orbs list", subshell);
            }
        }

        // Step 6: Fill unoccupied orbitals with final J value (position 2)
        let all_orbitals: std::collections::HashSet<_> =
            (0..self.orbital_count).collect();
        let occupied: std::collections::HashSet<_> =
            occupied_orbitals.iter().cloned().collect();
        let remaining: Vec<_> =
            all_orbitals.difference(&occupied).cloned().collect();

        for idx in remaining {
            let descriptor_idx = idx * 3 + 2;
            if descriptor_idx < descriptor.len() {
                descriptor[descriptor_idx] = final_double_j as f32;
            }
        }

        Ok(CSFDescriptor {
            values: descriptor,
            metadata: DescriptorMetadata {
                orbital_count: self.orbital_count,
                occupied_orbitals,
                final_double_j: final_double_j,
            },
        })
    }

    /// Parse a CSF and return only the descriptor values (as Vec<f32>)
    ///
    /// This is a convenience method that returns just the array values
    /// without the metadata.
    pub fn parse_cfl_to_vec(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
    ) -> Result<Vec<f32>, String> {
        self.parse_csf(line1, line2, line3)
            .map(|desc| desc.values)
    }
}

//////////////////////////////////////////////////////////////////////////////
/// Python Bindings (PyO3)
//////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-exposed function to convert J string to 2J value
#[cfg(feature = "python")]
#[pyfunction]
fn py_j_to_double_j(j_str: &str) -> PyResult<i32> {
    j_to_double_j(j_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Python-exposed CSF Descriptor Generator class
#[cfg(feature = "python")]
#[pyclass(name = "CSFDescriptorGenerator")]
pub struct PyCSFDescriptorGenerator {
    inner: CSFDescriptorGenerator,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCSFDescriptorGenerator {
    /// Create a new descriptor generator
    #[new]
    fn new(peel_subshells: Vec<String>) -> Self {
        Self {
            inner: CSFDescriptorGenerator::new(peel_subshells),
        }
    }

    /// Get the number of orbitals
    fn orbital_count(&self) -> usize {
        self.inner.orbital_count()
    }

    /// Get the peel subshells list
    fn peel_subshells(&self) -> Vec<String> {
        self.inner.peel_subshells().to_vec()
    }

    /// Parse a single CSF into a descriptor array
    ///
    /// Args:
    ///     line1: First line of CSF (subshell configurations)
    ///     line2: Second line of CSF (intermediate J coupling)
    ///     line3: Third line of CSF (final coupling and total J)
    ///
    /// Returns:
    ///     List of float32 descriptor values
    fn parse_csf(&self, line1: &str, line2: &str, line3: &str) -> PyResult<Vec<f32>> {
        self.inner
            .parse_cfl_to_vec(line1, line2, line3)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Parse CSF from a list of 3 strings (Python list format)
    fn parse_csf_from_list(&self, csf_lines: Vec<String>) -> PyResult<Vec<f32>> {
        if csf_lines.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "csf_lines must contain exactly 3 elements",
            ));
        }
        self.inner
            .parse_cfl_to_vec(&csf_lines[0], &csf_lines[1], &csf_lines[2])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Batch parse multiple CSFs
    ///
    /// Args:
    ///     csf_list: List of CSF data, each being a list of 3 strings
    ///
    /// Returns:
    ///     List of descriptor arrays
    fn batch_parse_csfs(&self, csf_list: Vec<Vec<String>>) -> PyResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(csf_list.len());

        for (idx, csf_lines) in csf_list.into_iter().enumerate() {
            match self.inner.parse_cfl_to_vec(
                &csf_lines.get(0).map(|s| s.as_str()).unwrap_or(""),
                &csf_lines.get(1).map(|s| s.as_str()).unwrap_or(""),
                &csf_lines.get(2).map(|s| s.as_str()).unwrap_or(""),
            ) {
                Ok(descriptor) => results.push(descriptor),
                Err(e) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Error parsing CSF at index {}: {}",
                        idx, e
                    )))
                }
            }
        }

        Ok(results)
    }

    /// Get the configuration as a dictionary
    fn get_config(&self, py: Python) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("orbital_count", self.inner.orbital_count())?;
        dict.set_item("peel_subshells", self.inner.peel_subshells())?;
        Ok(dict.into())
    }
}

/// Register the Python module functions and classes
#[cfg(feature = "python")]
pub fn register_descriptor_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_j_to_double_j, module)?)?;
    module.add_class::<PyCSFDescriptorGenerator>()?;
    Ok(())
}

//////////////////////////////////////////////////////////////////////////////
/// Rust Tests
//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j_to_double_j_fractional() {
        assert_eq!(j_to_double_j("3/2"), Ok(3));
        assert_eq!(j_to_double_j("5/2"), Ok(5));
        assert_eq!(j_to_double_j("1/2"), Ok(1));
    }

    #[test]
    fn test_j_to_double_j_integer() {
        assert_eq!(j_to_double_j("2"), Ok(4));
        assert_eq!(j_to_double_j("3"), Ok(6));
        assert_eq!(j_to_double_j("4"), Ok(8));
    }

    #[test]
    fn test_j_to_double_j_with_parity() {
        assert_eq!(j_to_double_j("4-"), Ok(8));
        assert_eq!(j_to_double_j("3+"), Ok(6));
    }

    #[test]
    fn test_j_to_double_j_empty() {
        assert_eq!(j_to_double_j(""), Ok(0));
        assert_eq!(j_to_double_j("   "), Ok(0));
    }

    #[test]
    fn test_j_to_double_j_invalid() {
        assert!(j_to_double_j("invalid").is_err());
        assert!(j_to_double_j("abc/def").is_err());
    }

    #[test]
    fn test_chunk_string() {
        let result = chunk_string("abcdefghi", 3);
        assert_eq!(result, vec!["abc", "def", "ghi"]);
    }

    #[test]
    fn test_descriptor_generator_creation() {
        let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
        let generator = CSFDescriptorGenerator::new(subshells.clone());

        assert_eq!(generator.orbital_count(), 3);
        assert_eq!(generator.peel_subshells(), &subshells);
    }
}
