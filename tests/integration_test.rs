//! Integration tests for rCSFs library
//!
//! These tests cover:
//! - Parquet file I/O operations
//! - Parallel processing edge cases
//! - Error recovery scenarios
//! - Large file handling

use std::fs;
use std::path::{Path, PathBuf};

//////////////////////////////////////////////////////////////////////////////
// Test Utilities
//////////////////////////////////////////////////////////////////////////////

/// Create a temporary directory for test outputs
fn temp_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("target/test_outputs");
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Clean up test output files
fn cleanup_test_file(path: &Path) {
    if path.exists() {
        fs::remove_file(path).ok();
    }
    // Clean up header file if it exists
    let header_path = path.with_extension("toml");
    if header_path.exists() {
        fs::remove_file(&header_path).ok();
    }
}

//////////////////////////////////////////////////////////////////////////////
// Test Fixtures
//////////////////////////////////////////////////////////////////////////////

/// Create a minimal valid CSF file for testing
fn create_minimal_csf(path: &Path) {
    let content = "  Header line 1\n\
                      Header line 2\n\
                      Header line 3\n\
                      Header line 4\n\
                      Header line 5\n\
                      5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)\n\
                                          3/2               2\n\
                                                                  4-\n\
                      5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)\n\
                                          3/2               2\n\
                                                                  4-\n";
    fs::write(path, content).expect("Failed to create test CSF file");
}

/// Create a CSF file with long lines (truncation test)
fn create_long_line_csf(path: &Path, line_length: usize) {
    let mut content = String::from("  Header line 1\n\
                                      Header line 2\n\
                                      Header line 3\n\
                                      Header line 4\n\
                                     Header line 5\n");

    let long_string = "x".repeat(line_length);
    for _ in 0..3 {
        content.push_str(&format!("  {}  \n", long_string));
        content.push_str("                  3/2               2\n");
        content.push_str("                                          4-\n");
    }

    fs::write(path, content).expect("Failed to create long-line CSF file");
}

/// Create a large CSF file with many CSFs
fn create_large_csf(path: &Path, csf_count: usize) {
    let mut content = String::from("  Header line 1\n\
                                      Header line 2\n\
                                      Header line 3\n\
                                      Header line 4\n\
                                     Header line 5\n");

    for _ in 0..csf_count {
        content.push_str("  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)\n");
        content.push_str("                  3/2               2\n");
        content.push_str("                                          4-\n");
    }

    fs::write(path, content).expect("Failed to create large CSF file");
}

//////////////////////////////////////////////////////////////////////////////
// Parquet I/O Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_parquet_io_basic() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("test_basic.csf");
    let output_path = temp_dir().join("test_basic.parquet");

    create_minimal_csf(&input_path);

    let result = convert_csfs_to_parquet(&input_path, &output_path, 256, 1000);

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Conversion should succeed");
    let stats = result.unwrap();
    assert_eq!(stats.csf_count, 2, "Should process 2 CSFs");
    assert_eq!(stats.total_lines, 6, "Should have 6 total lines (2 CSFs * 3 lines)");
}

#[test]
fn test_parquet_io_creates_header_file() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("test_header.csf");
    let output_path = temp_dir().join("test_header.parquet");
    let header_path = temp_dir().join("test_header_header.toml");

    create_minimal_csf(&input_path);

    let result = convert_csfs_to_parquet(&input_path, &output_path, 256, 1000);

    assert!(result.is_ok(), "Conversion should succeed");
    assert!(header_path.exists(), "Header file should be created");

    // Verify header file contains expected data
    let header_content = fs::read_to_string(&header_path).unwrap();
    assert!(header_content.contains("header_lines"), "Header file should contain header_lines");
    assert!(header_content.contains("conversion_stats"), "Header file should contain conversion_stats");

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);
    cleanup_test_file(&header_path);
}

#[test]
fn test_parquet_io_invalid_input_path() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("nonexistent.csf");
    let output_path = temp_dir().join("output.parquet");

    let result = convert_csfs_to_parquet(&input_path, &output_path, 256, 1000);

    assert!(result.is_err(), "Conversion should fail for non-existent input file");
}

#[test]
fn test_parquet_io_invalid_output_directory() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("test.csf");
    let output_path = PathBuf::from("/nonexistent/directory/output.parquet");

    create_minimal_csf(&input_path);

    let result = convert_csfs_to_parquet(&input_path, &output_path, 256, 1000);

    cleanup_test_file(&input_path);

    assert!(result.is_err(), "Conversion should fail for invalid output directory");
}

//////////////////////////////////////////////////////////////////////////////
// Line Truncation Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_line_truncation() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("test_truncate.csf");
    let output_path = temp_dir().join("test_truncate.parquet");

    create_long_line_csf(&input_path, 500); // Lines longer than max_line_len=256

    let result = convert_csfs_to_parquet(&input_path, &output_path, 256, 1000);

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Conversion should succeed");
    let stats = result.unwrap();
    assert!(stats.truncated_count > 0, "Should truncate some lines");
    assert!(stats.truncated_count <= 9, "Should truncate at most 9 lines (3 per CSF)");
}

//////////////////////////////////////////////////////////////////////////////
// Parallel Processing Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_parallel_processing_basic() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet_parallel;

    let input_path = temp_dir().join("test_parallel_basic.csf");
    let output_path = temp_dir().join("test_parallel_basic.parquet");

    create_minimal_csf(&input_path);

    let result = convert_csfs_to_parquet_parallel(&input_path, &output_path, 256, 1000, Some(2));

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Parallel conversion should succeed");
    let stats = result.unwrap();
    assert_eq!(stats.csf_count, 2, "Should process 2 CSFs");
}

#[test]
fn test_parallel_processing_single_worker() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet_parallel;

    let input_path = temp_dir().join("test_parallel_single.csf");
    let output_path = temp_dir().join("test_parallel_single.parquet");

    create_minimal_csf(&input_path);

    let result = convert_csfs_to_parquet_parallel(&input_path, &output_path, 256, 1000, Some(1));

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Single worker conversion should succeed");
}

#[test]
fn test_parallel_processing_many_workers() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet_parallel;

    let input_path = temp_dir().join("test_parallel_many.csf");
    let output_path = temp_dir().join("test_parallel_many.parquet");

    create_minimal_csf(&input_path);

    let result = convert_csfs_to_parquet_parallel(&input_path, &output_path, 256, 1000, Some(8));

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Many workers conversion should succeed");
}

//////////////////////////////////////////////////////////////////////////////
// Edge Cases Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_empty_csf() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("test_empty.csf");
    let output_path = temp_dir().join("test_empty.parquet");

    // Create a CSF file with only headers (no data)
    let content = "  Header line 1\n\
                      Header line 2\n\
                      Header line 3\n\
                      Header line 4\n\
                     Header line 5\n";
    fs::write(&input_path, content).expect("Failed to create empty CSF file");

    let result = convert_csfs_to_parquet(&input_path, &output_path, 256, 1000);

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Empty CSF should handle gracefully");
    let stats = result.unwrap();
    assert_eq!(stats.csf_count, 0, "Should have 0 CSFs");
}

#[test]
fn test_incomplete_csf() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("test_incomplete.csf");
    let output_path = temp_dir().join("test_incomplete.parquet");

    // Create a CSF file with incomplete final CSF (only 2 lines instead of 3)
    let content = "  Header line 1\n\
                      Header line 2\n\
                      Header line 3\n\
                      Header line 4\n\
                     Header line 5\n\
                      5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)\n\
                                          3/2               2\n";
    fs::write(&input_path, content).expect("Failed to create incomplete CSF file");

    let result = convert_csfs_to_parquet(&input_path, &output_path, 256, 1000);

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Incomplete CSF should be skipped gracefully");
    let stats = result.unwrap();
    assert_eq!(stats.csf_count, 0, "Should skip incomplete CSF");
}

#[test]
fn test_very_small_max_line_len() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("test_small_max.csf");
    let output_path = temp_dir().join("test_small_max.parquet");

    create_minimal_csf(&input_path);

    let result = convert_csfs_to_parquet(&input_path, &output_path, 10, 1000);

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Very small max_line_len should work");
    let stats = result.unwrap();
    assert!(stats.truncated_count > 0, "Should truncate lines");
}

//////////////////////////////////////////////////////////////////////////////
// Large File Tests
//////////////////////////////////////////////////////////////////////////////

#[test]
fn test_large_file_sequential() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

    let input_path = temp_dir().join("test_large_seq.csf");
    let output_path = temp_dir().join("test_large_seq.parquet");

    // Create a file with 1000 CSFs
    create_large_csf(&input_path, 1000);

    let result = convert_csfs_to_parquet(&input_path, &output_path, 256, 10000);

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Large file sequential conversion should succeed");
    let stats = result.unwrap();
    assert_eq!(stats.csf_count, 1000, "Should process all 1000 CSFs");
}

#[test]
fn test_large_file_parallel() {
    use _rcsfs::csfs_conversion::convert_csfs_to_parquet_parallel;

    let input_path = temp_dir().join("test_large_par.csf");
    let output_path = temp_dir().join("test_large_par.parquet");

    // Create a file with 1000 CSFs
    create_large_csf(&input_path, 1000);

    let result = convert_csfs_to_parquet_parallel(&input_path, &output_path, 256, 5000, Some(4));

    cleanup_test_file(&input_path);
    cleanup_test_file(&output_path);

    assert!(result.is_ok(), "Large file parallel conversion should succeed");
    let stats = result.unwrap();
    assert_eq!(stats.csf_count, 1000, "Should process all 1000 CSFs");
}

#[test]
fn test_large_file_integrity() {
    use _rcsfs::csfs_conversion::{convert_csfs_to_parquet, convert_csfs_to_parquet_parallel};

    let input_path1 = temp_dir().join("test_integrity_seq.csf");
    let input_path2 = temp_dir().join("test_integrity_par.csf");
    let output_path1 = temp_dir().join("test_integrity_seq.parquet");
    let output_path2 = temp_dir().join("test_integrity_par.parquet");

    // Create identical input files
    create_large_csf(&input_path1, 100);
    create_large_csf(&input_path2, 100);

    let result_seq = convert_csfs_to_parquet(&input_path1, &output_path1, 256, 10000);
    let result_par = convert_csfs_to_parquet_parallel(&input_path2, &output_path2, 256, 5000, Some(4));

    cleanup_test_file(&input_path1);
    cleanup_test_file(&input_path2);
    cleanup_test_file(&output_path1);
    cleanup_test_file(&output_path2);

    assert!(result_seq.is_ok(), "Sequential conversion should succeed");
    assert!(result_par.is_ok(), "Parallel conversion should succeed");

    let stats_seq = result_seq.unwrap();
    let stats_par = result_par.unwrap();

    // Both should produce identical results
    assert_eq!(stats_seq.csf_count, stats_par.csf_count, "CSF count should match");
    assert_eq!(stats_seq.total_lines, stats_par.total_lines, "Total lines should match");
    assert_eq!(stats_seq.truncated_count, stats_par.truncated_count, "Truncated count should match");
}
