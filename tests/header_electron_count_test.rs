use _rcsfs::csfs_conversion::{
    HeaderData, convert_csfs_to_parquet, convert_csfs_to_parquet_parallel,
};
use std::fs;

#[test]
fn conversion_counts_core_and_first_csf_before_truncation() {
    for core in ["", "  1s   2s   2p-  2p"] {
        for parallel in [false, true] {
            for max_line_len in [6, 256] {
                let directory = tempfile::tempdir().unwrap();
                let source = directory.path().join("source.c");
                // Leading separators and batches shorter than a record must not
                // hide the first occupation line. The two rows have the same N.
                fs::write(
                    &source,
                    format!(
                        "Core subshells:\n{core}\nPeel subshells:\n  3s   3p-  3p\nCSF(s):\n *\n\
                         3s ( 2)  3p-( 2)  3p ( 3)\nstate\ncoupling\n *\n\
                         3s ( 1)  3p-( 2)  3p ( 4)\nstate\ncoupling\n"
                    ),
                )
                .unwrap();
                let output = directory.path().join("out.parquet");
                if parallel {
                    convert_csfs_to_parquet_parallel(&source, &output, max_line_len, 1, Some(2))
                        .unwrap();
                } else {
                    convert_csfs_to_parquet(&source, &output, max_line_len, 1).unwrap();
                }
                let text = fs::read_to_string(directory.path().join("source_header.toml")).unwrap();
                let header: HeaderData = toml::from_str(&text).unwrap();
                assert_eq!(
                    header.header_info.electron_count,
                    Some(if core.is_empty() { 7 } else { 17 })
                );
                assert_eq!(header.conversion_stats.csf_count, 2);
                assert_eq!(header.block_info.block_lengths, [0, 1, 1]);
            }
        }
    }
}

#[test]
fn legacy_header_deserializes_without_inventing_an_electron_count() {
    let text = r#"
[header_info]
header_lines = ["Core subshells:", "  1s", "Peel subshells:", "  2s", "CSF(s):"]
[block_info]
block_lengths = [1]
block_count = 1
[conversion_stats]
csf_count = 1
total_lines = 3
truncated_count = 0
"#;
    let header: HeaderData = toml::from_str(text).unwrap();
    assert_eq!(header.header_info.electron_count, None);
    assert!(!toml::to_string(&header).unwrap().contains("electron_count"));
}
