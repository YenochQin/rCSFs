//! Integration tests for `csf_block_split::split_csfs_by_j`.
//!
//! Fixtures: `complete.csf` (two blocks: 2J=8 odd, 2J=5 even) and
//! `multi_j.c` (three blocks: 2J=8 odd with two CSFs, 2J=5 even, 2J=0
//! even) plus its synthetic `multi_j.w` orbital file. Error cases build
//! their inputs inside a per-test temporary directory.

use _rcsfs::complete_csf::CompleteCsfFile;
use _rcsfs::csf_block_split::split_csfs_by_j;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const COMPLETE_CSF: &str = include_str!("fixtures/complete.csf");
const MULTI_J_CSF: &str = include_str!("fixtures/multi_j.c");
const MULTI_J_W: &str = include_str!("fixtures/multi_j.w");
const BLOCK_SEPARATOR: &str = " *";

static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rcsfs-jsplit-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The data section of a CSF file: everything after the five header lines.
fn data_section(csf: &str) -> String {
    csf.split_inclusive('\n').skip(5).collect()
}

fn header_section(csf: &str) -> String {
    csf.split_inclusive('\n').take(5).collect()
}

#[test]
fn splits_complete_fixture_into_one_file_per_block() {
    let directory = TestDirectory::new();
    let input = directory.join("complete.c");
    fs::write(&input, COMPLETE_CSF).unwrap();

    let stats = split_csfs_by_j(&input, &directory.0, "complete", false, false).unwrap();

    assert_eq!(stats.block_count, 2);
    assert_eq!(stats.input_csf_count, 2);
    let by_two_j: Vec<_> = stats
        .outputs
        .iter()
        .map(|output| (output.total_two_j, output.parity.as_str(), output.csf_count))
        .collect();
    assert_eq!(
        by_two_j,
        [(8, "odd", 1), (5, "even", 1)],
        "blocks keep file order with their parsed symmetry"
    );

    // Each output re-parses as a single-block canonical CSF file.
    for (index, output) in stats.outputs.iter().enumerate() {
        assert_eq!(output.block_index, index);
        assert_eq!(output.w_file, None);
        let path = Path::new(&output.output_file);
        let parsed = CompleteCsfFile::parse_path(path).unwrap();
        assert_eq!(parsed.blocks.len(), 1);
        assert_eq!(parsed.blocks[0].total_two_j, output.total_two_j);
    }

    // Byte round-trip: the outputs' data sections, rejoined with the
    // separator, reproduce the input's data section exactly, and every
    // output carries the input's five header lines verbatim.
    let first = fs::read_to_string(Path::new(&stats.outputs[0].output_file)).unwrap();
    let second = fs::read_to_string(Path::new(&stats.outputs[1].output_file)).unwrap();
    assert_eq!(header_section(&first), header_section(COMPLETE_CSF));
    assert_eq!(header_section(&second), header_section(COMPLETE_CSF));
    assert_eq!(
        format!(
            "{}{BLOCK_SEPARATOR}\n{}",
            data_section(&first),
            data_section(&second)
        ),
        data_section(COMPLETE_CSF)
    );
}

#[test]
fn splits_multi_j_fixture_with_w_copy() {
    let directory = TestDirectory::new();
    let input = directory.join("multi_j.c");
    fs::write(&input, MULTI_J_CSF).unwrap();
    fs::write(directory.join("multi_j.w"), MULTI_J_W).unwrap();

    let stats = split_csfs_by_j(&input, &directory.0, "multi_j", true, false).unwrap();

    assert_eq!(stats.block_count, 3);
    assert_eq!(stats.input_csf_count, 4);
    let names: Vec<&str> = stats
        .outputs
        .iter()
        .map(|output| {
            Path::new(&output.output_file)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        })
        .collect();
    assert_eq!(names, ["multi_j_8.c", "multi_j_5.c", "multi_j_0.c"]);
    assert_eq!(stats.outputs[0].csf_count, 2);
    assert_eq!(stats.outputs[1].csf_count, 1);
    assert_eq!(stats.outputs[2].csf_count, 1);
    assert_eq!(stats.outputs[2].total_two_j, 0);
    assert_eq!(stats.outputs[2].parity, "even");

    // Every output gains a byte-identical .w copy reported in the stats.
    for output in &stats.outputs {
        let w = output.w_file.as_deref().expect("w copy reported");
        assert_eq!(fs::read(w).unwrap(), MULTI_J_W.as_bytes());
        assert!(w.ends_with(".w"));
    }
    assert_eq!(
        Path::new(&stats.outputs[0].output_file)
            .with_extension("w")
            .display()
            .to_string(),
        stats.outputs[0].w_file.clone().unwrap()
    );

    // Every output — including the hand-written J=0 block — re-parses as a
    // canonical single-block CSF file.
    for output in &stats.outputs {
        let parsed = CompleteCsfFile::parse_path(Path::new(&output.output_file)).unwrap();
        assert_eq!(parsed.blocks.len(), 1);
        assert_eq!(parsed.blocks[0].total_two_j, output.total_two_j);
    }
}

#[test]
fn missing_or_disabled_w_leaves_w_file_null() {
    let directory = TestDirectory::new();
    let input = directory.join("multi_j.c");
    fs::write(&input, MULTI_J_CSF).unwrap();

    let stats = split_csfs_by_j(&input, &directory.0, "multi_j", true, false).unwrap();
    assert!(stats.outputs.iter().all(|output| output.w_file.is_none()));
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 4); // input + 3 outputs

    let disabled = split_csfs_by_j(&input, &directory.0, "again", false, false).unwrap();
    assert!(
        disabled
            .outputs
            .iter()
            .all(|output| output.w_file.is_none())
    );
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 7);
}

#[test]
fn duplicate_two_j_across_blocks_is_an_error_that_publishes_nothing() {
    let directory = TestDirectory::new();
    let input = directory.join("dup.c");
    // Keep the well-formed two-block body but replace the second block's
    // record so its 2J matches the first block's.
    let mut lines: Vec<&str> = COMPLETE_CSF.lines().collect();
    lines.truncate(9); // 5 header lines + first block's record + separator
    lines.push("  4f-( 3)  4f ( 4)  5d ( 1)");
    lines.push("      3/2   4;   4      3/2");
    lines.push("                  7/2      4-");
    fs::write(&input, format!("{}\n", lines.join("\n"))).unwrap();

    let error = split_csfs_by_j(&input, &directory.0, "dup", false, false)
        .expect_err("duplicate 2J must fail");
    let message = format!("{error:#}");
    assert!(
        message.contains("repeats the 2J of block"),
        "error should name both blocks: {message}"
    );
    // Nothing published and no staged temporaries left behind.
    let remaining: Vec<String> = fs::read_dir(&directory.0)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(remaining, vec!["dup.c".to_owned()]);
}

#[test]
fn overwrite_replaces_existing_outputs() {
    let directory = TestDirectory::new();
    let input = directory.join("complete.c");
    fs::write(&input, COMPLETE_CSF).unwrap();
    split_csfs_by_j(&input, &directory.0, "complete", false, false).unwrap();

    let error = split_csfs_by_j(&input, &directory.0, "complete", false, false)
        .expect_err("existing outputs must be refused");
    assert!(format!("{error:#}").contains("already exists"));

    split_csfs_by_j(&input, &directory.0, "complete", false, true).unwrap();
    assert!(directory.join("complete_8.c").is_file());
    assert!(directory.join("complete_5.c").is_file());
}

#[test]
fn output_aliasing_the_input_is_rejected() {
    let directory = TestDirectory::new();
    // The input's own name collides with the output name of its 2J=8 block.
    let input = directory.join("x_8.c");
    fs::write(&input, COMPLETE_CSF).unwrap();
    let error = split_csfs_by_j(&input, &directory.0, "x", false, false)
        .expect_err("outputs must not overwrite the input");
    assert!(format!("{error:#}").contains("must not be the CSF input path"));
    assert_eq!(fs::read_to_string(&input).unwrap(), COMPLETE_CSF);
}

#[cfg(unix)]
#[test]
fn orbital_output_aliases_are_rejected_without_publishing_csfs() {
    for hard_link in [false, true] {
        let directory = TestDirectory::new();
        let input = directory.join("multi_j.c");
        let orbital = directory.join("multi_j.w");
        let output = directory.join("multi_j_8.w");
        fs::write(&input, MULTI_J_CSF).unwrap();
        fs::write(&orbital, MULTI_J_W).unwrap();
        if hard_link {
            fs::hard_link(&orbital, &output).unwrap();
        } else {
            std::os::unix::fs::symlink(&orbital, &output).unwrap();
        }

        let error = split_csfs_by_j(&input, &directory.0, "multi_j", true, true)
            .expect_err("orbital outputs must not alias their source");
        assert!(format!("{error:#}").contains("orbital input"));
        assert_eq!(fs::read_to_string(&input).unwrap(), MULTI_J_CSF);
        assert_eq!(fs::read_to_string(&orbital).unwrap(), MULTI_J_W);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 3);
    }
}

#[test]
fn noncanonical_second_character_star_lines_are_rejected_before_publication() {
    // Put the invalid line in a later block so cleanup must also remove a
    // previously staged output. A second-line marker used to go unparsed.
    for marker in [" *garbage", " * ", "x*garbage"] {
        let directory = TestDirectory::new();
        let input = directory.join("malformed.c");
        let mut lines: Vec<&str> = COMPLETE_CSF.lines().collect();
        lines[10] = marker;
        fs::write(&input, format!("{}\n", lines.join("\n"))).unwrap();

        let error = split_csfs_by_j(&input, &directory.0, "malformed", false, false)
            .expect_err("second-character star requires a canonical separator");
        let message = format!("{error:#}");
        assert!(message.contains("line 11: block separator must be exactly"));
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    }
}

#[cfg(unix)]
#[test]
fn publication_conflicts_preserve_the_io_error_kind_and_published_paths() {
    let directory = TestDirectory::new();
    let input = directory.join("complete.c");
    fs::write(&input, COMPLETE_CSF).unwrap();
    let conflict = directory.join("complete_5.c");
    std::os::unix::fs::symlink(directory.join("missing.c"), &conflict).unwrap();

    let error = split_csfs_by_j(&input, &directory.0, "complete", false, false)
        .expect_err("a dangling output symlink still occupies the destination");
    let cause = error
        .downcast_ref::<std::io::Error>()
        .expect("publication should preserve the underlying I/O error");
    assert_eq!(cause.kind(), std::io::ErrorKind::AlreadyExists);
    let message = format!("{error:#}");
    assert!(message.contains("already published:"));
    assert!(message.contains(&directory.join("complete_8.c").display().to_string()));
    let parsed = CompleteCsfFile::parse_path(&directory.join("complete_8.c")).unwrap();
    assert_eq!(parsed.blocks.len(), 1);
    assert!(
        fs::symlink_metadata(conflict)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 3);
}

#[test]
fn malformed_inputs_are_rejected_with_line_numbers() {
    let directory = TestDirectory::new();

    // Truncated final record.
    let input = directory.join("truncated.c");
    fs::write(&input, format!("{COMPLETE_CSF}  4f-( 2)\n")).unwrap();
    let error = split_csfs_by_j(&input, &directory.0, "truncated", false, false)
        .expect_err("truncated record");
    assert!(format!("{error:#}").contains("final CSF record has 1 of 3 lines"));

    // Empty symmetry block.
    let input = directory.join("empty.c");
    fs::write(&input, COMPLETE_CSF.replacen(" *\n", " *\n *\n", 1)).unwrap();
    let error =
        split_csfs_by_j(&input, &directory.0, "empty", false, false).expect_err("empty block");
    assert!(format!("{error:#}").contains("empty symmetry block"));

    // Mixed symmetry inside one block: two records in block 1 whose total
    // J values differ ("4-" then "3-").
    let lines: Vec<&str> = COMPLETE_CSF.lines().collect();
    let mixed = format!(
        "{}\n  4f-( 3)  4f ( 4)  5d-( 1)\n      3/2   4;   4      3/2\n                  7/2      4-\n  4f-( 3)  4f ( 4)  5d ( 1)\n      3/2   4;   4      3/2\n                  7/2      3-\n",
        lines[..5].join("\n")
    );
    let input = directory.join("mixed.c");
    fs::write(&input, mixed).unwrap();
    let error = split_csfs_by_j(&input, &directory.0, "mixed", false, false)
        .expect_err("mixed block symmetry");
    assert!(format!("{error:#}").contains("mixes total J/parity"));

    // Non-canonical separator.
    let input = directory.join("bare.c");
    fs::write(&input, COMPLETE_CSF.replacen(" *", "*", 1)).unwrap();
    let error =
        split_csfs_by_j(&input, &directory.0, "bare", false, false).expect_err("bare separator");
    assert!(format!("{error:#}").contains("block separator must be exactly"));

    // CRLF input.
    let input = directory.join("crlf.c");
    let crlf = COMPLETE_CSF.replace("\n", "\r\n");
    fs::write(&input, crlf).unwrap();
    let error =
        split_csfs_by_j(&input, &directory.0, "crlf", false, false).expect_err("CRLF rejected");
    assert!(format!("{error:#}").contains("CRLF is not canonical"));

    // Header-only file.
    let input = directory.join("header_only.c");
    fs::write(&input, header_section(COMPLETE_CSF)).unwrap();
    let error =
        split_csfs_by_j(&input, &directory.0, "header_only", false, false).expect_err("no records");
    assert!(format!("{error:#}").contains("contains no CSF records"));
}

#[test]
fn bad_prefix_and_missing_output_directory_are_rejected() {
    let directory = TestDirectory::new();
    let input = directory.join("complete.c");
    fs::write(&input, COMPLETE_CSF).unwrap();

    for prefix in ["", ".", "..", "a/b", "a\\b"] {
        let error = split_csfs_by_j(&input, &directory.0, prefix, false, false)
            .expect_err("prefix must be validated");
        assert!(format!("{error:#}").contains("prefix must be a nonempty filename stem"));
    }

    let missing_dir = directory.join("nope");
    let error = split_csfs_by_j(&input, &missing_dir, "complete", false, false)
        .expect_err("missing output directory");
    assert!(format!("{error:#}").contains("output directory does not exist"));
}
