//! Integration tests for `csf_partition::partition_csfs`.
//!
//! Each test synthesizes zero-order and full CSF text files with known
//! structure, converts both to Parquet via `convert_csfs_to_parquet`, runs
//! `partition_csfs`, then parses the output CSF text and asserts ordering,
//! separators, header provenance, and statistics.

use std::fs;
use std::path::{Path, PathBuf};

use _rcsfs::csf_partition::partition_csfs;
use _rcsfs::csfs_conversion::convert_csfs_to_parquet;

const HEADER_LINES: [&str; 5] = ["H1", "H2", "H3", "H4", "H5"];
const LINE2: &str = "L2";
const LINE3: &str = "L3";
const BLOCK_SEP: &str = "*";

fn temp_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("target/test_outputs");
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup(paths: &[PathBuf]) {
    for p in paths {
        if p.is_dir() {
            let _ = fs::remove_dir_all(p);
        } else if p.exists() {
            let _ = fs::remove_file(p);
        }
    }
}

/// Build a CSF text file from a nested block/tag structure.
/// Each tag becomes the line1 of a CSF; line2/line3 are constant so that
/// tags uniquely identify CSFs (matching is on the full triple).
fn build_csf_text(blocks: &[Vec<&str>]) -> String {
    let mut s = String::new();
    for h in HEADER_LINES {
        s.push_str(h);
        s.push('\n');
    }
    for (bi, block) in blocks.iter().enumerate() {
        for tag in block {
            s.push_str(tag);
            s.push('\n');
            s.push_str(LINE2);
            s.push('\n');
            s.push_str(LINE3);
            s.push('\n');
        }
        if bi + 1 < blocks.len() {
            s.push_str(" ");
            s.push_str(BLOCK_SEP);
            s.push('\n');
        }
    }
    s
}

/// Parsed CSF output: 5 header lines + per-block lists of (line1, line2, line3).
type Triple = [String; 3];
struct ParsedCsf {
    header: Vec<String>,
    blocks: Vec<Vec<Triple>>,
}

fn parse_csf_output(path: &Path) -> ParsedCsf {
    let content = fs::read_to_string(path).unwrap();
    let mut lines = content.lines();
    let header: Vec<String> = lines.by_ref().take(5).map(String::from).collect();
    let mut blocks: Vec<Vec<Triple>> = Vec::new();
    let mut current: Vec<Triple> = Vec::new();
    let mut buf: Vec<String> = Vec::with_capacity(3);
    for line in lines {
        if line.trim() == BLOCK_SEP {
            if !current.is_empty() {
                blocks.push(std::mem::take(&mut current));
            }
            continue;
        }
        buf.push(line.to_string());
        if buf.len() == 3 {
            let triple = [buf[0].clone(), buf[1].clone(), buf[2].clone()];
            current.push(triple);
            buf.clear();
        }
    }
    if !current.is_empty() {
        blocks.push(current);
    }
    ParsedCsf { header, blocks }
}

/// Convert a CSF text file to Parquet and return (parquet_path, header_path).
fn convert(name: &str, csf_text: &str) -> (PathBuf, PathBuf) {
    let dir = temp_dir().join(name);
    fs::create_dir_all(&dir).unwrap();
    let csf = dir.join("input.csf");
    let parquet = dir.join("input.parquet");
    let header = dir.join("input_header.toml");
    fs::write(&csf, csf_text).unwrap();
    convert_csfs_to_parquet(&csf, &parquet, 256, 1000).unwrap();
    assert!(header.exists(), "header TOML should exist at {header:?}");
    (parquet, header)
}

fn tags_of(block: &[Triple]) -> Vec<String> {
    block.iter().map(|t| t[0].clone()).collect()
}

#[test]
fn test_partition_locks_zero_to_head_and_appends_complement() {
    // full: 2 blocks of 4 CSFs each (A,B,C,D | E,F,G,H)
    // zero: 2 blocks of 2 CSFs each (B,D | F,H) -- NOT first in full
    // expected: [B,D] + complement[A,C] | [F,H] + complement[E,G]
    let full_text = build_csf_text(&[vec!["A", "B", "C", "D"], vec!["E", "F", "G", "H"]]);
    let zero_text = build_csf_text(&[vec!["B", "D"], vec!["F", "H"]]);

    let (zero_pq, zero_hdr) = convert("zf_lock_zero", &zero_text);
    let (full_pq, full_hdr) = convert("zf_lock_full", &full_text);
    let output = temp_dir().join("zf_lock_out.csf");

    let stats = partition_csfs(&zero_pq, &zero_hdr, &full_pq, &full_hdr, &output).unwrap();
    let parsed = parse_csf_output(&output);

    assert_eq!(stats.block_count, 2);
    assert_eq!(stats.zero_csf_count, 4);
    assert_eq!(stats.full_csf_count, 8);
    assert_eq!(stats.first_order_count, 4); // A,C + E,G
    assert_eq!(stats.output_csf_count, 8);

    assert_eq!(parsed.blocks.len(), 2);
    // zero locked to head, complement after, preserving full's relative order
    assert_eq!(tags_of(&parsed.blocks[0]), vec!["B", "D", "A", "C"]);
    assert_eq!(tags_of(&parsed.blocks[1]), vec!["F", "H", "E", "G"]);

    cleanup(&[zero_pq, zero_hdr, full_pq, full_hdr, output]);
    cleanup(&[
        temp_dir().join("zf_lock_zero"),
        temp_dir().join("zf_lock_full"),
    ]);
}

#[test]
fn test_partition_writes_block_separator_between_blocks_only() {
    let full_text = build_csf_text(&[vec!["A", "B"], vec!["C", "D"], vec!["E", "F"]]);
    let zero_text = build_csf_text(&[vec!["A"], vec!["C"], vec!["E"]]);

    let (zero_pq, zero_hdr) = convert("zf_sep_zero", &zero_text);
    let (full_pq, full_hdr) = convert("zf_sep_full", &full_text);
    let output = temp_dir().join("zf_sep_out.csf");

    partition_csfs(&zero_pq, &zero_hdr, &full_pq, &full_hdr, &output).unwrap();
    let content = fs::read_to_string(&output).unwrap();
    let separator_count = content.lines().filter(|l| l.trim() == BLOCK_SEP).count();

    // 3 blocks => exactly 2 separators, none trailing
    assert_eq!(separator_count, 2);
    assert!(
        !content.ends_with("*\n") && !content.ends_with("*"),
        "output must not end with a block separator"
    );

    let parsed = parse_csf_output(&output);
    assert_eq!(parsed.blocks.len(), 3);

    cleanup(&[
        zero_pq,
        zero_hdr,
        full_pq,
        full_hdr,
        output,
        temp_dir().join("zf_sep_zero"),
        temp_dir().join("zf_sep_full"),
    ]);
}

#[test]
fn test_partition_preserves_5_line_header_from_full_file() {
    let full_text = build_csf_text(&[vec!["A", "B"]]);
    let zero_text = build_csf_text(&[vec!["A"]]);

    let (zero_pq, zero_hdr) = convert("zf_hdr_zero", &zero_text);
    let (full_pq, full_hdr) = convert("zf_hdr_full", &full_text);
    let output = temp_dir().join("zf_hdr_out.csf");

    partition_csfs(&zero_pq, &zero_hdr, &full_pq, &full_hdr, &output).unwrap();
    let parsed = parse_csf_output(&output);

    assert_eq!(parsed.header, HEADER_LINES);

    cleanup(&[
        zero_pq,
        zero_hdr,
        full_pq,
        full_hdr,
        output,
        temp_dir().join("zf_hdr_zero"),
        temp_dir().join("zf_hdr_full"),
    ]);
}

#[test]
fn test_partition_rejects_block_count_mismatch() {
    // zero has 2 blocks, full has 3 => must error
    let full_text = build_csf_text(&[vec!["A"], vec!["B"], vec!["C"]]);
    let zero_text = build_csf_text(&[vec!["A"], vec!["B"]]);

    let (zero_pq, zero_hdr) = convert("zf_mis_zero", &zero_text);
    let (full_pq, full_hdr) = convert("zf_mis_full", &full_text);
    let output = temp_dir().join("zf_mis_out.csf");

    let result = partition_csfs(&zero_pq, &zero_hdr, &full_pq, &full_hdr, &output);
    assert!(result.is_err(), "block count mismatch must error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("block count mismatch"),
        "error should mention block count mismatch, got: {err_msg}"
    );
    assert!(!output.exists(), "no output should be created on mismatch");

    cleanup(&[
        zero_pq,
        zero_hdr,
        full_pq,
        full_hdr,
        output,
        temp_dir().join("zf_mis_zero"),
        temp_dir().join("zf_mis_full"),
    ]);
}

#[test]
fn test_partition_zero_equals_full_yields_zero_complement() {
    let blocks = [vec!["A", "B", "C"], vec!["D", "E"]];
    let text = build_csf_text(&blocks);

    let (zero_pq, zero_hdr) = convert("zf_eq_zero", &text);
    let (full_pq, full_hdr) = convert("zf_eq_full", &text);
    let output = temp_dir().join("zf_eq_out.csf");

    let stats = partition_csfs(&zero_pq, &zero_hdr, &full_pq, &full_hdr, &output).unwrap();
    let parsed = parse_csf_output(&output);

    assert_eq!(stats.first_order_count, 0);
    assert_eq!(stats.output_csf_count, stats.full_csf_count);
    assert_eq!(parsed.blocks.len(), 2);
    assert_eq!(tags_of(&parsed.blocks[0]), vec!["A", "B", "C"]);
    assert_eq!(tags_of(&parsed.blocks[1]), vec!["D", "E"]);

    cleanup(&[
        zero_pq,
        zero_hdr,
        full_pq,
        full_hdr,
        output,
        temp_dir().join("zf_eq_zero"),
        temp_dir().join("zf_eq_full"),
    ]);
}

#[test]
fn test_partition_disjoint_zero_and_full() {
    // zero and full share no CSFs: output = zero ++ full, all of full is complement
    let zero_text = build_csf_text(&[vec!["X", "Y"], vec!["Z"]]);
    let full_text = build_csf_text(&[vec!["A", "B"], vec!["C"]]);

    let (zero_pq, zero_hdr) = convert("zf_dis_zero", &zero_text);
    let (full_pq, full_hdr) = convert("zf_dis_full", &full_text);
    let output = temp_dir().join("zf_dis_out.csf");

    let stats = partition_csfs(&zero_pq, &zero_hdr, &full_pq, &full_hdr, &output).unwrap();
    let parsed = parse_csf_output(&output);

    assert_eq!(stats.zero_csf_count, 3);
    assert_eq!(stats.full_csf_count, 3);
    assert_eq!(stats.first_order_count, 3);
    assert_eq!(stats.output_csf_count, 6);
    // zero first, then the entire full block (all complement)
    assert_eq!(tags_of(&parsed.blocks[0]), vec!["X", "Y", "A", "B"]);
    assert_eq!(tags_of(&parsed.blocks[1]), vec!["Z", "C"]);

    cleanup(&[
        zero_pq,
        zero_hdr,
        full_pq,
        full_hdr,
        output,
        temp_dir().join("zf_dis_zero"),
        temp_dir().join("zf_dis_full"),
    ]);
}
