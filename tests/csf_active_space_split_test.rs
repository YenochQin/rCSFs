use _rcsfs::descriptor_dataset::{DescriptorRows, header_path, import_text};
use _rcsfs::descriptor_operations::split_active_spaces;
use std::{fs, path::Path};

fn pair(root: &Path, stem: &str, space: &str) -> (std::path::PathBuf, std::path::PathBuf, String) {
    let output = root.join(format!("{stem}_descriptors.parquet"));
    let header = header_path(&output).unwrap();
    (output, header, space.into())
}

fn fixture(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let text = root.join("input.c");
    let (output, header, _) = pair(root, "input", "");
    fs::write(&text, "Core subshells:\n  1s\nPeel subshells:\n  2s   5s  15s\nCSF(s):\n  5s ( 2)\n\n        0+\n 15s ( 2)\n\n        0+\n *\n  2s ( 2)\n\n        0+\n").unwrap();
    import_text(&text, &output, &header).unwrap();
    fs::remove_file(text).unwrap();
    (output, header)
}

#[test]
fn overlapping_spaces_preserve_order_and_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let (input, header) = fixture(dir.path());
    let small = pair(dir.path(), "small", "5s");
    let large = pair(dir.path(), "large", "15s");
    let stats = split_active_spaces(&input, &header, &[small.clone(), large.clone()]).unwrap();
    assert_eq!(stats[0].block_lengths, [1, 1]);
    assert_eq!(stats[1].block_lengths, [2, 1]);
    let mut small_rows = DescriptorRows::open(&small.0, &small.1).unwrap();
    assert_eq!(small_rows.header.peel, ["2s", "5s"]);
    assert_eq!(small_rows.next_row().unwrap().unwrap().2[4], 2);
    assert_eq!(small_rows.next_row().unwrap().unwrap().2[0], 2);
    let mut large_rows = DescriptorRows::open(&large.0, &large.1).unwrap();
    assert_eq!(large_rows.header.peel, ["2s", "5s", "15s"]);
    assert_eq!(large_rows.next_row().unwrap().unwrap().2[4], 2);
    assert_eq!(large_rows.next_row().unwrap().unwrap().2[8], 2);
    assert_eq!(large_rows.next_row().unwrap().unwrap().2[0], 2);
}

#[test]
fn invalid_second_target_and_existing_output_preserve_first_target() {
    let dir = tempfile::tempdir().unwrap();
    let (input, header) = fixture(dir.path());
    let good = pair(dir.path(), "good", "5s");
    let invalid = pair(dir.path(), "bad", "5s,6s");
    assert!(split_active_spaces(&input, &header, &[good.clone(), invalid]).is_err());
    let existing = pair(dir.path(), "existing", "15s");
    fs::write(&existing.1, b"sentinel").unwrap();
    assert!(split_active_spaces(&input, &header, &[good.clone(), existing.clone()]).is_err());
    assert!(!good.0.exists() && !good.1.exists());
    assert!(!existing.0.exists());
    assert_eq!(fs::read(&existing.1).unwrap(), b"sentinel");
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 3);
}
