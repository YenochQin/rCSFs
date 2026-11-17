use _rcsfs::descriptor_dataset::{DescriptorRows, header_path, import_text, restore_text, select};
use _rcsfs::descriptor_operations::{
    filter_active_space, match_indices, merge, split_blocks, zero_first,
};
use std::fs;
use std::path::{Path, PathBuf};

const CSFS: &str = "Core subshells:\n  1s\nPeel subshells:\n  2s   3s\nCSF(s):\n  2s ( 2)\n\n        0+\n *\n  3s ( 2)\n\n        0+\n";

#[test]
fn import_select_restore_preserves_explicit_equal_symmetry_blocks() {
    let tmp = tempfile::tempdir().unwrap();
    let text = tmp.path().join("input.c");
    fs::write(&text, CSFS).unwrap();
    let path = tmp.path().join("basis_descriptors.parquet");
    let header = header_path(&path).unwrap();
    let stats = import_text(&text, &path, &header).unwrap();
    assert_eq!(stats.block_lengths, [1, 1]);
    assert_eq!(stats.electron_count, 4);
    fs::remove_file(&text).unwrap();
    let mut source = DescriptorRows::open(&path, &header).unwrap();
    assert_eq!(
        source.next_row().unwrap().unwrap(),
        (0, 0, vec![2, -1, 0, -1, 0, -1, 0, -1, 0, 1])
    );
    assert_eq!(source.next_row().unwrap().unwrap().1, 1);
    assert!(source.next_row().unwrap().is_none());
    let restored = tmp.path().join("restored.c");
    restore_text(&path, &header, &restored, None).unwrap();
    assert!(fs::read_to_string(&restored).unwrap().contains("\n *\n"));
    let selected = tmp.path().join("selected_descriptors.parquet");
    let selected_header = header_path(&selected).unwrap();
    let stats = select(&path, &header, &selected, &selected_header, &[1]).unwrap();
    assert_eq!(stats.block_lengths, [1]);
    let mut source = DescriptorRows::open(&selected, &selected_header).unwrap();
    assert_eq!(source.next_row().unwrap().unwrap().2[4], 2);
    assert!(source.next_row().unwrap().is_none());
}

#[test]
fn pure_closed_core_exports_and_reimports_explicit_record() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("core.c");
    fs::write(
        &input,
        "Core subshells:\n  1s\nPeel subshells:\n\nCSF(s):\n",
    )
    .unwrap();
    let path = tmp.path().join("core_descriptors.parquet");
    let header = header_path(&path).unwrap();
    let stats = import_text(&input, &path, &header).unwrap();
    assert_eq!((stats.record_count, stats.electron_count), (1, 2));
    let restored = tmp.path().join("export.c");
    restore_text(&path, &header, &restored, None).unwrap();
    let copy = tmp.path().join("copy_descriptors.parquet");
    let copy_header = header_path(&copy).unwrap();
    import_text(&restored, &copy, &copy_header).unwrap();
    let mut source = DescriptorRows::open(&copy, &copy_header).unwrap();
    assert_eq!(source.next_row().unwrap().unwrap().2, [0, 1]);
}

#[test]
fn invalid_inputs_do_not_publish_partial_pairs() {
    let tmp = tempfile::tempdir().unwrap();
    for (index, content) in [
        CSFS.trim_end_matches("        0+\n").to_owned(),
        CSFS.to_owned() + " *\n",
    ]
    .iter()
    .enumerate()
    {
        let text = tmp.path().join(format!("bad{index}.c"));
        fs::write(&text, content).unwrap();
        let output = tmp.path().join(format!("bad{index}_descriptors.parquet"));
        let header = header_path(&output).unwrap();
        assert!(import_text(&text, &output, &header).is_err());
        assert!(!output.exists() && !header.exists());
    }
    let text = tmp.path().join("valid.c");
    fs::write(&text, CSFS).unwrap();
    let output = tmp.path().join("valid_descriptors.parquet");
    let header = header_path(&output).unwrap();
    import_text(&text, &output, &header).unwrap();
    for indices in [&[2][..], &[1, 0][..], &[0, 0][..], &[][..]] {
        let selected = tmp.path().join("selected_descriptors.parquet");
        let selected_header = header_path(&selected).unwrap();
        assert!(select(&output, &header, &selected, &selected_header, indices).is_err());
        assert!(!selected.exists() && !selected_header.exists());
    }
    fs::write(&header, fs::read_to_string(&header).unwrap() + "\n").unwrap();
    assert!(DescriptorRows::open(&output, &header).is_err());
}

fn basis(root: &Path, name: &str, peel: &str, occupied: &[&str]) -> (PathBuf, PathBuf) {
    let text = root.join(format!("{name}.c"));
    let mut content = format!("Core subshells:\n\nPeel subshells:\n{peel}\nCSF(s):\n");
    for shell in occupied {
        content.push_str(&format!("{shell:>4} ( 2)\n\n        0+\n"));
    }
    fs::write(&text, content).unwrap();
    let output = root.join(format!("{name}_descriptors.parquet"));
    let header = header_path(&output).unwrap();
    import_text(&text, &output, &header).unwrap();
    fs::remove_file(text).unwrap();
    (output, header)
}

#[test]
fn numerical_merge_partition_matching_and_active_space() {
    let tmp = tempfile::tempdir().unwrap();
    let full = basis(tmp.path(), "full", "1s 2s 3s", &["1s", "2s", "3s"]);
    let seed = basis(tmp.path(), "seed", "1s 2s", &["2s"]);
    assert_eq!(
        match_indices(&full.0, &full.1, &seed.0, &seed.1, true).unwrap(),
        [Some(1)]
    );
    let reordered = tmp.path().join("zf_descriptors.parquet");
    let header = header_path(&reordered).unwrap();
    let stats = zero_first(&seed.0, &seed.1, &full.0, &full.1, &reordered, &header).unwrap();
    assert_eq!((stats.zero_csf_count, stats.first_order_count), (1, 2));
    let mut rows = DescriptorRows::open(&reordered, &header).unwrap();
    assert_eq!(rows.next_row().unwrap().unwrap().2[4], 2);
    assert_eq!(rows.next_row().unwrap().unwrap().2[0], 2);
    assert_eq!(rows.next_row().unwrap().unwrap().2[8], 2);
    let active = tmp.path().join("as_descriptors.parquet");
    let active_header = header_path(&active).unwrap();
    assert_eq!(
        filter_active_space(&full.0, &full.1, &active, &active_header, "2s")
            .unwrap()
            .record_count,
        2
    );
    assert_eq!(
        DescriptorRows::open(&active, &active_header)
            .unwrap()
            .header
            .peel,
        ["1s", "2s"]
    );
    let merged = tmp.path().join("merged_descriptors.parquet");
    let merged_header = header_path(&merged).unwrap();
    assert_eq!(
        merge(&[seed, full], &merged, &merged_header)
            .unwrap()
            .block_lengths,
        [1, 3]
    );
    let blocks = split_blocks(&merged, &merged_header, tmp.path(), "blocks").unwrap();
    assert_eq!(blocks.len(), 2);
    assert_ne!(blocks[0].output_file, blocks[1].output_file);
}

#[test]
fn conflicting_orbital_orders_are_rejected_before_output() {
    let tmp = tempfile::tempdir().unwrap();
    let left = basis(tmp.path(), "left", "1s 2s", &["1s"]);
    let right = basis(tmp.path(), "right", "2s 1s", &["2s"]);
    let output = tmp.path().join("output_descriptors.parquet");
    let header = header_path(&output).unwrap();
    assert!(merge(&[left, right], &output, &header).is_err());
    assert!(!output.exists() && !header.exists());
}

#[test]
fn selection_crosses_record_batch_and_row_group_boundaries() {
    let tmp = tempfile::tempdir().unwrap();
    let input = basis(tmp.path(), "large", "1s", &vec!["1s"; 16_385]);
    let output = tmp.path().join("subset_descriptors.parquet");
    let header = header_path(&output).unwrap();
    assert_eq!(
        select(&input.0, &input.1, &output, &header, &[8191, 8192, 16384])
            .unwrap()
            .record_count,
        3
    );
    let mut rows = DescriptorRows::open(&output, &header).unwrap();
    for _ in 0..3 {
        assert_eq!(rows.next_row().unwrap().unwrap().2[0], 2);
    }
    assert!(rows.next_row().unwrap().is_none());
}
