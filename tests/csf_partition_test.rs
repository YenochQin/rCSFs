use _rcsfs::descriptor_dataset::{DescriptorRows, header_path, import_text, read_header, select};
use _rcsfs::descriptor_operations::zero_first;
use std::{
    fs,
    path::{Path, PathBuf},
};

fn dataset(dir: &Path, name: &str, blocks: &[&[u8]]) -> (PathBuf, PathBuf) {
    let text = dir.join(format!("{name}.c"));
    let path = dir.join(format!("{name}_descriptors.parquet"));
    let header = header_path(&path).unwrap();
    let contents = blocks
        .iter()
        .map(|block| {
            block
                .iter()
                .map(|n| format!("  {n}s ( 2)\n\n        0+\n"))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" *\n");
    fs::write(
        &text,
        format!("Core subshells:\n  1s\nPeel subshells:\n  2s   3s   4s\nCSF(s):\n{contents}"),
    )
    .unwrap();
    import_text(&text, &path, &header).unwrap();
    fs::remove_file(text).unwrap();
    (path, header)
}

#[test]
fn zero_first_preserves_each_block_and_appends_ordered_complement() {
    for (reference, full, expected, counts) in [
        (vec![2], vec![3, 2, 4], vec![2, 3, 4], (1, 2)),
        (vec![3, 2], vec![3, 2], vec![3, 2], (2, 0)),
        (vec![2], vec![3, 4], vec![2, 3, 4], (1, 2)),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let zero = dataset(dir.path(), "zero", &[&reference, &reference]);
        let all = dataset(dir.path(), "all", &[&full, &full]);
        let output = dir.path().join("partition_descriptors.parquet");
        let header = header_path(&output).unwrap();
        let stats = zero_first(&zero.0, &zero.1, &all.0, &all.1, &output, &header).unwrap();
        assert_eq!(stats.dataset.block_lengths, [expected.len(); 2]);
        assert_eq!(stats.blocks.len(), 2);
        assert!(
            stats
                .blocks
                .iter()
                .all(|b| (b.zero_count, b.first_order_count) == counts)
        );
        let mut rows = DescriptorRows::open(&output, &header).unwrap();
        let block_info = &rows.header().document["block_info"];
        for count in block_info["zero_order_counts"].as_array().unwrap() {
            assert_eq!(count.as_integer(), Some(counts.0 as i64));
        }
        assert_eq!(block_info["zero_order_counts"].as_array().unwrap().len(), 2);
        assert_eq!(rows.header().peel, ["2s", "3s", "4s"]);
        for block in 0..2 {
            for &n in &expected {
                let (_, actual_block, row) = rows.next_row().unwrap().unwrap();
                assert_eq!(actual_block, block);
                assert_eq!(row[(n - 2) as usize * 4], 2);
            }
        }
        assert!(rows.next_row().unwrap().is_none());

        let selected = dir.path().join("selected_descriptors.parquet");
        let selected_header = header_path(&selected).unwrap();
        select(&output, &header, &selected, &selected_header, &[0]).unwrap();
        let selection = read_header(&selected_header).unwrap();
        assert_eq!(
            selection.document["block_info"]["zero_order_counts"][0].as_integer(),
            Some(1)
        );
        if counts.1 > 0 {
            let complement = dir.path().join("complement_desc.parquet");
            let complement_header = header_path(&complement).unwrap();
            select(
                &output,
                &header,
                &complement,
                &complement_header,
                &[counts.0 as u64],
            )
            .unwrap();
            let selection = read_header(&complement_header).unwrap();
            assert_eq!(
                selection.document["block_info"]["zero_order_counts"][0].as_integer(),
                Some(0)
            );
        }
    }
}

#[test]
fn block_count_mismatch_creates_no_output() {
    let dir = tempfile::tempdir().unwrap();
    let zero = dataset(dir.path(), "zero", &[&[2]]);
    let all = dataset(dir.path(), "all", &[&[2], &[3]]);
    let output = dir.path().join("partition_descriptors.parquet");
    let header = header_path(&output).unwrap();
    let error = zero_first(&zero.0, &zero.1, &all.0, &all.1, &output, &header).unwrap_err();
    assert!(error.to_string().contains("block counts differ"));
    assert!(!output.exists() && !header.exists());
}
