use _rcsfs::descriptor_dataset::{DescriptorRows, header_path, import_text};
use _rcsfs::descriptor_operations::split_blocks;
use std::{fs, path::Path};

#[test]
fn splitting_preserves_numeric_rows_and_explicit_block_order() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input_descriptors.parquet");
    let header = header_path(&input).unwrap();
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/descriptor_multiblock.c");
    let stats = import_text(&fixture, &input, &header).unwrap();
    let mut source = DescriptorRows::open(&input, &header).unwrap();
    let outputs = split_blocks(&input, &header, dir.path(), "split").unwrap();
    assert_eq!(outputs.len(), stats.block_count);
    for (i, stats) in outputs.iter().enumerate() {
        assert_eq!(stats.block_lengths, [1]);
        let expected = source.next_row().unwrap().unwrap();
        assert_eq!(expected.1, i);
        let mut output =
            DescriptorRows::open(Path::new(&stats.output_file), Path::new(&stats.header_file))
                .unwrap();
        assert_eq!(output.next_row().unwrap().unwrap(), (0, 0, expected.2));
        assert!(output.next_row().unwrap().is_none());
    }
    assert!(source.next_row().unwrap().is_none());
}

#[test]
fn invalid_prefix_missing_directory_and_late_existing_output_do_not_publish() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input_descriptors.parquet");
    let header = header_path(&input).unwrap();
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/descriptor_multiblock.c");
    import_text(&fixture, &input, &header).unwrap();
    assert!(split_blocks(&input, &header, dir.path(), "../escape").is_err());
    assert!(split_blocks(&input, &header, &dir.path().join("missing"), "split").is_err());
    let second = dir.path().join("split_b001_2_odd_desc.parquet");
    fs::write(&second, b"existing").unwrap();
    assert!(split_blocks(&input, &header, dir.path(), "split").is_err());
    assert_eq!(fs::read(second).unwrap(), b"existing");
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 3);
}
