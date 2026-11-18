use _rcsfs::descriptor_dataset::{DescriptorRows, header_path, import_text, read_header};
use std::fs;

#[test]
fn text_import_counts_core_and_peel_electrons_and_binds_header() {
    for (core, electrons) in [("", 2), ("  1s   2s   2p-  2p", 12)] {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("source.c");
        let output = dir.path().join("out_descriptors.parquet");
        let header = header_path(&output).unwrap();
        fs::write(
            &text,
            format!("Core subshells:\n{core}\nPeel subshells:\n  3s   4s\nCSF(s):\n  3s ( 2)\n\n        0+\n *\n  4s ( 2)\n\n        0+\n"),
        ).unwrap();
        let stats = import_text(&text, &output, &header).unwrap();
        assert_eq!(stats.electron_count, electrons);
        assert_eq!(stats.block_lengths, [1, 1]);
        let parsed = read_header(&header).unwrap();
        assert_eq!(parsed.electron_count, Some(electrons));
        assert_eq!(parsed.block_lengths, [1, 1]);
        let mut descriptors = DescriptorRows::open(&output, &header).unwrap();
        assert!(descriptors.next_row().unwrap().is_some());
        let original = fs::read_to_string(&header).unwrap();
        fs::write(&header, format!("{original}\n# changed bytes\n")).unwrap();
        assert!(DescriptorRows::open(&output, &header).is_err());
    }
}
