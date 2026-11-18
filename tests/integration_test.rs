use _rcsfs::descriptor_dataset::{DescriptorRows, header_path, import_text, restore_text};
use std::{fs, path::Path};

const HEADER: &str = "Core subshells:\n  1s\nPeel subshells:\n  2s   3s\nCSF(s):\n";
const A: &str = "  2s ( 2)\n\n        0+\n";
const B: &str = "  3s ( 2)\n\n        0+\n";

#[test]
fn text_import_and_explicit_restore_keep_rows_and_blocks_across_batches() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.c");
    let output = dir.path().join("output_descriptors.parquet");
    let header = header_path(&output).unwrap();
    let first = [A, B].repeat(4_500).concat();
    fs::write(&input, format!("{HEADER}{first} *\n{B}")).unwrap();
    let stats = import_text(&input, &output, &header).unwrap();
    assert_eq!(stats.block_lengths, [9_000, 1]);
    assert_eq!(stats.record_count, 9_001);
    let mut reader = DescriptorRows::open(&output, &header).unwrap();
    for i in 0..9_001 {
        let (index, block, row) = reader.next_row().unwrap().unwrap();
        assert_eq!(index, i);
        assert_eq!(block, usize::from(i == 9_000));
        let occupations = if i % 2 == 0 && i != 9_000 {
            (2, 0)
        } else {
            (0, 2)
        };
        assert_eq!((row[0], row[4]), occupations);
    }
    assert!(reader.next_row().unwrap().is_none());
    let restored = dir.path().join("restored.c");
    restore_text(&output, &header, &restored, Some(&[0, 8_191, 8_192, 9_000])).unwrap();
    let text = fs::read_to_string(restored).unwrap();
    assert_eq!(text.matches("\n *\n").count(), 1);
    assert_eq!(text.matches("  2s ( 2)").count(), 2);
    assert_eq!(text.matches("  3s ( 2)").count(), 2);
}

#[test]
fn long_valid_lines_are_imported_without_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.c");
    let output = dir.path().join("output_descriptors.parquet");
    let header = header_path(&output).unwrap();
    // Fixed-width occupation fields longer than the retired 256-byte limit.
    let peel = (2..=15)
        .flat_map(|n| [format!("{n}s"), format!("{n}p-"), format!("{n}p")])
        .collect::<Vec<_>>();
    let peel_electrons = (2..=15).map(|_| 2 + 2 + 4).sum::<usize>();
    let occupations = peel
        .iter()
        .map(|shell| {
            format!(
                " {:>4}({:>2})",
                shell,
                if shell.ends_with('p') { 4 } else { 2 }
            )
        })
        .collect::<String>();
    assert!(occupations.len() > 256);
    let final_j = format!("{}0+", " ".repeat(occupations.len() - 2));
    fs::write(
        &input,
        format!(
            "Core subshells:\n  1s\nPeel subshells:\n{}\nCSF(s):\n{occupations}\n\n{final_j}\n",
            peel.join(" ")
        ),
    )
    .unwrap();
    let stats = import_text(&input, &output, &header).unwrap();
    assert_eq!(stats.electron_count, peel_electrons + 2);
    let mut rows = DescriptorRows::open(&output, &header).unwrap();
    let row = rows.next_row().unwrap().unwrap().2;
    assert_eq!(row[4 * 41], 4);
}

#[test]
fn invalid_text_or_paths_leave_no_descriptor_pair() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.c");
    let output = dir.path().join("output_descriptors.parquet");
    let header = header_path(&output).unwrap();
    assert!(import_text(&input, &output, &header).is_err());
    for body in [
        "".to_owned(),
        format!("{A}  2s ( 2)\n"),
        format!("{A} 测试 ( 2)\n\n0+\n"),
        format!("{A}  9s ( 2)\n\n0+\n"),
        format!("{A} *\n"),
    ] {
        fs::write(&input, format!("{HEADER}{body}")).unwrap();
        assert!(import_text(&input, &output, &header).is_err(), "{body:?}");
        assert!(!output.exists() && !header.exists());
    }
    fs::write(&input, format!("{HEADER}{A}")).unwrap();
    let missing = dir.path().join("missing/out.parquet");
    assert!(import_text(&input, &missing, &header).is_err());
    assert!(!header.exists());
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    assert!(Path::new(&input).is_file());
}
