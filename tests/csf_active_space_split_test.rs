use _rcsfs::csf_active_space_split::{ActiveSpaceTarget, split_csfs_by_active_spaces};
use _rcsfs::csfs_conversion::convert_csfs_to_parquet;
use std::fs;
use std::path::PathBuf;

fn fixture(name: &str, invalid_line: bool) -> (PathBuf, PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/test_outputs/active_space_split")
        .join(name);
    if root.exists() {
        fs::remove_dir_all(&root).unwrap();
    }
    fs::create_dir_all(&root).unwrap();
    let input = root.join("input.c");
    let parquet = root.join("input.parquet");
    let header = root.join("input_header.toml");
    let third = if invalid_line {
        "  2s ( 2)  15g (oops)"
    } else {
        "  2s ( 2)  15g ( 1)"
    };
    let lines = [
        "Core subshells:",
        "",
        "Peel subshells:",
        "  2s   5g-  5g  15g",
        "CSF(s):",
        "  2s ( 2)  5g-( 1)",
        "L2-a",
        "L3-a",
        "  2s ( 2)  5g ( 1)",
        "L2-b",
        "L3-b",
        third,
        "L2-c",
        "L3-c",
        " *",
        "  2s ( 2)  5g ( 2)",
        "L2-d",
        "L3-d",
        "  2s ( 2)  15g ( 0)",
        "L2-e",
        "L3-e",
    ];
    fs::write(&input, format!("{}\n", lines.join("\n"))).unwrap();
    convert_csfs_to_parquet(&input, &parquet, 256, 1000).unwrap();
    (root, parquet, header)
}

#[test]
fn overlapping_spaces_preserve_order_blocks_and_exact_orbital_names() {
    let (root, parquet, header) = fixture("overlap", false);
    let small = root.join("small.c");
    let large = root.join("large.c");
    let targets = [
        ActiveSpaceTarget {
            maximum_orbitals: "5s,5g".into(),
            output: small.clone(),
        },
        ActiveSpaceTarget {
            maximum_orbitals: "5s,15g".into(),
            output: large.clone(),
        },
    ];
    let stats = split_csfs_by_active_spaces(&parquet, &header, &targets).unwrap();
    assert_eq!(stats.input_csf_count, 5);
    assert_eq!(stats.block_count, 2);
    assert_eq!(stats.outputs[0].block_lengths, vec![2, 1]);
    assert_eq!(stats.outputs[1].block_lengths, vec![3, 2]);
    let small_text = fs::read_to_string(&small).unwrap();
    let large_text = fs::read_to_string(&large).unwrap();
    assert_eq!(
        large_text,
        fs::read_to_string(root.join("input.c")).unwrap()
    );
    assert!(small_text.contains("L2-a\nL3-a\n"));
    assert!(small_text.contains("L2-b\nL3-b\n"));
    assert!(!small_text.contains("15g ( 1)"));
    assert!(!small_text.contains("15g ( 0)"));
    assert!(large_text.contains("15g ( 1)"));
    assert!(large_text.contains("15g ( 0)"));
    assert_eq!(small_text.matches("\n *\n").count(), 1);
    assert_eq!(large_text.matches("\n *\n").count(), 1);
    assert_eq!(small_text.lines().nth(3), Some("  2s   5g-  5g"));
    assert_eq!(large_text.lines().nth(3), Some("  2s   5g-  5g  15g"));
    assert!(small_text.find("L2-a").unwrap() < small_text.find("L2-b").unwrap());
    assert!(small_text.find("L2-b").unwrap() < small_text.find("L2-d").unwrap());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_csf_does_not_publish_partial_outputs() {
    let (root, parquet, header) = fixture("invalid", true);
    let output = root.join("result.c");
    let target = [ActiveSpaceTarget {
        maximum_orbitals: "5s,15g".into(),
        output: output.clone(),
    }];
    let error = split_csfs_by_active_spaces(&parquet, &header, &target).unwrap_err();
    assert!(format!("{error:#}").contains("invalid occupation"));
    assert!(!output.exists());
    assert!(!fs::read_dir(&root).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    }));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn existing_output_is_never_replaced() {
    let (root, parquet, header) = fixture("existing", false);
    let output = root.join("result.c");
    fs::write(&output, "sentinel").unwrap();
    let target = [ActiveSpaceTarget {
        maximum_orbitals: "5s,15g".into(),
        output: output.clone(),
    }];
    let error = split_csfs_by_active_spaces(&parquet, &header, &target).unwrap_err();
    assert!(format!("{error:#}").contains("output already exists"));
    assert_eq!(fs::read_to_string(output).unwrap(), "sentinel");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_orbital_limits_fail_before_any_output_is_created() {
    let (root, parquet, header) = fixture("invalid-limits", false);
    let output = root.join("result.c");
    let target = [ActiveSpaceTarget {
        maximum_orbitals: "5s,6s".into(),
        output: output.clone(),
    }];
    let error = split_csfs_by_active_spaces(&parquet, &header, &target).unwrap_err();
    assert!(format!("{error:#}").contains("duplicate active orbital symmetry"));
    assert!(!output.exists());
    assert!(!fs::read_dir(&root).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    }));
    fs::remove_dir_all(root).unwrap();
}
