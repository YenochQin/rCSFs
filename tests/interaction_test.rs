use _rcsfs::complete_csf::CompleteCsfFile;
use _rcsfs::csf_generation::{GenerationRequest, SubshellOccupation, generate_csfs};
use _rcsfs::descriptor_dataset::{
    DatasetHeader, DatasetWriter, DescriptorRows, header_path, import_text, restore_text,
};
use _rcsfs::descriptor_operations::zero_first;
use _rcsfs::interaction::{HamiltonianMode, InteractionMethod, select_interacting_csfs};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rcsfs-interaction-test-{}-{sequence}",
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

fn generated_record(entries: &[(&str, u8)], target_two_j: u16) -> String {
    let generated = generate_csfs(&GenerationRequest {
        core_subshells: Vec::new(),
        configuration: entries
            .iter()
            .map(|&(label, electrons)| SubshellOccupation {
                subshell: label.parse().unwrap(),
                electrons,
            })
            .collect(),
        min_two_j: target_two_j,
        max_two_j: target_two_j,
    })
    .unwrap();
    assert_eq!(generated.records.len(), 1, "test configuration ambiguity");
    let mut bytes = Vec::new();
    generated
        .write_record_to(&generated.records[0], &mut bytes)
        .unwrap();
    String::from_utf8(bytes).unwrap()
}

fn csf(core: &str, peel: &str, records: &[&str]) -> String {
    format!(
        "Core subshells:\n{core}\nPeel subshells:\n{peel}\nCSF(s):\n{}",
        records.concat()
    )
}

fn write_dataset(path: &Path, text: &str) {
    let input = path.with_extension("c");
    fs::write(&input, text).unwrap();
    import_text(&input, path, &header_path(path).unwrap()).unwrap();
    fs::remove_file(input).unwrap();
}

fn write_inputs(directory: &TestDirectory) -> (PathBuf, PathBuf, String, String, String) {
    let reference_record = generated_record(&[("1s", 2), ("2s", 2), ("2p", 1)], 3);
    let selected_record = generated_record(&[("1s", 2), ("2s", 2), ("3p", 1)], 3);
    // A double shell replacement requires four one-electron substitutions.
    let rejected_record = generated_record(&[("2p", 1), ("3s", 2), ("4s", 2)], 3);
    let reference = directory.join("reference_descriptors.parquet");
    let candidates = directory.join("candidates_descriptors.parquet");
    write_dataset(&reference, &csf("", "  1s   2s   2p", &[&reference_record]));
    write_dataset(
        &candidates,
        &csf(
            "",
            "  1s   2s   2p   3s   4s   3p",
            &[
                &reference_record,
                &selected_record,
                &selected_record,
                &rejected_record,
            ],
        ),
    );
    (
        reference,
        candidates,
        reference_record,
        selected_record,
        rejected_record,
    )
}

fn record_texts(path: &Path) -> (CompleteCsfFile, Vec<String>) {
    let dir = tempfile::tempdir().unwrap();
    let text = dir.path().join("restored.c");
    restore_text(path, &header_path(path).unwrap(), &text, None).unwrap();
    let parsed = CompleteCsfFile::parse_path(&text).unwrap();
    let records = parsed
        .records
        .iter()
        .map(|record| {
            let mut bytes = Vec::new();
            parsed.write_record_to(record, &mut bytes).unwrap();
            String::from_utf8(bytes).unwrap()
        })
        .collect();
    (parsed, records)
}

#[test]
fn selects_structural_upper_bound_skips_exact_mr_and_preserves_other_duplicates() {
    let directory = TestDirectory::new();
    let (reference, candidates, reference_record, selected_record, _) = write_inputs(&directory);
    let output = directory.join("selected_descriptors.parquet");

    let stats = select_interacting_csfs(
        &reference,
        &candidates,
        &output,
        HamiltonianMode::DiracCoulombBreit,
        InteractionMethod::StructuralUpperBound,
        Some(3),
        false,
    )
    .unwrap();

    assert!(!stats.exact);
    assert_eq!(stats.mode, HamiltonianMode::DiracCoulombBreit);
    assert_eq!(stats.method, InteractionMethod::StructuralUpperBound);
    assert_eq!(stats.block_count, 1);
    assert_eq!(stats.reference_count, 1);
    assert_eq!(stats.candidate_count, 4);
    assert_eq!(stats.exact_reference_skipped, 1);
    assert_eq!(stats.selected_count, 2);
    assert_eq!(stats.rejected_count, 1);
    assert_eq!(stats.output_count, 3);
    assert_eq!(stats.blocks[0].reference_count, 1);
    assert_eq!(stats.blocks[0].candidate_count, 4);
    assert_eq!(stats.blocks[0].exact_reference_skipped, 1);
    assert_eq!(stats.blocks[0].selected_count, 2);
    assert_eq!(stats.blocks[0].rejected_count, 1);
    assert_eq!(stats.blocks[0].output_count, 3);
    assert_eq!(stats.output_bytes, fs::metadata(&output).unwrap().len());

    let (parsed, records) = record_texts(&output);
    assert_eq!(parsed.header_lines[3], "  1s   2s   2p   3s   4s   3p");
    assert_eq!(
        records,
        [reference_record, selected_record.clone(), selected_record]
    );
}

#[test]
fn worker_count_does_not_change_output_order_or_stats() {
    let directory = TestDirectory::new();
    let (reference, candidates, _, _, _) = write_inputs(&directory);
    let serial_output = directory.join("serial_descriptors.parquet");
    let parallel_output = directory.join("parallel_descriptors.parquet");
    let serial = select_interacting_csfs(
        &reference,
        &candidates,
        &serial_output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(1),
        false,
    )
    .unwrap();
    let parallel = select_interacting_csfs(
        &reference,
        &candidates,
        &parallel_output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(4),
        false,
    )
    .unwrap();

    assert_eq!(record_texts(&serial_output), record_texts(&parallel_output));
    assert_eq!(serial.blocks, parallel.blocks);
    assert_eq!(serial.output_count, parallel.output_count);
}

#[test]
fn numerical_selection_and_partition_keep_explicit_zero_channels_across_batches() {
    let dir = tempfile::tempdir().unwrap();
    let path = |name: &str| dir.path().join(format!("{name}_descriptors.parquet"));
    let reference = path("reference");
    let candidates = path("candidates");
    let reference_header = header_path(&reference).unwrap();
    let candidate_header = header_path(&candidates).unwrap();
    let header = |peel: &str| {
        DatasetHeader::from_lines(
            ["Core subshells:", "", "Peel subshells:", peel, "CSF(s):"].map(str::to_owned),
        )
        .unwrap()
    };

    // Two closed s subshells followed by two spin-1/2 subshells coupled to J=0.
    let row = [2, -1, 0, -1, 2, -1, 0, -1, 1, 1, 1, -1, 1, 1, 1, -1, 0, 1];
    let mut writer =
        DatasetWriter::new(header("2s 3s 4s 5s"), &reference, &reference_header, &[]).unwrap();
    writer.push(&row, 0).unwrap();
    writer.finish().unwrap();

    let mut padded = row[..16].to_vec();
    padded.extend([0, -1, 0, -1, 0, 1]);
    let mut explicit_j = padded.clone();
    explicit_j[1] = 0;
    let mut explicit_k = padded.clone();
    explicit_k[7] = 0;

    let mut writer = DatasetWriter::new(
        header("2s 3s 4s 5s 6s"),
        &candidates,
        &candidate_header,
        &[],
    )
    .unwrap();
    for index in 0..=8192 {
        let row = match index {
            0 | 8192 => &explicit_j,
            4096 => &explicit_k,
            _ => &padded,
        };
        writer.push(row, 0).unwrap();
    }
    writer.finish().unwrap();

    let filtered = path("interacting");
    let interaction = select_interacting_csfs(
        &reference,
        &candidates,
        &filtered,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(3),
        false,
    )
    .unwrap();
    assert_eq!(interaction.exact_reference_skipped, 8190);
    assert_eq!(interaction.selected_count, 3);
    assert_eq!(interaction.rejected_count, 0);

    let partitioned = path("zero_first");
    let partition = zero_first(
        &reference,
        &reference_header,
        &candidates,
        &candidate_header,
        &partitioned,
        &header_path(&partitioned).unwrap(),
    )
    .unwrap();
    assert_eq!(partition.dataset.block_lengths, [4]);
    assert_eq!(partition.first_order_count, 3);

    for path in [filtered, partitioned] {
        let mut rows = DescriptorRows::open(&path, &header_path(&path).unwrap()).unwrap();
        for expected in [&padded, &explicit_j, &explicit_k, &explicit_j] {
            assert_eq!(&rows.next_row().unwrap().unwrap().2, expected);
        }
        assert!(rows.next_row().unwrap().is_none());
    }
}

#[test]
fn preserves_adjacent_explicit_blocks_with_the_same_symmetry() {
    let directory = TestDirectory::new();
    let record = generated_record(&[("2p", 1)], 3);
    let two_blocks = csf("", "  2p", &[&record, " *\n", &record]);
    let reference = directory.join("reference_descriptors.parquet");
    let candidates = directory.join("candidates_descriptors.parquet");
    let output = directory.join("selected_descriptors.parquet");
    write_dataset(&reference, &two_blocks);
    write_dataset(&candidates, &two_blocks);

    let stats = select_interacting_csfs(
        &reference,
        &candidates,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(2),
        false,
    )
    .unwrap();

    assert_eq!(stats.block_count, 2);
    let (parsed, _) = record_texts(&output);
    assert_eq!(parsed.blocks.len(), 2);
    assert_eq!(parsed.blocks[0].total_two_j, parsed.blocks[1].total_two_j);
    assert_eq!(parsed.blocks[0].parity, parsed.blocks[1].parity);
}

#[test]
fn rejects_duplicate_reference_csfs_without_creating_output() {
    let directory = TestDirectory::new();
    let record = generated_record(&[("2p", 1)], 3);
    let reference = directory.join("reference_descriptors.parquet");
    let candidates = directory.join("candidates_descriptors.parquet");
    let output = directory.join("selected_descriptors.parquet");
    write_dataset(&reference, &csf("", "  2p", &[&record, &record]));
    write_dataset(&candidates, &csf("", "  2p", &[&record]));

    let error = select_interacting_csfs(
        &reference,
        &candidates,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(2),
        false,
    )
    .unwrap_err();

    assert!(error.to_string().contains("duplicate CSFs"));
    assert!(!output.exists());
}

#[test]
fn rejects_the_same_file_for_reference_and_candidates() {
    let directory = TestDirectory::new();
    let (reference, _, _, _, _) = write_inputs(&directory);
    let output = directory.join("selected_descriptors.parquet");

    let error = select_interacting_csfs(
        &reference,
        &reference,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(1),
        false,
    )
    .unwrap_err();

    assert!(error.to_string().contains("must be different files"));
    assert!(!output.exists());
}

#[test]
fn validation_and_no_overwrite_errors_preserve_existing_output() {
    let directory = TestDirectory::new();
    let (reference, candidates, _, _, _) = write_inputs(&directory);
    let output = directory.join("existing_descriptors.parquet");
    fs::write(&output, b"sentinel").unwrap();

    let error = select_interacting_csfs(
        &reference,
        &candidates,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(1),
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("already exists"));
    assert_eq!(fs::read(&output).unwrap(), b"sentinel");

    let bad_candidates = directory.join("bad-candidates_descriptors.parquet");
    let record = generated_record(&[("2p", 1)], 3);
    write_dataset(&bad_candidates, &csf("  1s", "  2p", &[&record]));
    let error = select_interacting_csfs(
        &reference,
        &bad_candidates,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(2),
        true,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("core header mismatch"));
    assert_eq!(fs::read(&output).unwrap(), b"sentinel");
}

#[test]
fn overwrite_rejects_output_aliasing_an_input() {
    let directory = TestDirectory::new();
    let (reference, candidates, _, _, _) = write_inputs(&directory);
    let original_reference = fs::read(&reference).unwrap();
    let error = select_interacting_csfs(
        &reference,
        &candidates,
        &reference,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(1),
        true,
    )
    .unwrap_err();
    assert!(error.to_string().contains("must not be the reference"));
    assert_eq!(fs::read(&reference).unwrap(), original_reference);

    #[cfg(unix)]
    {
        let alias = directory.join("candidate-hard-link_descriptors.parquet");
        fs::hard_link(&candidates, &alias).unwrap();
        let original_candidates = fs::read(&candidates).unwrap();
        let error = select_interacting_csfs(
            &reference,
            &candidates,
            &alias,
            HamiltonianMode::DiracCoulomb,
            InteractionMethod::StructuralUpperBound,
            Some(1),
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("hard link to the candidate"));
        assert_eq!(fs::read(&candidates).unwrap(), original_candidates);
    }
}

#[test]
fn rejects_non_prefix_peel_and_misaligned_jp_without_creating_output() {
    let directory = TestDirectory::new();
    let reference_record = generated_record(&[("2p", 1)], 3);
    let reference = directory.join("reference_descriptors.parquet");
    write_dataset(&reference, &csf("", "  2p", &[&reference_record]));

    let non_prefix = directory.join("non-prefix_descriptors.parquet");
    write_dataset(&non_prefix, &csf("", "  3p   2p", &[&reference_record]));
    let output = directory.join("must-not-exist_descriptors.parquet");
    let error = select_interacting_csfs(
        &reference,
        &non_prefix,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        None,
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("must be a prefix"));
    assert!(!output.exists());

    let different_jp_record = generated_record(&[("2s", 1)], 1);
    let different_jp = directory.join("different-jp_descriptors.parquet");
    write_dataset(
        &different_jp,
        &csf("", "  2p   2s", &[&different_jp_record]),
    );
    let error = select_interacting_csfs(
        &reference,
        &different_jp,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        None,
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("J/P mismatch"));
    assert!(!output.exists());
}

#[test]
fn zero_workers_is_rejected_before_touching_output() {
    let directory = TestDirectory::new();
    let (reference, candidates, _, _, _) = write_inputs(&directory);
    let output = directory.join("output_descriptors.parquet");
    let error = select_interacting_csfs(
        &reference,
        &candidates,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(0),
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("greater than zero"));
    assert!(!output.exists());
}

#[test]
fn complete_output_is_parseable_after_overwrite() {
    let directory = TestDirectory::new();
    let (reference, candidates, _, _, _) = write_inputs(&directory);
    let output = directory.join("output_descriptors.parquet");
    fs::write(&output, b"old").unwrap();
    select_interacting_csfs(
        &reference,
        &candidates,
        &output,
        HamiltonianMode::DiracCoulomb,
        InteractionMethod::StructuralUpperBound,
        Some(2),
        true,
    )
    .unwrap();
    let (parsed, _) = record_texts(&output);
    assert_eq!(parsed.records.len(), 3);
}
