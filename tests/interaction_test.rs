use _rcsfs::complete_csf::CompleteCsfFile;
use _rcsfs::csf_generation::{GenerationRequest, SubshellOccupation, generate_csfs};
use _rcsfs::interaction::{HamiltonianMode, InteractionMethod, select_interacting_csfs};
use std::fs;
use std::io::Cursor;
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

fn write_inputs(directory: &TestDirectory) -> (PathBuf, PathBuf, String, String, String) {
    let reference_record = generated_record(&[("2p", 1)], 3);
    let selected_record = generated_record(&[("3p", 1)], 3);
    // Same J/P, but two additional peel electrons: electron conservation rejects it.
    let rejected_record = generated_record(&[("2p", 1), ("3s", 2)], 3);
    let reference = directory.join("reference.csf");
    let candidates = directory.join("candidates.csf");
    fs::write(&reference, csf("", "  2p", &[&reference_record])).unwrap();
    fs::write(
        &candidates,
        csf(
            "",
            "  2p   3s   3p",
            &[
                &reference_record,
                &selected_record,
                &selected_record,
                &rejected_record,
            ],
        ),
    )
    .unwrap();
    (
        reference,
        candidates,
        reference_record,
        selected_record,
        rejected_record,
    )
}

fn record_texts(path: &Path) -> (CompleteCsfFile, Vec<String>) {
    let parsed = CompleteCsfFile::parse_path(path).unwrap();
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
    let output = directory.join("selected.csf");

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
    assert_eq!(parsed.header_lines[3], "  2p   3s   3p");
    assert_eq!(
        records,
        [reference_record, selected_record.clone(), selected_record]
    );
}

#[test]
fn worker_count_does_not_change_output_order_or_stats() {
    let directory = TestDirectory::new();
    let (reference, candidates, _, _, _) = write_inputs(&directory);
    let serial_output = directory.join("serial.csf");
    let parallel_output = directory.join("parallel.csf");
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

    assert_eq!(
        fs::read(serial_output).unwrap(),
        fs::read(parallel_output).unwrap()
    );
    assert_eq!(serial, parallel);
}

#[test]
fn preserves_adjacent_explicit_blocks_with_the_same_symmetry() {
    let directory = TestDirectory::new();
    let record = generated_record(&[("2p", 1)], 3);
    let two_blocks = csf("", "  2p", &[&record, " *\n", &record]);
    let reference = directory.join("reference.csf");
    let candidates = directory.join("candidates.csf");
    let output = directory.join("selected.csf");
    fs::write(&reference, &two_blocks).unwrap();
    fs::write(&candidates, &two_blocks).unwrap();

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
    let parsed = CompleteCsfFile::parse_path(&output).unwrap();
    assert_eq!(parsed.blocks.len(), 2);
    assert_eq!(parsed.blocks[0].total_two_j, parsed.blocks[1].total_two_j);
    assert_eq!(parsed.blocks[0].parity, parsed.blocks[1].parity);
}

#[test]
fn rejects_duplicate_reference_csfs_without_creating_output() {
    let directory = TestDirectory::new();
    let record = generated_record(&[("2p", 1)], 3);
    let reference = directory.join("reference.csf");
    let candidates = directory.join("candidates.csf");
    let output = directory.join("selected.csf");
    fs::write(&reference, csf("", "  2p", &[&record, &record])).unwrap();
    fs::write(&candidates, csf("", "  2p", &[&record])).unwrap();

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
    let output = directory.join("selected.csf");

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
    let output = directory.join("existing.csf");
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

    let bad_candidates = directory.join("bad-candidates.csf");
    let text = fs::read_to_string(&candidates).unwrap().replacen(
        "Core subshells:\n\n",
        "Core subshells:\n  1s\n",
        1,
    );
    fs::write(&bad_candidates, text).unwrap();
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
        let alias = directory.join("candidate-hard-link.csf");
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
    let reference = directory.join("reference.csf");
    fs::write(&reference, csf("", "  2p", &[&reference_record])).unwrap();

    let non_prefix = directory.join("non-prefix.csf");
    fs::write(&non_prefix, csf("", "  3p   2p", &[&reference_record])).unwrap();
    let output = directory.join("must-not-exist.csf");
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
    let different_jp = directory.join("different-jp.csf");
    fs::write(&different_jp, csf("", "  2p   2s", &[&different_jp_record])).unwrap();
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
    let output = directory.join("output.csf");
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
    let output = directory.join("output.csf");
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
    let bytes = fs::read(&output).unwrap();
    CompleteCsfFile::parse_reader(Cursor::new(bytes)).unwrap();
}
