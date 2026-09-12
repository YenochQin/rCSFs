use _rcsfs::complete_csf::{CompleteCsfFile, Parity};
use _rcsfs::csf_generation::{
    EnumeratedConfiguration, ExcitationRequest, GenerationRequest, Subshell, SubshellOccupation,
    enumerate_occupations, generate_csfs, generate_csfs_parallel,
};
use std::collections::BTreeMap;
use std::io::Cursor;

fn request(entries: &[(&str, u8)], min_two_j: u16, max_two_j: u16) -> GenerationRequest {
    GenerationRequest {
        core_subshells: Vec::new(),
        configuration: entries
            .iter()
            .map(|&(label, electrons)| SubshellOccupation {
                subshell: label.parse().unwrap(),
                electrons,
            })
            .collect(),
        min_two_j,
        max_two_j,
    }
}

#[test]
fn parallel_generation_preserves_task_order_and_records() {
    let requests = vec![
        closed_configuration(2),
        closed_configuration(3),
        closed_configuration(4),
    ];
    let serial = requests
        .iter()
        .map(|request| generate_csfs(request).unwrap())
        .collect::<Vec<_>>();
    let parallel = generate_csfs_parallel(&requests, Some(2)).unwrap();
    assert_eq!(parallel.len(), serial.len());
    for (left, right) in parallel.iter().zip(serial.iter()) {
        assert_eq!(left.records, right.records);
        assert_eq!(left.blocks, right.blocks);
    }
}

#[test]
fn complete_integer_record_derives_dense_descriptor() {
    let generated = generate_csfs(&request(&[("2p", 1)], 1, 5)).unwrap();
    let descriptor = generated.descriptor_for(&generated.records[0]).unwrap();
    assert_eq!(descriptor, [1, 3, 3]);
}

#[test]
fn complete_integer_descriptor_matches_normalization_contract() {
    let generated = generate_csfs(&request(&[("2p", 1)], 1, 5)).unwrap();
    let descriptor = generated.descriptor_for(&generated.records[0]).unwrap();
    let normalized = _rcsfs::descriptor_normalization::normalize_descriptor_per_csf(
        &descriptor,
        &generated.subshells,
        generated.records[0].total_two_j as i32,
    )
    .unwrap();
    assert_eq!(normalized.len(), descriptor.len());
    assert!(normalized.iter().all(|value| value.is_finite()));
}

fn roundtrip(request: &GenerationRequest) -> CompleteCsfFile {
    let generated = generate_csfs(request).unwrap();
    let mut text = Vec::new();
    generated.write_to(&mut text).unwrap();
    let parsed = CompleteCsfFile::parse_reader(Cursor::new(text)).unwrap();
    assert_eq!(generated, parsed);
    generated
}

#[test]
fn one_electron_generates_a_new_half_integer_csf() {
    let result = roundtrip(&request(&[("2p", 1)], 1, 5));
    assert_eq!(result.records.len(), 1);
    assert_eq!(result.records[0].total_two_j, 3);
    assert_eq!(result.records[0].parity, Parity::Odd);
}

#[test]
fn equivalent_electrons_follow_pauli_allowed_j_values() {
    let result = roundtrip(&request(&[("2p", 2)], 0, 6));
    assert_eq!(
        result
            .blocks
            .iter()
            .map(|b| b.total_two_j)
            .collect::<Vec<_>>(),
        [0, 4]
    );
    assert_eq!(
        result.occupied(&result.records[0]).unwrap()[0]
            .state
            .unwrap()
            .two_j,
        0
    );
    let holes = roundtrip(&request(&[("3d", 4)], 0, 8));
    let electrons = roundtrip(&request(&[("3d", 2)], 0, 8));
    assert_eq!(holes.blocks, electrons.blocks);
}

#[test]
fn repeated_j_values_keep_distinct_seniority_states_in_table_order() {
    let result = roundtrip(&request(&[("4f", 4)], 4, 4));
    let seniorities = result
        .records
        .iter()
        .map(|r| result.occupied(r).unwrap()[0].state.unwrap().seniority)
        .collect::<Vec<_>>();
    assert_eq!(seniorities, [Some(2), Some(4)]);
}

#[test]
fn closed_shells_suppress_states_and_leading_couplings() {
    let mut input = request(&[("2s", 2), ("2p-", 1), ("2p", 1), ("3s", 1)], 1, 5);
    input.core_subshells.push("1s".parse().unwrap());
    let result = roundtrip(&input);
    assert_eq!(result.header_lines[1], "  1s");
    for record in &result.records {
        assert_eq!(result.occupied(record).unwrap()[0].state, None);
        let couplings = result.couplings(record).unwrap();
        assert_eq!(couplings.len(), 1);
        assert_eq!(couplings[0].boundary, 3);
    }
}

#[test]
fn coupling_chain_enumerates_multiplicities_and_sorted_blocks() {
    let result = roundtrip(&request(&[("1s", 1), ("2s", 1), ("3s", 1)], 1, 3));
    assert_eq!(result.records.len(), 3);
    assert_eq!(
        result
            .blocks
            .iter()
            .map(|b| (b.total_two_j, b.record_len))
            .collect::<Vec<_>>(),
        [(1, 2), (3, 1)]
    );
    assert_eq!(result.couplings(&result.records[0]).unwrap()[0].two_j, 0);
    assert_eq!(result.couplings(&result.records[1]).unwrap()[0].two_j, 2);
}

#[test]
fn zero_occupations_are_omitted_and_zero_results_are_explicit() {
    let input = request(&[("2s", 0), ("2p", 1)], 3, 3);
    assert_eq!(roundtrip(&input).subshells, ["2p"]);
    for input in [request(&[], 0, 0), request(&[("1s", 2)], 2, 2)] {
        let result = generate_csfs(&input).unwrap();
        assert!(result.records.is_empty());
        let mut bytes = Vec::new();
        assert!(result.write_to(&mut bytes).is_err());
        assert!(bytes.is_empty());
    }
}

#[test]
fn invalid_inputs_and_unsupported_tables_fail() {
    for label in ["0s", "1p", "2s-", "16s", "02p", "+2p", "2p+", "2j"] {
        assert!(label.parse::<Subshell>().is_err(), "accepted {label}");
    }
    for input in [
        request(&[("2p-", 3)], 1, 1),
        request(&[("2s", 1), ("2s", 1)], 0, 0),
        request(&[("1s", 1)], 0, 2),
        request(&[("6h", 3)], 1, 3),
        request(&[("1s", 2)], 4, 0),
    ] {
        assert!(generate_csfs(&input).is_err());
    }
    let mut input = request(&[("1s", 1)], 1, 1);
    input.core_subshells.push("1s".parse().unwrap());
    assert!(generate_csfs(&input).is_err());
}

#[test]
fn custom_coupling_order_is_preserved() {
    let result = roundtrip(&request(&[("3s", 1), ("1s", 1), ("2s", 1)], 1, 1));
    assert_eq!(result.subshells, ["3s", "1s", "2s"]);
}

/// Per-symmetry-block record counts from the unmodified `rcsfgenerate` driven
/// by the registered transcripts.  The `o1` case reproduces the registered
/// baseline `o1_cc1as1.c` byte for byte, so its counts are the file's own.
///
/// Comparing blocks rather than a single total keeps this sensitive to the
/// reference-parity filter of `blanda.f90`: dropping it roughly doubles every
/// block instead of shifting records between them.
/// One registered transcript: its text, the parity every CSF it yields must
/// have, and the `(2J, record count)` pair of each symmetry block.
type RegisteredBlocks = (&'static str, Parity, &'static [(u16, usize)]);

const REGISTERED_BLOCK_COUNTS: [RegisteredBlocks; 2] = [
    (
        include_str!("fixtures/e1_cc1as1.rcsfgenerate"),
        Parity::Even,
        &[
            (0, 18514),
            (2, 51172),
            (4, 75529),
            (6, 86587),
            (8, 86016),
            (10, 75114),
            (12, 59441),
        ],
    ),
    (
        include_str!("fixtures/o1_cc1as1.rcsfgenerate"),
        Parity::Odd,
        &[(5, 42663), (7, 47123)],
    ),
];

#[test]
fn registered_rcsfgenerate_inputs_reproduce_grasp_block_counts() {
    for (input, parity, expected) in REGISTERED_BLOCK_COUNTS {
        let request = ExcitationRequest::from_transcript(input).unwrap();
        let occupations = enumerate_occupations(&request).unwrap();
        let mut counts = BTreeMap::<u16, usize>::new();
        for configuration in &occupations.configurations {
            let generated = generate_csfs(&GenerationRequest {
                core_subshells: occupations.core_subshells.clone(),
                configuration: configuration.occupations.clone(),
                min_two_j: request.min_two_j,
                max_two_j: request.max_two_j,
            })
            .unwrap();
            for record in &generated.records {
                assert_eq!(record.parity, parity, "reference parity is not preserved");
                *counts.entry(record.total_two_j).or_default() += 1;
            }
        }
        assert_eq!(counts.into_iter().collect::<Vec<_>>(), expected);
    }
}

#[test]
fn registered_rcsfgenerate_inputs_parse_and_merge_reference_tasks() {
    for input in [
        include_str!("fixtures/e1_cc1as1.rcsfgenerate"),
        include_str!("fixtures/o1_cc1as1.rcsfgenerate"),
    ] {
        let request = ExcitationRequest::from_transcript(input).unwrap();
        assert_eq!(request.core, 3);
        assert_eq!(request.references.len(), 2);
        assert_eq!(request.max_excitations, 2);
        let occupations = enumerate_occupations(&request).unwrap();
        assert!(!occupations.configurations.is_empty());
        assert!(!occupations.core_subshells.is_empty());
        assert!(!occupations.active_subshells.is_empty());

        // LIKA/TEST merge both consumes equal keys and keeps the descending
        // order, so the merged list is strictly descending and duplicate-free.
        if let Some((index, pair)) = occupations
            .configurations
            .windows(2)
            .enumerate()
            .find(|(_, pair)| pair[0].key() <= pair[1].key())
        {
            panic!(
                "tasks are not strictly descending at {index}: {:?} then {:?}",
                pair[0].key(),
                pair[1].key()
            );
        }

        // Excitations move electrons between shells, never add or remove any.
        let total = |configuration: &EnumeratedConfiguration| {
            configuration
                .occupations
                .iter()
                .map(|entry| u16::from(entry.electrons))
                .sum::<u16>()
        };
        let expected = total(&occupations.configurations[0]);
        assert!(
            occupations
                .configurations
                .iter()
                .all(|configuration| total(configuration) == expected)
        );
    }
}

/// Per-reference record counts for `e1_cc1as1`, each from a single-reference
/// run of the registered `rcsfgenerate` binary.  They exceed the merged
/// 452,373 of [`REGISTERED_BLOCK_COUNTS`] by the 5,839 records whose
/// occupations both references reach.
#[test]
fn registered_rcsfgenerate_references_expand_individually() {
    let request =
        ExcitationRequest::from_transcript(include_str!("fixtures/e1_cc1as1.rcsfgenerate"))
            .unwrap();
    let mut counts = Vec::new();
    for reference in request.references.iter().cloned() {
        let single = ExcitationRequest {
            references: vec![reference],
            ..request.clone()
        };
        let occupations = enumerate_occupations(&single).unwrap();
        counts.push(
            occupations
                .configurations
                .iter()
                .map(|configuration| {
                    generate_csfs(&GenerationRequest {
                        core_subshells: occupations.core_subshells.clone(),
                        configuration: configuration.occupations.clone(),
                        min_two_j: single.min_two_j,
                        max_two_j: single.max_two_j,
                    })
                    .unwrap()
                    .records
                    .len()
                })
                .sum::<usize>(),
        );
    }
    assert_eq!(counts, [198_911, 259_301]);
}

fn closed_configuration(count: usize) -> GenerationRequest {
    let mut input = request(&[], 0, 0);
    for n in 1..=15 {
        for l in 0..n.min(11) {
            for kappa in [l as i8, -(l as i8) - 1] {
                if kappa == 0 {
                    continue;
                }
                let subshell = Subshell::new(n, kappa).unwrap();
                input.configuration.push(SubshellOccupation {
                    subshell,
                    electrons: subshell.capacity(),
                });
                if input.configuration.len() == count {
                    return input;
                }
            }
        }
    }
    unreachable!()
}

#[test]
fn twenty_subshell_limit_is_checked_without_truncation() {
    let result = roundtrip(&closed_configuration(20));
    assert_eq!(result.records.len(), 1);
    assert_eq!(result.occupied(&result.records[0]).unwrap().len(), 20);
    assert!(
        generate_csfs(&closed_configuration(21))
            .unwrap_err()
            .to_string()
            .contains("20 occupied")
    );
}

/// Optional differential test. Point GRASP_SOURCE at a GRASP source checkout.
/// The reference routines are compiled and run in an isolated temporary directory.
#[test]
#[ignore = "requires GRASP_SOURCE and gfortran; see docs/CSF_GENERATION.md"]
fn generated_records_match_unmodified_fortran_gen() {
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::time::{SystemTime, UNIX_EPOCH};

    let source = PathBuf::from(std::env::var_os("GRASP_SOURCE").expect("set GRASP_SOURCE"))
        .join("src/appl/rcsfgenerate90");
    let work = std::env::temp_dir().join(format!(
        "rcsfs-gen-reference-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&work).unwrap();
    let compiler = Command::new("gfortran")
        .current_dir(&work)
        .args(["-O0", "-fcheck=bounds", "-o", "reference"])
        .args(
            [
                "kopp1_I.f90",
                "kopp2_I.f90",
                "kopp1.f90",
                "kopp2.f90",
                "genb.f90",
            ]
            .map(|p| source.join(p)),
        )
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen_reference.f90"))
        .output()
        .unwrap();
    assert!(
        compiler.status.success(),
        "{}",
        String::from_utf8_lossy(&compiler.stderr)
    );
    let mut cases = vec![
        closed_configuration(20),
        request(&[("1s", 1), ("2s", 1), ("3s", 1)], 1, 3),
        request(&[("4f-", 3), ("4f", 4), ("5d-", 1)], 0, 12),
        request(&[("2s", 2), ("2p-", 1), ("2p", 1), ("3s", 1)], 1, 5),
        request(&[("2s", 1), ("5g", 5)], 0, 26),
        request(&[("2p", 2), ("3s", 2), ("3p", 1)], 1, 7),
    ];
    // Every populated JKVANT/SENIOR table, including particle-hole mirrors.
    for l in 0..=10i8 {
        for kappa in [l, -l - 1] {
            if kappa == 0 {
                continue;
            }
            let subshell = Subshell::new((l + 1) as u8, kappa).unwrap();
            for electrons in 1..=subshell.capacity() {
                let population = electrons.min(subshell.capacity() - electrons);
                if subshell.capacity() > 10 && population > 2 {
                    continue;
                }
                cases.push(GenerationRequest {
                    core_subshells: vec![],
                    configuration: vec![SubshellOccupation {
                        subshell,
                        electrons,
                    }],
                    min_two_j: u16::from(electrons % 2),
                    max_two_j: 40 + u16::from(electrons % 2),
                });
            }
        }
    }
    for (index, input) in cases.iter().enumerate() {
        let path = work.join(format!("case-{index}.c"));
        let mut child = Command::new(work.join("reference"))
            .arg(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        writeln!(
            stdin,
            "{} {} {}",
            input.configuration.len(),
            input.min_two_j,
            input.max_two_j
        )
        .unwrap();
        for entry in &input.configuration {
            let orbital = entry.subshell;
            let branch = u8::from(orbital.l() > 0 && orbital.kappa() < 0);
            writeln!(
                stdin,
                "{} {} {} {}",
                orbital.n(),
                orbital.l(),
                branch,
                entry.electrons
            )
            .unwrap();
        }
        drop(stdin);
        let reference = child.wait_with_output().unwrap();
        assert!(
            reference.status.success(),
            "case {index}: {}",
            String::from_utf8_lossy(&reference.stderr)
        );
        let raw = fs::read_to_string(&path).unwrap();
        let lines = raw.lines().map(str::trim_end).collect::<Vec<_>>();
        assert!(lines.len().is_multiple_of(3));
        let mut expected = lines
            .chunks_exact(3)
            .map(|r| r.to_vec())
            .collect::<Vec<_>>();
        // rcsfblock groups by ascending J without changing order within a block.
        expected.sort_by_key(|record| {
            let final_line = record[2];
            let value = final_line[..final_line.len() - 1]
                .split_whitespace()
                .last()
                .unwrap();
            value.strip_suffix("/2").map_or_else(
                || value.parse::<u16>().unwrap() * 2,
                |n| n.parse::<u16>().unwrap(),
            )
        });
        let generated = generate_csfs(input).unwrap();
        let mut bytes = Vec::new();
        generated.write_to(&mut bytes).unwrap();
        let actual = String::from_utf8(bytes).unwrap();
        let actual = actual
            .lines()
            .skip(5)
            .filter(|line| *line != " *")
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            expected.into_iter().flatten().collect::<Vec<_>>(),
            "case {index}: {input:?}"
        );
    }
    println!(
        "{} configurations match unmodified Fortran GEN",
        cases.len()
    );
    fs::remove_dir_all(work).unwrap();
}

#[test]
fn transcript_rejects_unimplemented_branches_and_trailing_lists() {
    let input = include_str!("fixtures/o1_cc1as1.rcsfgenerate");
    for selector in ["r", "s", "u", "e"] {
        let changed = input.replace("* ! Orbital order", &format!("{selector} ! Orbital order"));
        assert!(
            ExcitationRequest::from_transcript(&changed)
                .unwrap_err()
                .to_string()
                .contains("orbital order")
        );
    }
    for answer in ["y", "Y"] {
        let changed = input.replace("\nn\n", &format!("\n{answer}\n"));
        assert!(
            ExcitationRequest::from_transcript(&changed)
                .unwrap_err()
                .to_string()
                .contains("multiple lists")
        );
    }
    assert!(
        ExcitationRequest::from_transcript(&input.replace("\nn\nEOF", "\nn\nextra\nEOF")).is_err()
    );
    assert!(ExcitationRequest::from_transcript(&input.replace("\nn\nEOF", "")).is_err());
}

#[test]
fn transcript_expansion_mode_is_rejected_explicitly() {
    let input = include_str!("fixtures/o1_cc1as1.rcsfgenerate");
    let changed = input.replace("* ! Orbital order", "e ! Orbital order");
    let error = ExcitationRequest::from_transcript(&changed).unwrap_err();
    assert!(error.to_string().contains("orbital order"));
}

/// External, complete direct outputs; never substitute truncated fixtures.
#[test]
#[ignore = "requires RCSFS_BASELINE_DIR containing the registered full CSF files"]
fn registered_inputs_match_every_baseline_record_in_order() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("RCSFS_BASELINE_DIR")
            .expect("set RCSFS_BASELINE_DIR to the registered baseline directory"),
    );
    for ((input, _, expected_counts), name) in REGISTERED_BLOCK_COUNTS
        .into_iter()
        .zip(["e1_cc1as1.c", "o1_cc1as1.c"])
    {
        let baseline = CompleteCsfFile::parse_path(&directory.join(name)).unwrap();
        assert_eq!(
            baseline
                .blocks
                .iter()
                .map(|block| (block.total_two_j, block.record_len as usize))
                .collect::<Vec<_>>(),
            expected_counts,
            "{name}: baseline must be the full direct rcsfgenerate output"
        );
        let request = ExcitationRequest::from_transcript(input).unwrap();
        let occupations = enumerate_occupations(&request).unwrap();
        let mut positions = BTreeMap::new();
        for block in &baseline.blocks {
            positions.insert(block.total_two_j, block.record_start as usize);
        }
        for (task, configuration) in occupations.configurations.iter().enumerate() {
            let generated = generate_csfs(&GenerationRequest {
                core_subshells: occupations.core_subshells.clone(),
                configuration: configuration.occupations.clone(),
                min_two_j: request.min_two_j,
                max_two_j: request.max_two_j,
            })
            .unwrap();
            for record in &generated.records {
                let position = positions.get_mut(&record.total_two_j).unwrap();
                let expected = &baseline.records[*position];
                let context = format!(
                    "{name}: task {task}, record {}, 2J={}",
                    *position, record.total_two_j
                );
                assert_eq!(
                    (record.total_two_j, record.parity),
                    (expected.total_two_j, expected.parity),
                    "{context}"
                );
                let occupied = |file: &CompleteCsfFile, record| {
                    file.occupied(record)
                        .unwrap()
                        .iter()
                        .map(|shell| {
                            (
                                file.subshells[usize::from(shell.subshell_index)].clone(),
                                shell.occupation,
                                shell.state,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    occupied(&generated, record),
                    occupied(&baseline, expected),
                    "{context}: occupation/state"
                );
                assert_eq!(
                    generated.couplings(record).unwrap(),
                    baseline.couplings(expected).unwrap(),
                    "{context}: coupling chain"
                );
                *position += 1;
            }
        }
        for block in &baseline.blocks {
            assert_eq!(
                positions[&block.total_two_j],
                (block.record_start + block.record_len) as usize,
                "{name}: incomplete block"
            );
        }
        eprintln!("{name}: {} records match in order", baseline.records.len());
    }
}

fn excitation_transcript(core: u8, reference: &str, active: &str, excitations: i16) -> String {
    format!(
        "* ! Orbital order\n{core} ! Selected core\n{reference}\n*\n{active}\n0,12 ! Lower and higher 2*J\n{excitations} ! Number of excitations\nn\n"
    )
}

#[test]
fn normalized_inputs_preserve_closed_cores_and_pair_only_missing_shells() {
    use _rcsfs::csf_generation::{OccupationMode, Orbital};
    for (core, reference, active, electrons) in [
        (5, "4f(2,*)", "4f", 54u16),
        (6, "5f(2,*)", "5f", 86),
        (6, "5g(2,*)", "5g", 86),
    ] {
        let request =
            ExcitationRequest::from_transcript(&excitation_transcript(core, reference, active, 0))
                .unwrap();
        assert_eq!(request.core, core - 1);
        let occupations = enumerate_occupations(&request).unwrap();
        assert_eq!(
            occupations
                .core_subshells
                .iter()
                .map(|shell| u16::from(shell.capacity()))
                .sum::<u16>(),
            electrons
        );
        assert!(
            !occupations
                .active_subshells
                .iter()
                .any(|shell| occupations.core_subshells.contains(shell))
        );
    }
    assert!("3f".parse::<Orbital>().is_err());
    let request = ExcitationRequest::from_transcript(&excitation_transcript(
        0,
        "1s(2,*)2s(0,*)",
        "2s,2p",
        -2,
    ))
    .unwrap();
    assert_eq!(request.max_excitations, 2);
    let mode = |label: &str| {
        request.references[0]
            .shells
            .iter()
            .find(|shell| shell.orbital == label.parse().unwrap())
            .unwrap()
            .mode
    };
    assert_eq!(mode("2s"), OccupationMode::Active);
    assert_eq!(mode("2p"), OccupationMode::Double);
    let occupations = enumerate_occupations(&request).unwrap();
    assert!(occupations.configurations.iter().any(|configuration| {
        configuration
            .occupations
            .iter()
            .any(|shell| shell.subshell.to_string() == "2s" && shell.electrons == 1)
    }));
    for configuration in &occupations.configurations {
        let p_electrons: u8 = configuration
            .occupations
            .iter()
            .filter(|shell| shell.subshell.l() == 1)
            .map(|shell| shell.electrons)
            .sum();
        assert!(p_electrons.is_multiple_of(2));
    }
    for reference in ["1s(2,d)", "1s(1,c)"] {
        let request =
            ExcitationRequest::from_transcript(&excitation_transcript(0, reference, "2s", 2))
                .unwrap();
        assert!(enumerate_occupations(&request).is_err());
    }
    assert!(
        ExcitationRequest::from_transcript(&excitation_transcript(0, "1s(2,*)", "2s", i16::MIN))
            .is_err()
    );
}

/// Live wrapper-level oracle, in separate directories to isolate fixed filenames.
#[test]
#[ignore = "requires GRASP_RCSFGENERATE pointing to the registered executable"]
fn input_rewrites_match_unmodified_rcsfgenerate() {
    use std::{
        fs,
        process::{Command, Stdio},
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };
    let executable =
        fs::canonicalize(std::env::var_os("GRASP_RCSFGENERATE").expect("set GRASP_RCSFGENERATE"))
            .unwrap();
    let directory = std::env::temp_dir().join(format!(
        "rcsfs-wrapper-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&directory).unwrap();
    let cases = [
        (0, "1s(2,*)", "2s,2p", -2),
        (0, "1s(2,*)2s(0,*)", "2s,2p", -2),
        (0, "1s(2,*)", "2s,2p", -1),
        (0, "1s(2,*)", "2s,2p", 2),
        (1, "2s(2,*)", "3s,3p", -2),
        (2, "3s(2,*)", "4s,4p", -2),
        (3, "3d(2,*)", "4s,4p,4d", -2),
        (4, "4d(2,*)", "5s,5p,5d", -2),
        (5, "4f(2,*)", "4f", 0),
        (5, "4f(2,*)", "5s,5p,5d,5f", -2),
        (5, "4f(2,*)\n5d(2,*)", "5s,5p,5d,5f", 0),
        (6, "5f(2,*)", "5f", 0),
        (6, "5g(2,*)", "5g", 0),
        (6, "5f(2,*)", "5f,5g", -2),
        (6, "7s(2,*)", "7s,6d", -2),
        (6, "7s(2,*)", "7s", 0),
        (6, "7s(2,*)", "7s,6g", -2),
        (6, "5f(2,*)", "6s,5f", -2),
    ];
    for (case, (core, reference, active, excitations)) in cases.into_iter().enumerate() {
        let work = directory.join(case.to_string());
        fs::create_dir(&work).unwrap();
        let transcript = excitation_transcript(core, reference, active, excitations);
        let raw = transcript
            .lines()
            .map(|line| line.split('!').next().unwrap().trim())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(work.join("input"), raw).unwrap();
        let mut child = Command::new(&executable)
            .current_dir(&work)
            .stdin(fs::File::open(work.join("input")).unwrap())
            .stdout(fs::File::create(work.join("stdout")).unwrap())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "case {case}: {work:?}");
                break;
            }
            if start.elapsed() > Duration::from_secs(20) {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("Fortran timeout in {work:?}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let baseline = CompleteCsfFile::parse_path(&work.join("rcsf.out")).unwrap();
        let request = ExcitationRequest::from_transcript(&transcript).unwrap();
        // The actual Fortran log already contains patched active limits.
        let logged = fs::read_to_string(work.join("rcsfgenerate.log")).unwrap();
        assert_eq!(
            request,
            ExcitationRequest::from_transcript(&logged).unwrap(),
            "case {case}: log replay"
        );
        let occupations = enumerate_occupations(&request).unwrap();
        assert_eq!(
            occupations
                .core_subshells
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            baseline.header_lines[1]
                .split_whitespace()
                .collect::<Vec<_>>(),
            "case {case}: closed core"
        );
        let mut actual = BTreeMap::<u16, Vec<String>>::new();
        for configuration in &occupations.configurations {
            let generated = generate_csfs(&GenerationRequest {
                core_subshells: occupations.core_subshells.clone(),
                configuration: configuration.occupations.clone(),
                min_two_j: request.min_two_j,
                max_two_j: request.max_two_j,
            })
            .unwrap();
            if generated.records.is_empty() {
                continue;
            }
            let mut text = Vec::new();
            generated.write_to(&mut text).unwrap();
            let text = String::from_utf8(text).unwrap();
            let mut lines = text.lines().skip(5).filter(|line| !line.starts_with(" *"));
            for record in &generated.records {
                let record_lines = (0..3)
                    .map(|_| lines.next().unwrap())
                    .collect::<Vec<_>>()
                    .join("\n");
                actual
                    .entry(record.total_two_j)
                    .or_default()
                    .push(record_lines);
            }
        }
        let expected_text = fs::read_to_string(work.join("rcsf.out")).unwrap();
        let expected_lines = expected_text
            .lines()
            .skip(5)
            .filter(|line| !line.starts_with(" *"))
            .collect::<Vec<_>>();
        let expected = expected_lines
            .chunks_exact(3)
            .map(|lines| lines.join("\n"))
            .collect::<Vec<_>>();
        let actual = actual.into_values().flatten().collect::<Vec<_>>();
        assert_eq!(
            actual.len(),
            baseline.records.len(),
            "case {case}: count; {work:?}"
        );
        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a, b, "case {case}: record {index}; {work:?}");
        }
        eprintln!(
            "case {case}: core={core}, active={active}, excitations={excitations}: {} records match",
            actual.len()
        );
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn high_l_excitation_occupations_follow_fortran_four_electron_limit() {
    let request = ExcitationRequest::from_transcript(&excitation_transcript(
        0,
        "1s(2,*)2s(2,*)3s(2,*)",
        "6h",
        6,
    ))
    .unwrap();
    let result = enumerate_occupations(&request).unwrap();
    let populations = result
        .configurations
        .iter()
        .map(|configuration| {
            configuration
                .occupations
                .iter()
                .filter(|shell| shell.subshell.l() == 5)
                .map(|shell| shell.electrons)
                .sum::<u8>()
        })
        .collect::<Vec<_>>();
    assert!(populations.contains(&4));
    assert!(populations.iter().all(|&count| count <= 4));
}

#[test]
fn parallel_state_prefixes_preserve_records_and_order() {
    // Four d_5/2 shells each have three states: 81 combinations force splitting.
    for max_two_j in [0, 4] {
        let input = request(&[("3d", 3), ("4d", 3), ("5d", 3), ("6d", 3)], 0, max_two_j);
        let serial = generate_csfs(&input).unwrap();
        assert!(!serial.records.is_empty());
        for threads in [1, 2, 4] {
            for _ in 0..2 {
                let parallel =
                    generate_csfs_parallel(std::slice::from_ref(&input), Some(threads)).unwrap();
                assert_eq!(parallel[0], serial);
            }
        }
    }
}
