//! P6a: exhaustive counterexample search for V2 record uniqueness.
//!
//! The P6a phase asks one question with an architectural consequence: can the
//! internal generation path emit the same V2 record twice? The exact
//! de-duplication chain (root buckets, recursive splits, survivor bitsets) only
//! earns its cost if it can fire; if the internal path is globally unique, that
//! chain is provably dead work for it and must not be optimized (P2) or
//! parallelized (P3) on the strength of a duplicate count that is always zero.
//!
//! The answer is *searched for*, not assumed. Every check below drives the
//! production generator and encoder over a bounded system and, when it finds a
//! duplicate, reports both records that produced it — so a later change that
//! reintroduces one leaves a counterexample behind instead of a bare assertion
//! failure. The companion argument is `docs/V2_GENERATION_UNIQUENESS.md`; a test
//! that passes is not a proof, and this file is only the exhaustive part.
//!
//! What is compared, for one family of configurations that shares a run-level
//! Peel table:
//!
//! * raw generation, with no de-duplication at all;
//! * the local layer, de-duplicating inside each configuration;
//! * the global layer, de-duplicating across the whole run.
//!
//! All three must agree record for record, in order. The control families
//! assert that the audit *does* find the duplicates they contain, so an empty
//! duplicate list means "none exist here", not "the check is vacuous".

use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use _rcsfs::complete_csf::{CompleteCsfFile, CsfRecord, OccupiedSubshell};
use _rcsfs::csf_generation::{
    EnumeratedOccupations, ExcitationRequest, GenerationRequest, Subshell, SubshellOccupation,
    enumerate_occupations, generate_csfs,
};
use _rcsfs::descriptor_schema::{DescriptorLayout, DescriptorVersion, MISSING};
use _rcsfs::descriptor_v2::{decode_v2_into, encode_v2};
use anyhow::{Context, Result, ensure};

const ORBITAL_LETTERS: [char; 7] = ['s', 'p', 'd', 'f', 'g', 'h', 'i'];

/// The two relativistic partners of a nonrelativistic orbital `n(l+1)`,
/// spelled the way `Subshell`'s `FromStr` accepts them.
fn partners(l: u8) -> Vec<Subshell> {
    let letter = ORBITAL_LETTERS[usize::from(l)];
    let n = l + 1;
    if l == 0 {
        return vec![format!("{n}{letter}").parse().expect("canonical s label")];
    }
    vec![
        format!("{n}{letter}-")
            .parse()
            .expect("canonical kappa>0 label"),
        format!("{n}{letter}")
            .parse()
            .expect("canonical kappa<0 label"),
    ]
}

fn shell(label: &str) -> Subshell {
    Subshell::from_str(label).expect("test Peel label")
}

fn request(entries: &[(Subshell, u8)], min_two_j: u16, max_two_j: u16) -> GenerationRequest {
    // Coupling order is Peel order. The excitation enumerator emits
    // configurations that way, and the disk writer rejects a record whose
    // occupied subshells are not increasing in the run-level Peel table, so a
    // family built in any other order would not be a production input.
    let mut entries = entries.to_vec();
    entries.sort_by_key(|&(subshell, _)| (subshell.n(), subshell.l(), subshell.kappa() < 0));
    GenerationRequest {
        core_subshells: Vec::new(),
        configuration: entries
            .into_iter()
            .map(|(subshell, electrons)| SubshellOccupation {
                subshell,
                electrons,
            })
            .collect(),
        min_two_j,
        max_two_j,
    }
}

/// A 2J range that reaches every record the configuration can produce.
///
/// Four times the total subshell capacity is above the largest total `2J` any
/// supported state table can reach — the widest single state is 25, for a
/// subshell of capacity 10 — and the bound is clamped to GRASP's printable
/// total-`2J` field.
fn full_range(entries: &[(Subshell, u8)]) -> (u16, u16) {
    let electrons: u16 = entries
        .iter()
        .map(|&(_, electrons)| u16::from(electrons))
        .sum();
    let capacity: u16 = entries
        .iter()
        .map(|&(subshell, _)| u16::from(subshell.capacity()))
        .sum();
    let ceiling = if electrons.is_multiple_of(2) { 98 } else { 99 };
    let mut max = (4 * capacity).min(ceiling);
    if max % 2 != electrons % 2 {
        max -= 1;
    }
    (electrons % 2, max)
}

/// The run-level Peel table: the union of the subshells the family occupies,
/// ordered exactly as `precompute_peel_subshells` orders a production run.
fn peel_table(files: &[CompleteCsfFile]) -> Vec<Subshell> {
    let mut used: Vec<Subshell> = Vec::new();
    for file in files {
        for label in &file.subshells {
            let subshell = Subshell::from_str(label).expect("generated label is canonical");
            if !used.contains(&subshell) {
                used.push(subshell);
            }
        }
    }
    used.sort_by_key(|shell| (shell.n(), shell.l(), shell.kappa() < 0));
    used
}

/// The run-level Peel table and one configuration's place in it.
///
/// The production writer encodes a configuration's local indices and remaps
/// them through this same correspondence; keeping it per configuration makes a
/// cross-configuration comparison use one frame without re-deriving the
/// mapping for every record.
struct PeelFrame {
    layout: DescriptorLayout,
    /// Run-level index of each of the configuration's local Peel subshells.
    mapping: Vec<usize>,
    local_layout: DescriptorLayout,
    local_row: Vec<i32>,
    row: Vec<i32>,
}

impl PeelFrame {
    fn new(
        file: &CompleteCsfFile,
        peel: &[Subshell],
        index_of: &HashMap<Subshell, usize>,
    ) -> Result<Self> {
        let mapping = file
            .subshells
            .iter()
            .map(|label| {
                let subshell = Subshell::from_str(label)
                    .with_context(|| format!("generated Peel label {label:?} is not a subshell"))?;
                index_of
                    .get(&subshell)
                    .copied()
                    .with_context(|| format!("{label} is absent from the run-level Peel table"))
            })
            .collect::<Result<Vec<_>>>()?;
        let layout = DescriptorLayout::new(DescriptorVersion::V2, peel.len());
        let local_layout = DescriptorLayout::new(DescriptorVersion::V2, file.subshells.len());
        Ok(Self {
            row: vec![0i32; layout.row_len()],
            local_row: vec![0i32; local_layout.row_len()],
            layout,
            mapping,
            local_layout,
        })
    }

    /// Encode one record into the run's Peel coordinates.
    fn encode(&mut self, file: &CompleteCsfFile, record: &CsfRecord) -> Result<&[i32]> {
        encode_v2(file, record, &mut self.local_row)?;
        self.row.fill(MISSING);
        for index in 0..self.layout.subshell_count() {
            self.row[self.layout.slot(index) + self.layout.n_offset()] = 0;
        }
        let channels = self.local_layout.channels_per_subshell();
        for (local_index, &global_index) in self.mapping.iter().enumerate() {
            for channel in 0..channels {
                self.row[self.layout.slot(global_index) + channel] =
                    self.local_row[self.local_layout.slot(local_index) + channel];
            }
        }
        for (local_column, global_column) in [
            (
                self.local_layout.total_two_j_index(),
                self.layout.total_two_j_index(),
            ),
            (self.local_layout.parity_index(), self.layout.parity_index()),
        ] {
            self.row[global_column.expect("V2 has global columns")] =
                self.local_row[local_column.expect("V2 has global columns")];
        }
        Ok(&self.row)
    }

    /// The record's occupied subshells in run-level indices.
    fn occupied(
        &self,
        file: &CompleteCsfFile,
        record: &CsfRecord,
    ) -> Result<Vec<OccupiedSubshell>> {
        file.occupied(record)?
            .iter()
            .map(|occupied| {
                let local = usize::from(occupied.subshell_index);
                Ok(OccupiedSubshell {
                    subshell_index: u16::try_from(*self.mapping.get(local).with_context(
                        || format!("local subshell index {local} exceeds the Peel table"),
                    )?)?,
                    occupation: occupied.occupation,
                    state: occupied.state,
                })
            })
            .collect()
    }
}

/// One generated record, in the run's Peel coordinates.
struct Sighted {
    row: Vec<i32>,
    text: String,
    origin: String,
}

/// A pair of records that the layer under test would have to collapse.
struct Duplicate {
    first: String,
    second: String,
    shared: String,
}

impl Duplicate {
    fn describe(&self) -> String {
        format!(
            "{} and {} are the same record: {}",
            self.first, self.second, self.shared
        )
    }
}

/// What auditing one family found.
struct Audit {
    records: usize,
    configurations: usize,
    peel: Vec<Subshell>,
    row_duplicates: Vec<Duplicate>,
    text_duplicates: Vec<Duplicate>,
    decode_failures: Vec<String>,
    /// Survivor indices when equal rows collapse across the whole run.
    global_survivors: Vec<usize>,
    /// Survivor indices when equal rows collapse inside each configuration.
    local_survivors: Vec<usize>,
}

impl Audit {
    fn report(&self, family: &str) {
        eprintln!(
            "{family}: {} records over {} configurations, {} Peel subshells, \
             {} row duplicates, {} text duplicates, {} decode failures",
            self.records,
            self.configurations,
            self.peel.len(),
            self.row_duplicates.len(),
            self.text_duplicates.len(),
            self.decode_failures.len(),
        );
    }

    /// The uniqueness conclusion for this family, phrased so a failure prints
    /// the counterexample rather than only a count.
    fn assert_unique(&self, family: &str) {
        self.report(family);
        assert!(
            self.row_duplicates.is_empty(),
            "{family} emitted duplicate rows:\n{}",
            self.row_duplicates
                .iter()
                .map(Duplicate::describe)
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert!(
            self.text_duplicates.is_empty(),
            "{family} emitted duplicate CSF text:\n{}",
            self.text_duplicates
                .iter()
                .map(Duplicate::describe)
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert!(
            self.decode_failures.is_empty(),
            "{family} decoded to something other than the record it was built from:\n{}",
            self.decode_failures.join("\n")
        );
        // With no duplicates there is nothing to remove, so all three layers
        // must publish the raw sequence itself: same records, same order.
        let identity = (0..self.records).collect::<Vec<_>>();
        assert_eq!(
            self.global_survivors, identity,
            "{family}: the global layer changed the published sequence"
        );
        assert_eq!(
            self.local_survivors, identity,
            "{family}: the local layer changed the published sequence"
        );
    }
}

/// Generate a family with the production generator and audit it.
fn audit(requests: &[GenerationRequest]) -> Result<Audit> {
    let files = requests
        .iter()
        .enumerate()
        .map(|(index, request)| {
            generate_csfs(request).with_context(|| format!("configuration {index} failed"))
        })
        .collect::<Result<Vec<_>>>()?;
    let peel = peel_table(&files);
    ensure!(!peel.is_empty(), "the family occupies no subshell");
    let index_of = peel
        .iter()
        .enumerate()
        .map(|(index, &subshell)| (subshell, index))
        .collect::<HashMap<_, _>>();

    let mut sighted = Vec::new();
    let mut decode_failures = Vec::new();
    for (configuration, file) in files.iter().enumerate() {
        let mut frame = PeelFrame::new(file, &peel, &index_of)?;
        for (index, record) in file.records.iter().enumerate() {
            let origin = format!("configuration {configuration} record {index}");
            let expected = frame.occupied(file, record)?;
            // The production writer validates this before encoding; a family
            // that violates it is not a production-shaped input and would
            // otherwise be compared in a coordinate system of its own.
            ensure!(
                expected
                    .windows(2)
                    .all(|pair| pair[0].subshell_index < pair[1].subshell_index),
                "{origin}: coupling order is not the run-level Peel order ({expected:?})"
            );
            let row = frame.encode(file, record)?.to_vec();
            let mut text = Vec::new();
            file.write_record_to(record, &mut text)?;
            let text = String::from_utf8(text).context("formatted CSF is not UTF-8")?;

            let mut occupied = Vec::new();
            let mut couplings = Vec::new();
            let (total_two_j, parity) =
                decode_v2_into(&row, frame.layout, &mut occupied, &mut couplings)?;
            if occupied != expected
                || couplings != file.couplings(record)?
                || total_two_j != record.total_two_j
                || parity != record.parity
            {
                decode_failures.push(format!(
                    "{origin}: row {row:?} decoded to {occupied:?} {couplings:?} \
                     {total_two_j} {parity:?}, not to {expected:?}"
                ));
            }
            sighted.push(Sighted { row, text, origin });
        }
    }

    let row_duplicates = duplicates_by(&sighted, |sighted| &sighted.row);
    let text_duplicates = duplicates_by(&sighted, |sighted| &sighted.text);
    Ok(Audit {
        records: sighted.len(),
        configurations: files.len(),
        peel,
        row_duplicates,
        text_duplicates,
        decode_failures,
        global_survivors: survivors(&sighted, 0..sighted.len()),
        local_survivors: local_survivors(&sighted, &files),
    })
}

/// Every pair of records that share a key, each reported once.
fn duplicates_by<K: Eq + std::hash::Hash + Clone + std::fmt::Debug>(
    sighted: &[Sighted],
    key: impl Fn(&Sighted) -> &K,
) -> Vec<Duplicate> {
    let mut seen: HashMap<K, usize> = HashMap::new();
    let mut duplicates = Vec::new();
    for (index, entry) in sighted.iter().enumerate() {
        match seen.get(key(entry)) {
            Some(&first) => duplicates.push(Duplicate {
                first: sighted[first].origin.clone(),
                second: entry.origin.clone(),
                shared: format!("{:?}", key(entry)),
            }),
            None => {
                seen.insert(key(entry).clone(), index);
            }
        }
    }
    duplicates
}

/// Indices of the first occurrence of every distinct row in `range`.
fn survivors(sighted: &[Sighted], range: std::ops::Range<usize>) -> Vec<usize> {
    let mut seen = HashSet::new();
    range
        .filter(|&index| seen.insert(&sighted[index].row))
        .collect()
}

/// The same, restricted to each configuration's own records.
fn local_survivors(sighted: &[Sighted], files: &[CompleteCsfFile]) -> Vec<usize> {
    let mut start = 0;
    let mut survivors = Vec::new();
    for file in files {
        let end = start + file.records.len();
        survivors.extend(self::survivors(sighted, start..end));
        start = end;
    }
    survivors
}

/// The registered multi-reference fixture. Its *enumeration* is cheap and is
/// used for the merge checks below; its 452,373 CSFs are audited end to end by
/// `tests/p6a_disk_uniqueness_test.py`, in the release-built extension, rather
/// than record by record in a debug build here.
const REGISTERED: &str = include_str!("fixtures/e1_cc1as1.rcsfgenerate");

/// A small two-reference transcript, cheap enough to generate completely: both
/// references hold eight electrons and differ in where they sit, so their
/// enumerations overlap and the merge has something to collapse.
const SMALL_MULTI_REFERENCE: &str = "\
rcsfgenerate<< EOF
 * ! Orbital order
0
2s(2,i)2p(5,i)3p(1,i)
2s(2,i)2p(4,i)3p(2,i)

3s,3p,3d
0,4
           1  ! Number of excitations
n
EOF
";

fn enumerated_requests(
    occupations: &EnumeratedOccupations,
    min_two_j: u16,
    max_two_j: u16,
) -> Vec<GenerationRequest> {
    occupations
        .configurations
        .iter()
        .map(|configuration| GenerationRequest {
            core_subshells: occupations.core_subshells.clone(),
            configuration: configuration.occupations.clone(),
            min_two_j,
            max_two_j,
        })
        .collect()
}

/// Every supported subshell state table, one configuration each, over every 2J
/// target the table can reach.
///
/// This is the intra-configuration claim at its smallest: a single occupied
/// subshell has no coupling chain, so the only way to repeat a row is for its
/// state table to repeat a `(2J, seniority)` pair. Covering every reachable
/// `2j_max` therefore covers every table GRASP can hand the generator.
#[test]
fn every_supported_state_table_emits_distinct_rows() {
    let mut requests = Vec::new();
    let mut two_j_maxima = HashSet::new();
    let mut refused = 0usize;
    for l in 0..=6u8 {
        for subshell in partners(l) {
            for electrons in 1..=subshell.capacity() {
                let entries = [(subshell, electrons)];
                let (min, max) = full_range(&entries);
                let candidate = request(&entries, min, max);
                match generate_csfs(&candidate) {
                    Ok(_) => {
                        two_j_maxima.insert(subshell.capacity() - 1);
                        requests.push(candidate);
                    }
                    Err(error) => {
                        // GRASP defines no table for this occupation, so the
                        // generator refuses it and it is outside the claim. Any
                        // other refusal would be a real loss of coverage.
                        assert!(
                            error.to_string().contains("unsupported occupation"),
                            "{subshell}({electrons}) failed for an unexpected reason: {error}"
                        );
                        refused += 1;
                    }
                }
            }
        }
    }
    // Every `2j_max` this family can reach must be covered, and the refused
    // occupations must stay refused: a table that disappears shows up here
    // rather than as a quietly smaller family.
    assert_eq!(
        two_j_maxima,
        HashSet::from([1, 3, 5, 7, 9, 11, 13]),
        "the family lost or gained a 2j_max"
    );
    assert!(
        refused >= 20,
        "only {refused} occupations were refused; the family no longer covers them"
    );
    let audit = audit(&requests).unwrap();
    audit.assert_unique("single-subshell state tables");
    assert!(
        audit.records >= 100,
        "the state-table family lost records: {}",
        audit.records
    );
}

/// Every subset of `shells` holding between `low` and `high` members.
fn subsets(shells: &[Subshell], low: usize, high: usize) -> Vec<Vec<Subshell>> {
    let mut result = Vec::new();
    for mask in 0u32..(1 << shells.len()) {
        let members = (0..shells.len())
            .filter(|index| mask & (1 << index) != 0)
            .map(|index| shells[index])
            .collect::<Vec<_>>();
        if (low..=high).contains(&members.len()) {
            result.push(members);
        }
    }
    result
}

/// Coupling chains over every subset of several open subshells, including
/// tables that carry duplicate `2J` values with different seniority.
#[test]
fn coupling_chains_emit_distinct_rows() {
    let shells = ["2s", "2p-", "4f-", "4f", "5g-", "5g", "3d-", "3d"].map(shell);
    let mut requests = Vec::new();
    for chosen in subsets(&shells, 1, 4) {
        for electrons in 1..=2u8 {
            let entries = chosen
                .iter()
                .map(|&subshell| (subshell, electrons.min(subshell.capacity())))
                .collect::<Vec<_>>();
            let (min, max) = full_range(&entries);
            requests.push(request(&entries, min, max));
        }
    }
    let audit = audit(&requests).unwrap();
    audit.assert_unique("coupling chains");
    assert!(
        audit.records >= 20_000,
        "the coupling-chain family lost records: {}",
        audit.records
    );
}

/// The `FIRST` flag cases: a filled leading subshell suppresses the first
/// interior coupling, and a subshell whose selected state is `2J = 0` prints
/// nothing at all. Both hide couplings from the row, so both are where an
/// injectivity argument could fail. The family walks every occupation of a
/// leading `2s` over a pair of open shells, so the flag is met both set and
/// clear with the same coupling chains behind it.
#[test]
fn hidden_couplings_stay_distinguishable() {
    let leading = shell("2s");
    let lower = shell("4f-");
    let upper = shell("4f");
    let mut requests = Vec::new();
    for filled in 1..=leading.capacity() {
        for second in 1..=lower.capacity() {
            for third in 1..=upper.capacity() {
                let entries = [(leading, filled), (lower, second), (upper, third)];
                let (min, max) = full_range(&entries);
                requests.push(request(&entries, min, max));
            }
        }
    }
    // Longer chains, where the flag can clear before the last printable
    // boundary and the interior states of a filled shell print nothing.
    let long: [&[(&str, u8)]; 3] = [
        &[("2s", 2), ("2p-", 2), ("2p", 4), ("5g", 2)],
        &[("2s", 2), ("3d", 2), ("3d-", 1), ("5g", 2)],
        &[("2p-", 2), ("2p", 4), ("4f", 4), ("5g", 3), ("5g-", 1)],
    ];
    for case in long {
        let entries = case
            .iter()
            .map(|&(label, electrons)| (shell(label), electrons))
            .collect::<Vec<_>>();
        let (min, max) = full_range(&entries);
        requests.push(request(&entries, min, max));
    }
    let audit = audit(&requests).unwrap();
    audit.assert_unique("hidden couplings");
    assert!(
        audit.records >= 5_000,
        "the hidden-coupling family lost records: {}",
        audit.records
    );
}

/// A bounded multi-reference run, taken through the real excitation
/// enumerator: the claim is about what a transcript actually generates.
#[test]
fn a_multi_reference_run_is_globally_unique() {
    let request =
        ExcitationRequest::from_transcript(SMALL_MULTI_REFERENCE).expect("test transcript");
    let occupations = enumerate_occupations(&request).expect("enumeration");
    let audit = audit(&enumerated_requests(
        &occupations,
        request.min_two_j,
        request.max_two_j,
    ))
    .unwrap();
    audit.assert_unique("multi-reference run");
    assert!(
        audit.configurations >= 4,
        "the multi-reference family lost configurations: {}",
        audit.configurations
    );
    assert!(
        audit.records >= 20,
        "the multi-reference family lost records: {}",
        audit.records
    );
}

/// The merged multi-reference list is the union of what each reference
/// enumerates on its own: merging may drop a configuration the two references
/// produce twice, and must never drop one only one of them produces.
///
/// Enumeration alone, so it runs on the full registered fixture: 1,374
/// configurations and the overlap between its two references.
#[test]
fn the_multi_reference_merge_is_the_union_of_its_references() {
    let request = ExcitationRequest::from_transcript(REGISTERED).expect("test transcript");
    assert!(
        request.references.len() > 1,
        "the fixture must have references"
    );
    let merged = enumerate_occupations(&request).expect("merged enumeration");
    let mut union = HashSet::new();
    let mut per_reference_total = 0usize;
    for reference in request.references.iter().cloned() {
        let single = ExcitationRequest {
            references: vec![reference],
            ..request.clone()
        };
        let occupations = enumerate_occupations(&single).expect("single-reference enumeration");
        per_reference_total += occupations.configurations.len();
        for configuration in &occupations.configurations {
            union.insert(configuration.key().to_vec());
        }
    }
    let merged_keys = merged
        .configurations
        .iter()
        .map(|configuration| configuration.key().to_vec())
        .collect::<HashSet<_>>();
    assert_eq!(
        merged_keys, union,
        "merging references changed which configurations exist"
    );
    assert!(
        merged_keys.len() < per_reference_total,
        "the references produced no configuration in common, so this input does not \
         test the merge at all: {} merged from {} enumerated",
        merged_keys.len(),
        per_reference_total
    );
}

/// The enumeration's merge key: one `(occupation, upper, lower)` triple per slot.
type MergeKey = Vec<(u8, u8, u8)>;

/// An occupation vector in printed form, because `SubshellOccupation` is not
/// hashable — and because a reader sees the configuration this way.
type PrintedOccupation = Vec<(String, u8)>;

/// The merge key and the occupation vector must determine each other: equal
/// keys must mean equal configurations (so collapsing them cannot lose a CSF),
/// and distinct keys must mean distinct occupations (so two configurations can
/// never produce the same occupation columns).
#[test]
fn the_enumeration_key_and_the_occupation_vector_agree() {
    let request = ExcitationRequest::from_transcript(REGISTERED).expect("test transcript");
    let occupations = enumerate_occupations(&request).expect("enumeration");
    assert!(
        occupations.configurations.len() >= 4,
        "the bounded family must have several configurations"
    );

    let mut by_key: HashMap<MergeKey, PrintedOccupation> = HashMap::new();
    let mut by_occupation: HashMap<PrintedOccupation, MergeKey> = HashMap::new();
    for configuration in &occupations.configurations {
        let key: MergeKey = configuration.key().to_vec();
        let occupation: PrintedOccupation = configuration
            .occupations
            .iter()
            .map(|entry| (entry.subshell.to_string(), entry.electrons))
            .collect();
        if let Some(other) = by_key.insert(key.clone(), occupation.clone()) {
            assert_eq!(
                other, occupation,
                "two configurations share the merge key {key:?} but differ in occupation"
            );
        }
        if let Some(other) = by_occupation.insert(occupation.clone(), key.clone()) {
            assert_eq!(
                other, key,
                "two configurations share an occupation vector but differ in key"
            );
        }
        for entry in &configuration.occupations {
            assert!(
                entry.electrons > 0,
                "an enumerated configuration carries a zero occupation for {}",
                entry.subshell
            );
        }
    }
}

/// Repeating a reference must not repeat its configurations: the merge consumes
/// equal keys. This is the only place the enumeration could produce a duplicate
/// configuration, and this is the control that proves it does not.
#[test]
fn a_repeated_reference_is_merged_not_generated_twice() {
    let single = "\
rcsfgenerate<< EOF
 * ! Orbital order
0
2s(2,i)2p(5,i)3p(1,i)

3s,3p,3d
0,4
           1  ! Number of excitations
n
EOF
";
    let doubled = single.replace(
        "2s(2,i)2p(5,i)3p(1,i)\n",
        "2s(2,i)2p(5,i)3p(1,i)\n2s(2,i)2p(5,i)3p(1,i)\n",
    );
    let once = enumerate_occupations(&ExcitationRequest::from_transcript(single).unwrap()).unwrap();
    let twice =
        enumerate_occupations(&ExcitationRequest::from_transcript(&doubled).unwrap()).unwrap();
    assert_eq!(
        once.configurations, twice.configurations,
        "a repeated reference changed the configuration list"
    );
    let audit = audit(&enumerated_requests(&twice, 0, 4)).unwrap();
    audit.assert_unique("doubled reference");
}

/// Distinct configurations never collide, even when they differ only in how
/// many electrons sit in the same shells — including the case where one omits a
/// shell that the other occupies.
#[test]
fn configurations_that_differ_only_in_occupation_never_collide() {
    let open = shell("4f");
    let other = shell("5g");
    let mut requests = Vec::new();
    for first in 1..=open.capacity() {
        for second in 1..=other.capacity() {
            let entries = [(open, first), (other, second)];
            let (min, max) = full_range(&entries);
            requests.push(request(&entries, min, max));
        }
    }
    // The same shells, one electron pair at a time, so the occupied set itself
    // differs between configurations.
    for electrons in 1..=2u8 {
        for subshell in [open, other] {
            let entries = [(subshell, electrons)];
            let (min, max) = full_range(&entries);
            requests.push(request(&entries, min, max));
        }
    }
    let audit = audit(&requests).unwrap();
    audit.assert_unique("occupation-only differences");
    assert!(
        audit.records >= 100,
        "the occupation family lost records: {}",
        audit.records
    );
}

/// The control that keeps the checks above honest: a family that *does* contain
/// duplicates — the same configuration scheduled twice — must be reported, and
/// the local layer must miss it while the global layer catches it. That
/// difference is exactly what the exact de-duplication chain is still for, and
/// what P6b must not remove from a path that can receive a repeated
/// configuration.
#[test]
fn the_audit_reports_duplicates_and_the_layers_disagree() {
    let entries = [(shell("4f"), 2u8), (shell("5g"), 2u8)];
    let (min, max) = full_range(&entries);
    let repeated = request(&entries, min, max);
    let audit = audit(&[repeated.clone(), repeated]).unwrap();
    audit.report("repeated configuration");
    assert!(
        !audit.row_duplicates.is_empty(),
        "the audit failed to find a duplicate that is plainly there"
    );
    assert_eq!(
        audit.row_duplicates.len(),
        audit.records / 2,
        "every record of the second copy must be reported once"
    );
    assert_eq!(
        audit.global_survivors.len(),
        audit.records / 2,
        "the global layer must keep one copy of every duplicated record"
    );
    assert_eq!(
        audit.local_survivors,
        (0..audit.records).collect::<Vec<_>>(),
        "the local layer cannot see a repeat that spans configurations"
    );
    assert!(
        audit.row_duplicates[0].first.starts_with("configuration 0"),
        "the survivor must be the first occurrence: {}",
        audit.row_duplicates[0].describe()
    );
}
