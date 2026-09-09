//! Nonrelativistic occupation enumeration used by `rcsfgenerate`.
//!
//! GRASP calls this part of the program `BLANDA`.  The reference
//! configuration is kept in nonrelativistic `nl` shells while the generated
//! tasks contain the two relativistic partners.  Keeping the two levels
//! separate is important: the excitation limit is measured in electrons
//! moved between `nl` shells, before an occupation is split into `j=l-1/2`
//! and `j=l+1/2`.

use anyhow::{Context, Result, ensure};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::str::FromStr;

use super::{Subshell, SubshellOccupation};

const ORBITAL_LETTERS: &[u8] = b"spdfghiklmn";

/// A nonrelativistic `nl` orbital.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Orbital {
    pub n: u8,
    pub l: u8,
}

impl Orbital {
    pub fn new(n: u8, l: u8) -> Result<Self> {
        ensure!(
            (1..=15).contains(&n),
            "principal quantum number must be 1..=15"
        );
        ensure!(
            l < n && l <= 10,
            "orbital angular momentum must satisfy 0 <= l < n <= 15"
        );
        Ok(Self { n, l })
    }

    pub fn label(self) -> String {
        format!(
            "{}{}",
            self.n,
            char::from(ORBITAL_LETTERS[usize::from(self.l)])
        )
    }

    fn capacity(self) -> u8 {
        2 + 4 * self.l
    }
}

impl FromStr for Orbital {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        ensure!(
            value.len() >= 2 && value.is_ascii(),
            "invalid orbital {value:?}"
        );
        let (n, letter) = value.split_at(value.len() - 1);
        let l = ORBITAL_LETTERS
            .iter()
            .position(|&item| item == letter.as_bytes()[0])
            .with_context(|| format!("invalid orbital letter in {value:?}"))?;
        let n = n
            .parse::<u8>()
            .with_context(|| format!("invalid orbital {value:?}"))?;
        Self::new(n, u8::try_from(l)?)
    }
}

/// How a reference shell participates in excitation enumeration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OccupationMode {
    /// The shell is fixed at its reference population.
    Inactive,
    /// The shell may exchange up to the configured number of electrons.
    Active,
    /// The shell is fixed and contributes to the closed core.
    Closed,
    /// The shell is active but cannot fall below this population.
    Minimum(u8),
    /// The shell changes in pairs.  This is the `d` selector in jjgen.
    Double,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceSubshell {
    pub orbital: Orbital,
    pub electrons: u8,
    pub mode: OccupationMode,
}

/// One reference configuration from the `rcsfgenerate` input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceConfiguration {
    pub shells: Vec<ReferenceSubshell>,
}

impl ReferenceConfiguration {
    pub fn from_spectroscopic(value: &str) -> Result<Self> {
        let mut shells = Vec::new();
        let bytes = value.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            let start = index;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
            ensure!(
                index > start && index < bytes.len(),
                "invalid configuration near {value:?}"
            );
            let letter_start = index;
            index += 1;
            let orbital = value[start..index].parse::<Orbital>()?;
            ensure!(
                index < bytes.len() && bytes[index] == b'(',
                "missing occupation in {value:?}"
            );
            index += 1;
            let occupation_start = index;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
            ensure!(
                index > occupation_start,
                "missing occupation near {value:?}"
            );
            let electrons = value[occupation_start..index].parse::<u8>()?;
            ensure!(
                index < bytes.len() && bytes[index] == b',',
                "missing selector near {value:?}"
            );
            index += 1;
            let selector_start = index;
            while index < bytes.len() && bytes[index] != b')' {
                index += 1;
            }
            ensure!(
                index > selector_start && index < bytes.len(),
                "missing selector near {value:?}"
            );
            let selector = value[selector_start..index].trim();
            index += 1;
            let mode = match selector {
                "i" | "I" => OccupationMode::Inactive,
                "c" | "C" => OccupationMode::Closed,
                "*" => OccupationMode::Active,
                "d" | "D" => OccupationMode::Double,
                digits => OccupationMode::Minimum(digits.parse::<u8>()?),
            };
            ensure!(
                electrons <= orbital.capacity(),
                "occupation {} exceeds capacity of {}",
                electrons,
                orbital.label()
            );
            ensure!(
                shells
                    .iter()
                    .all(|shell: &ReferenceSubshell| shell.orbital != orbital),
                "duplicate orbital {} in configuration",
                orbital.label()
            );
            let _ = letter_start;
            shells.push(ReferenceSubshell {
                orbital,
                electrons,
                mode,
            });
        }
        ensure!(!shells.is_empty(), "configuration cannot be empty");
        Ok(Self { shells })
    }
}

/// The complete input needed for the new-list path of `rcsfgenerate`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExcitationRequest {
    pub core: u8,
    pub active_orbitals: Vec<Orbital>,
    pub references: Vec<ReferenceConfiguration>,
    pub min_two_j: u16,
    pub max_two_j: u16,
    pub max_excitations: u8,
}

impl ExcitationRequest {
    pub fn from_transcript(input: &str) -> Result<Self> {
        let lines = input.lines().map(str::trim_end).collect::<Vec<_>>();
        let order = lines
            .iter()
            .position(|line| line.contains("Orbital order"))
            .context("missing orbital-order line")?;
        let core = lines
            .get(order + 1)
            .context("missing core selector")?
            .trim()
            .parse::<u8>()?;
        ensure!(core <= 6, "core selector must be 0..=6");
        let mut index = order + 2;
        let mut references = Vec::new();
        while let Some(line) = lines.get(index) {
            let line = line.trim();
            index += 1;
            if line.is_empty() || line == "*" {
                break;
            }
            references.push(ReferenceConfiguration::from_spectroscopic(line)?);
        }
        ensure!(
            !references.is_empty(),
            "at least one reference configuration is required"
        );
        let active_line = lines
            .get(index)
            .context("missing active-orbital line")?
            .trim();
        let active_orbitals = active_line
            .split(',')
            .map(|value| value.trim().parse())
            .collect::<Result<Vec<Orbital>>>()?;
        ensure!(
            !active_orbitals.is_empty(),
            "active orbital list cannot be empty"
        );
        index += 1;
        let j_range = lines
            .get(index)
            .context("missing 2J range")?
            .split(',')
            .map(|value| value.trim().parse::<u16>())
            .collect::<Result<Vec<_>, _>>()?;
        ensure!(
            j_range.len() == 2,
            "2J range must contain lower and upper bounds"
        );
        index += 1;
        let excitation_line = lines.get(index).context("missing excitation count")?;
        let max_excitations = excitation_line
            .split('!')
            .next()
            .unwrap_or_default()
            .trim()
            .parse::<i16>()?;
        ensure!(
            max_excitations >= 0,
            "negative excitation counts are not supported yet"
        );
        Ok(Self {
            core,
            active_orbitals,
            references,
            min_two_j: j_range[0],
            max_two_j: j_range[1],
            max_excitations: u8::try_from(max_excitations)?,
        })
    }
}

/// A relativistic occupation task passed to [`super::generate_csfs`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnumeratedConfiguration {
    pub occupations: Vec<SubshellOccupation>,
    key: Vec<(u8, u8, u8)>,
}

impl EnumeratedConfiguration {
    pub fn key(&self) -> &[(u8, u8, u8)] {
        &self.key
    }
}

/// Result of BLANDA-style enumeration and multi-reference merge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnumeratedOccupations {
    pub core_subshells: Vec<Subshell>,
    pub active_subshells: Vec<Subshell>,
    pub configurations: Vec<EnumeratedConfiguration>,
}

/// Enumerate all nonrelativistic occupations, split each into relativistic
/// partners, and merge reference lists using GRASP's `TEST/LIKA` ordering.
pub fn enumerate_occupations(request: &ExcitationRequest) -> Result<EnumeratedOccupations> {
    ensure!(
        request.min_two_j <= request.max_two_j,
        "minimum 2J exceeds maximum 2J"
    );
    ensure!(
        !request.references.is_empty(),
        "at least one reference configuration is required"
    );
    let active_max = active_limits(&request.active_orbitals)?;
    let max_n = request
        .active_orbitals
        .iter()
        .map(|orbital| orbital.n)
        .max()
        .context("active orbital list cannot be empty")?;
    let max_l = request
        .active_orbitals
        .iter()
        .map(|orbital| orbital.l)
        .max()
        .unwrap_or(0);
    let slots = slot_list(max_n, max_l);
    let core_orbitals = predefined_core(request.core)?;
    let core_set = core_orbitals
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let core_subshells = core_orbitals
        .iter()
        .flat_map(|&orbital| relativistic_partners(orbital))
        .collect::<Vec<_>>();
    let active_subshells = slots
        .iter()
        .filter(|slot| {
            active_max.get(&slot.l).is_some_and(|&n| slot.n <= n) && !core_set.contains(slot)
        })
        .flat_map(|&orbital| relativistic_partners(orbital))
        .collect::<Vec<_>>();
    let mut lists = Vec::new();
    for reference in &request.references {
        let fields = fields_for_reference(reference, &slots, &active_max, &core_set)?;
        let list = enumerate_reference(&fields, request.max_excitations)?;
        lists.push(list);
    }
    let configurations = merge_lists(lists);
    Ok(EnumeratedOccupations {
        core_subshells,
        active_subshells,
        configurations,
    })
}

fn active_limits(active: &[Orbital]) -> Result<HashMap<u8, u8>> {
    let mut limits = HashMap::new();
    for &orbital in active {
        if let Some(previous) = limits.insert(orbital.l, orbital.n) {
            ensure!(
                previous == orbital.n,
                "active orbital list contains two limits for l={}",
                orbital.l
            );
        }
    }
    Ok(limits)
}

fn slot_list(max_n: u8, max_l: u8) -> Vec<Orbital> {
    (1..=max_n)
        .flat_map(|n| (0..=max_l.min(n - 1)).map(move |l| Orbital { n, l }))
        .collect()
}

fn predefined_core(core: u8) -> Result<Vec<Orbital>> {
    ensure!(core <= 6, "core selector must be 0..=6");
    let mut result = Vec::new();
    for n in 1..=core {
        for l in 0..=3.min(n - 1) {
            result.push(Orbital { n, l });
        }
    }
    if core >= 3 {
        result.retain(|orbital| !(orbital.n == 3 && orbital.l == 2));
    }
    if core >= 4 {
        result.retain(|orbital| !(orbital.n == 4 && (orbital.l == 2 || orbital.l == 3)));
    }
    if core >= 5 {
        result.retain(|orbital| !(orbital.n == 5 && (orbital.l == 2 || orbital.l == 3)));
    }
    if core >= 6 {
        result.retain(|orbital| !(orbital.n == 6 && (orbital.l == 2 || orbital.l == 3)));
    }
    // The excitation wrapper's predefined core has the following additions.
    if core >= 4 {
        result.extend([
            Orbital { n: 3, l: 2 },
            Orbital { n: 4, l: 0 },
            Orbital { n: 4, l: 1 },
        ]);
    }
    if core >= 5 {
        result.extend([
            Orbital { n: 4, l: 2 },
            Orbital { n: 5, l: 0 },
            Orbital { n: 5, l: 1 },
        ]);
    }
    if core >= 6 {
        result.extend([
            Orbital { n: 4, l: 3 },
            Orbital { n: 5, l: 2 },
            Orbital { n: 6, l: 0 },
            Orbital { n: 6, l: 1 },
        ]);
    }
    result.sort_unstable();
    result.dedup();
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
struct SlotField {
    orbital: Orbital,
    reference: u8,
    mode: OccupationMode,
}

fn fields_for_reference(
    reference: &ReferenceConfiguration,
    slots: &[Orbital],
    active_max: &HashMap<u8, u8>,
    core: &std::collections::HashSet<Orbital>,
) -> Result<Vec<SlotField>> {
    let explicit = reference
        .shells
        .iter()
        .map(|shell| (shell.orbital, *shell))
        .collect::<HashMap<_, _>>();
    ensure!(
        !reference.shells.is_empty(),
        "reference configuration cannot be empty"
    );
    slots
        .iter()
        .map(|&orbital| {
            if let Some(shell) = explicit.get(&orbital) {
                ensure!(
                    !core.contains(&orbital) || shell.mode == OccupationMode::Closed,
                    "reference shell {} overlaps predefined core without c selector",
                    orbital.label()
                );
                Ok(SlotField {
                    orbital,
                    reference: shell.electrons,
                    mode: if core.contains(&orbital) {
                        OccupationMode::Closed
                    } else {
                        shell.mode
                    },
                })
            } else if core.contains(&orbital) {
                Ok(SlotField {
                    orbital,
                    reference: orbital.capacity(),
                    mode: OccupationMode::Closed,
                })
            } else {
                let mode = if active_max.get(&orbital.l).is_some_and(|&n| orbital.n <= n) {
                    OccupationMode::Active
                } else {
                    OccupationMode::Inactive
                };
                Ok(SlotField {
                    orbital,
                    reference: 0,
                    mode,
                })
            }
        })
        .collect()
}

/// The part of the walk that does not change during the recursion: the slot
/// table and the two reference quantities every candidate is measured against.
struct Walk<'a> {
    fields: &'a [SlotField],
    max_excitations: u8,
    /// `antal` in blanda.f90.
    electrons: u16,
    /// `par0` in blanda.f90.  Parity is `sum(l * occupation) mod 2`; the
    /// relativistic split leaves it unchanged, so it is decided at this level.
    parity: u16,
}

/// The running quantities blanda.f90 carries down its slot loops: `antel`,
/// `varupp`, `varned` and `par`.
#[derive(Clone, Copy, Default)]
struct Partial {
    index: usize,
    electrons: u16,
    varupp: u8,
    varned: u8,
    parity: u16,
}

fn enumerate_reference(
    fields: &[SlotField],
    max_excitations: u8,
) -> Result<Vec<EnumeratedConfiguration>> {
    let walk = Walk {
        fields,
        max_excitations,
        electrons: fields.iter().map(|field| u16::from(field.reference)).sum(),
        parity: fields
            .iter()
            .map(|field| u16::from(field.orbital.l) * u16::from(field.reference))
            .sum::<u16>()
            % 2,
    };
    let mut occupations = vec![0u8; fields.len()];
    let mut output = Vec::new();
    enumerate_nonrel(&walk, Partial::default(), &mut occupations, &mut output)?;
    output.sort_by(|left, right| right.key.cmp(&left.key));
    Ok(output)
}

fn enumerate_nonrel(
    walk: &Walk<'_>,
    partial: Partial,
    occupations: &mut [u8],
    output: &mut Vec<EnumeratedConfiguration>,
) -> Result<()> {
    if partial.varupp > walk.max_excitations || partial.varned > walk.max_excitations {
        return Ok(());
    }
    if partial.index == walk.fields.len() {
        // blanda.f90 drops a candidate whose parity differs from the reference
        // before it reaches GEN, so one reference configuration only ever
        // yields CSFs of its own parity.
        if partial.electrons == walk.electrons && partial.parity == walk.parity {
            split_configuration(walk.fields, occupations, output)?;
        }
        return Ok(());
    }
    let field = walk.fields[partial.index];
    let (start, stop, step) = bounds(field, walk.max_excitations, partial.varupp, partial.varned);
    let mut candidate = start;
    loop {
        if candidate >= stop {
            let difference = i16::from(candidate) - i16::from(field.reference);
            let electrons = partial.electrons + u16::from(candidate);
            if electrons <= walk.electrons {
                occupations[partial.index] = candidate;
                let (varupp, varned) = if difference >= 0 {
                    (partial.varupp + u8::try_from(difference)?, partial.varned)
                } else {
                    (partial.varupp, partial.varned + u8::try_from(-difference)?)
                };
                enumerate_nonrel(
                    walk,
                    Partial {
                        index: partial.index + 1,
                        electrons,
                        varupp,
                        varned,
                        parity: (partial.parity
                            + u16::from(field.orbital.l) * u16::from(candidate))
                            % 2,
                    },
                    occupations,
                    output,
                )?;
            }
        }
        if candidate < stop || candidate < step {
            break;
        }
        candidate -= step;
    }
    Ok(())
}

fn bounds(field: SlotField, max_excitations: u8, varupp: u8, varned: u8) -> (u8, u8, u8) {
    match field.mode {
        OccupationMode::Inactive | OccupationMode::Closed => (field.reference, field.reference, 1),
        OccupationMode::Active | OccupationMode::Minimum(_) | OccupationMode::Double => {
            let capacity = field.orbital.capacity();
            let start = (i16::from(field.reference) + i16::from(max_excitations)
                - i16::from(varupp))
            .clamp(0, i16::from(capacity)) as u8;
            let low = match field.mode {
                OccupationMode::Minimum(value) => field.reference.min(value),
                _ => 0,
            };
            let stop = (i16::from(field.reference) - i16::from(max_excitations) + i16::from(varned))
                .max(i16::from(low))
                .clamp(0, i16::from(capacity)) as u8;
            let start = if field.mode == OccupationMode::Double {
                start - start % 2
            } else {
                start
            };
            (
                start,
                stop,
                if field.mode == OccupationMode::Double {
                    2
                } else {
                    1
                },
            )
        }
    }
}

fn split_configuration(
    fields: &[SlotField],
    occupations: &[u8],
    output: &mut Vec<EnumeratedConfiguration>,
) -> Result<()> {
    let key = fields
        .iter()
        .zip(occupations)
        .map(|(_, &occupation)| (occupation, 0, 0))
        .collect::<Vec<_>>();
    let mut branches = Vec::new();
    split_branches(fields, occupations, 0, &mut branches, Vec::new(), key)
        .with_context(|| "failed to split nonrelativistic occupation")?;
    output.extend(branches);
    Ok(())
}

fn split_branches(
    fields: &[SlotField],
    occupations: &[u8],
    index: usize,
    output: &mut Vec<EnumeratedConfiguration>,
    current: Vec<SubshellOccupation>,
    mut key: Vec<(u8, u8, u8)>,
) -> Result<()> {
    if index == fields.len() {
        output.push(EnumeratedConfiguration {
            occupations: current,
            key,
        });
        return Ok(());
    }
    let field = fields[index];
    let occupation = occupations[index];
    if field.mode == OccupationMode::Closed {
        return split_branches(fields, occupations, index + 1, output, current, key);
    }
    for (lower, upper) in split_occupation(field.orbital, occupation) {
        let mut next = current.clone();
        if field.orbital.l == 0 {
            if lower > 0 {
                next.push(SubshellOccupation {
                    subshell: Subshell::new(field.orbital.n, -1).expect("validated orbital"),
                    electrons: lower,
                });
            }
        } else {
            let lower_subshell =
                Subshell::new(field.orbital.n, field.orbital.l as i8).expect("validated orbital");
            let upper_subshell = Subshell::new(field.orbital.n, -(field.orbital.l as i8) - 1)
                .expect("validated orbital");
            if lower > 0 {
                next.push(SubshellOccupation {
                    subshell: lower_subshell,
                    electrons: lower,
                });
            }
            if upper > 0 {
                next.push(SubshellOccupation {
                    subshell: upper_subshell,
                    electrons: upper,
                });
            }
        }
        key[index].1 = upper;
        key[index].2 = lower;
        split_branches(fields, occupations, index + 1, output, next, key.clone())?;
    }
    Ok(())
}

fn relativistic_partners(orbital: Orbital) -> Vec<Subshell> {
    if orbital.l == 0 {
        return vec![Subshell::new(orbital.n, -1).expect("validated orbital")];
    }
    vec![
        Subshell::new(orbital.n, orbital.l as i8).expect("validated orbital"),
        Subshell::new(orbital.n, -(orbital.l as i8) - 1).expect("validated orbital"),
    ]
}

fn split_occupation(orbital: Orbital, electrons: u8) -> Vec<(u8, u8)> {
    if orbital.l == 0 {
        return vec![(electrons, 0)];
    }
    let mut result = Vec::new();
    let mut upper = electrons.min(2 * orbital.l + 2);
    let lower_bound = electrons.saturating_sub(2 * orbital.l);
    loop {
        let lower = electrons - upper;
        result.push((lower, upper));
        if upper == lower_bound {
            break;
        }
        upper -= 1;
    }
    result
}

fn merge_lists(lists: Vec<Vec<EnumeratedConfiguration>>) -> Vec<EnumeratedConfiguration> {
    let mut merged = Vec::new();
    for list in lists {
        merged = merge_two(merged, list);
    }
    merged
}

fn merge_two(
    left: Vec<EnumeratedConfiguration>,
    right: Vec<EnumeratedConfiguration>,
) -> Vec<EnumeratedConfiguration> {
    let mut output = Vec::with_capacity(left.len() + right.len());
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() || right_index < right.len() {
        match (left.get(left_index), right.get(right_index)) {
            (Some(a), Some(b)) => match a.key.cmp(&b.key) {
                Ordering::Greater => {
                    output.push(a.clone());
                    left_index += 1;
                }
                Ordering::Less => {
                    output.push(b.clone());
                    right_index += 1;
                }
                Ordering::Equal => {
                    output.push(a.clone());
                    left_index += 1;
                    right_index += 1;
                }
            },
            (Some(a), None) => {
                output.push(a.clone());
                left_index += 1;
            }
            (None, Some(b)) => {
                output.push(b.clone());
                right_index += 1;
            }
            (None, None) => break,
        }
    }
    output
}
