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

use super::{ResourceBudget, ResourcePermit, Subshell, SubshellOccupation};
use crate::orbital_label::{self, MAX_PRINCIPAL_QUANTUM_NUMBER, ORBITAL_LETTERS};
use std::mem::size_of;

/// A nonrelativistic `nl` orbital.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Orbital {
    n: u8,
    l: u8,
}

impl Orbital {
    /// Create an orbital through the established constructor.
    ///
    /// See [`Self::try_new`] for the allowed quantum numbers.
    pub fn new(n: u8, l: u8) -> Result<Self> {
        Self::try_new(n, l)
    }

    /// Create a nonrelativistic orbital within rCSFs' supported generation range.
    pub fn try_new(n: u8, l: u8) -> Result<Self> {
        ensure!(
            (1..=MAX_PRINCIPAL_QUANTUM_NUMBER).contains(&n),
            "principal quantum number must be 1..={MAX_PRINCIPAL_QUANTUM_NUMBER}"
        );
        ensure!(
            l < n && l <= 10,
            "orbital angular momentum must satisfy l < n and l <= 10"
        );
        Ok(Self { n, l })
    }

    /// Principal quantum number within the generator's supported range.
    #[must_use]
    pub fn n(self) -> u8 {
        self.n
    }

    /// Orbital angular momentum, with `0 <= l < n`.
    #[must_use]
    pub fn l(self) -> u8 {
        self.l
    }

    /// Format the orbital as its principal quantum number followed by its orbital letter.
    #[must_use]
    pub fn label(self) -> String {
        format!(
            "{}{}",
            self.n,
            char::from(ORBITAL_LETTERS.as_bytes()[usize::from(self.l)])
        )
    }

    fn capacity(self) -> u8 {
        2 + 4 * self.l
    }
}

impl FromStr for Orbital {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (n, l) = orbital_label::parse_nl(value)?;
        Self::try_new(
            u8::try_from(n).with_context(|| {
                format!("orbital n must be in 1..={MAX_PRINCIPAL_QUANTUM_NUMBER}")
            })?,
            l,
        )
        .with_context(|| format!("invalid generator orbital {value:?}"))
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

/// Initial population and excitation rule for one nonrelativistic reference shell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceSubshell {
    /// Nonrelativistic shell shared by both relativistic partners, where applicable.
    pub orbital: Orbital,
    /// Total reference electrons in the shell before splitting into relativistic partners.
    pub electrons: u8,
    /// Allowed occupation changes when generating configurations from this reference.
    pub mode: OccupationMode,
}

/// One reference configuration from the `rcsfgenerate` input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceConfiguration {
    /// Reference shell specifications in input order.
    pub shells: Vec<ReferenceSubshell>,
}

impl ReferenceConfiguration {
    /// Parse adjacent `nl(electrons,selector)` entries, such as `3s(2,i)3p(1,*)`.
    ///
    /// Selectors `i`, `c`, `*`, and `d` select inactive, closed, active, and
    /// pair transfer modes. An integer sets the minimum occupation.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed entries, non-ASCII orbital letters,
    /// duplicate shells, or populations exceeding the shell capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use _rcsfs::configuration_generation::{OccupationMode, ReferenceConfiguration};
    ///
    /// let reference = ReferenceConfiguration::from_spectroscopic("3s(2,i)3p(1,*)")?;
    /// assert_eq!(reference.shells[1].electrons, 1);
    /// assert_eq!(reference.shells[1].mode, OccupationMode::Active);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_spectroscopic(value: &str) -> Result<Self> {
        let mut shells = Vec::new();
        let bytes = value.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            let (label, _) = value[index..]
                .split_once('(')
                .with_context(|| format!("missing occupation in {value:?}"))?;
            let orbital = label.parse::<Orbital>()?;
            index += label.len() + 1;
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
    /// Low-level jjgen core selector, after transcript normalization.
    pub core: u8,
    /// Highest principal quantum number per angular symmetry in the orbital walk.
    /// Closed-core subshells inside these limits remain excluded from excitations.
    pub active_orbitals: Vec<Orbital>,
    /// Reference configurations in input-list order, including any core shells
    /// inserted during transcript normalization.
    pub references: Vec<ReferenceConfiguration>,
    /// Inclusive lower bound on twice the total angular momentum of generated CSFs.
    pub min_two_j: u16,
    /// Inclusive upper bound on twice the total angular momentum of generated CSFs.
    pub max_two_j: u16,
    /// Maximum number of electrons moved between nonrelativistic shells per reference.
    pub max_excitations: u8,
}

impl ExcitationRequest {
    /// Parse one new-list `rcsfgenerate` transcript using default orbital order (`*`).
    ///
    /// Applies the wrapper's core and active-orbital conventions. A negative
    /// excitation limit enables pair transfers for otherwise unspecified active
    /// shells and uses the absolute value as the electron limit.
    ///
    /// # Errors
    ///
    /// Returns an error for an incomplete transcript, a malformed reference
    /// configuration, unsupported orbital ordering, or a request to continue
    /// with another list.
    pub fn from_transcript(input: &str) -> Result<Self> {
        let lines = input.lines().map(str::trim_end).collect::<Vec<_>>();
        let order = lines
            .iter()
            .position(|line| line.contains("Orbital order"))
            .context("missing orbital-order line")?;
        let selector = lines[order].split('!').next().unwrap_or_default().trim();
        ensure!(
            selector == "*",
            "orbital order {selector:?} is not supported; only * is implemented"
        );
        let core = lines
            .get(order + 1)
            .context("missing core selector")?
            .split('!')
            .next()
            .unwrap_or_default()
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
            let prefix = match core {
                5 => "4d(10,c)5s(2,c)5p(6,c)",
                6 => "4f(14,c)5d(10,c)6s(2,c)6p(6,c)",
                _ => "",
            };
            references.push(ReferenceConfiguration::from_spectroscopic(&format!(
                "{prefix}{line}"
            ))?);
        }
        ensure!(
            !references.is_empty(),
            "at least one reference configuration is required"
        );
        let active_line = lines
            .get(index)
            .context("missing active-orbital line")?
            .trim();
        let active_line = match (core, active_line) {
            (5, "4f") => "5s,4f",
            (6, "5f") => "6s,5f",
            (6, "5f,5g") => "6s,5f,5g",
            (6, "5g") => "6s,5g",
            _ => active_line,
        };
        let mut active_orbitals = active_line
            .split(',')
            .map(|label| label.trim().parse())
            .collect::<Result<Vec<Orbital>>>()?;
        // The wrapper walks through closed 4f when its implicit f limit lands
        // after a final d limit. Use the closed shell itself as the upper bound.
        if core == 6
            && !active_orbitals.iter().any(|orbital| orbital.l == 3)
            && active_orbitals
                .iter()
                .position(|orbital| orbital.l == 2)
                .is_some_and(|index| index + 1 == active_orbitals.len())
        {
            active_orbitals.push(Orbital { n: 4, l: 3 });
        }
        let last_l = active_orbitals
            .last()
            .context("active orbital list cannot be empty")?
            .l;
        ensure!(
            active_orbitals.iter().all(|orbital| orbital.l <= last_l),
            "active orbital limits must end with their highest l symmetry"
        );
        // rcsfexcitation uses the LAST listed l, raised to the predefined
        // core's lmax, rather than the maximum listed l. Its inserted 3f can
        // therefore be ignored (e.g. core 6 with 7s); in that case it also
        // omits the injected closed 4f. Preserve this upstream behavior.
        if core == 6 && last_l < 3 {
            for reference in &mut references {
                reference
                    .shells
                    .retain(|shell| shell.orbital != Orbital { n: 4, l: 3 });
            }
        }
        index += 1;
        let j_range = lines
            .get(index)
            .context("missing 2J range")?
            .split('!')
            .next()
            .unwrap_or_default()
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .filter(|value| !value.is_empty())
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
        if max_excitations < 0 {
            // Only missing reference shells receive d. An explicitly specified
            // zero-population shell retains its selector, just as in the wrapper.
            let limits = active_limits(&active_orbitals)?;
            let max_n = active_orbitals
                .iter()
                .map(|orbital| orbital.n)
                .max()
                .context("active orbital list cannot be empty")?;
            let max_l = active_orbitals
                .iter()
                .map(|orbital| orbital.l)
                .max()
                .context("active orbital list cannot be empty")?;
            let predefined = predefined_core(if core >= 5 { core - 1 } else { core })?;
            for reference in &mut references {
                for orbital in slot_list(max_n, max_l) {
                    if limits.get(&orbital.l).is_some_and(|&n| orbital.n <= n)
                        && !predefined.contains(&orbital)
                        && !reference
                            .shells
                            .iter()
                            .any(|shell| shell.orbital == orbital)
                    {
                        reference.shells.push(ReferenceSubshell {
                            orbital,
                            electrons: 0,
                            mode: OccupationMode::Double,
                        });
                    }
                }
            }
        }
        let continuation = lines
            .get(index + 1)
            .context("missing list continuation answer")?
            .split('!')
            .next()
            .unwrap_or_default()
            .trim();
        ensure!(
            continuation != "y" && continuation != "Y",
            "multiple lists are not supported yet"
        );
        ensure!(
            continuation == "n" || continuation == "N",
            "unsupported list continuation answer {continuation:?}"
        );
        ensure!(
            lines[index + 2..]
                .iter()
                .all(|line| line.trim().is_empty() || line.trim() == "EOF"),
            "unexpected input after list termination"
        );
        Ok(Self {
            core: if core >= 5 { core - 1 } else { core },
            active_orbitals,
            references,
            min_two_j: j_range[0],
            max_two_j: j_range[1],
            max_excitations: u8::try_from(max_excitations.unsigned_abs())?,
        })
    }
}

/// A relativistic occupation task passed to [`super::generate_csfs`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnumeratedConfiguration {
    /// Nonzero peel occupations in the order used for relativistic coupling.
    pub occupations: Vec<SubshellOccupation>,
    key: Vec<(u8, u8, u8)>,
}

impl EnumeratedConfiguration {
    /// Borrow the key used to sort and deduplicate the enumerated occupations.
    ///
    /// Each slot stores `(total, upper, lower)` for a nonrelativistic shell in
    /// the enumeration order. The partners are `j = l + 1/2` and `j = l - 1/2`
    /// for `l > 0`. For a peel s shell, the sole population is stored in
    /// `lower`. Closed-core slots keep the total with both partner counts zero.
    #[must_use]
    pub fn key(&self) -> &[(u8, u8, u8)] {
        &self.key
    }
}

/// Result of BLANDA-style enumeration and multi-reference merge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnumeratedOccupations {
    /// Fully occupied relativistic subshells assigned to the closed core.
    pub core_subshells: Vec<Subshell>,
    /// Non-core relativistic subshells allowed by the active orbital limits.
    pub active_subshells: Vec<Subshell>,
    /// Unique occupation configurations merged across the input references.
    pub configurations: Vec<EnumeratedConfiguration>,
}

/// Charges the non-spillable occupation arena while it is being built.
///
/// The charge intentionally includes a small safety factor because reference
/// lists are merged by cloning into a second vector before the old vectors are
/// released.  It is released as one scoped reservation when the caller drops
/// the returned guard after generation has finished consuming the occupations.
pub(crate) struct EnumerationCharge {
    permit: ResourcePermit,
}

impl EnumerationCharge {
    fn new(budget: ResourceBudget) -> Result<Self> {
        Ok(Self {
            permit: budget.try_reserve(0, "occupation enumeration")?,
        })
    }

    fn add_configuration(&mut self, occupation_capacity: usize, key_capacity: usize) -> Result<()> {
        let bytes = size_of::<EnumeratedConfiguration>()
            .checked_add(
                occupation_capacity
                    .checked_mul(size_of::<SubshellOccupation>())
                    .context("occupation capacity byte count overflow")?,
            )
            .and_then(|total| {
                total.checked_add(key_capacity.saturating_mul(size_of::<(u8, u8, u8)>()))
            })
            .and_then(|total| total.checked_add(32))
            .context("occupation configuration byte count overflow")?;
        // Account for the source lists and the temporary merged output.  This
        // is intentionally conservative; it protects the non-spillable path
        // without claiming to be an operating-system RSS limit.
        let bytes = u64::try_from(bytes)
            .ok()
            .and_then(|value| value.checked_mul(3))
            .context("occupation configuration byte count exceeds u64")?;
        let next = self
            .permit
            .bytes()
            .checked_add(bytes)
            .context("occupation memory accounting overflow")?;
        self.permit
            .resize(next, "occupation enumeration")
            .with_context(|| {
                format!("cannot retain another occupation configuration ({bytes} bytes)")
            })
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.permit.bytes()
    }
}

/// Enumerate all nonrelativistic occupations, split each into relativistic
/// partners, and merge reference lists using GRASP's `TEST/LIKA` ordering.
pub fn enumerate_occupations(request: &ExcitationRequest) -> Result<EnumeratedOccupations> {
    let (occupations, _charge) = enumerate_occupations_with_budget(request, None)?;
    Ok(occupations)
}

/// Enumerate occupations while charging the non-spillable configuration
/// structure against a shared generation budget.
pub(crate) fn enumerate_occupations_with_budget(
    request: &ExcitationRequest,
    budget: Option<&ResourceBudget>,
) -> Result<(EnumeratedOccupations, Option<EnumerationCharge>)> {
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
        .unwrap_or(0)
        .max(
            predefined_core(request.core)?
                .iter()
                .map(|orbital| orbital.l)
                .max()
                .unwrap_or(0),
        );
    let slots = slot_list(max_n, max_l);
    let mut core_orbitals = predefined_core(request.core)?;
    for shell in &request.references[0].shells {
        if shell.mode == OccupationMode::Closed {
            ensure!(
                shell.electrons == shell.orbital.capacity(),
                "closed shell {} must be full",
                shell.orbital.label()
            );
            if !core_orbitals.contains(&shell.orbital) {
                core_orbitals.push(shell.orbital);
            }
        }
    }
    core_orbitals.sort_unstable();
    for reference in &request.references {
        for shell in &reference.shells {
            ensure!(
                shell.electrons <= shell.orbital.capacity(),
                "occupation exceeds orbital capacity"
            );
            ensure!(
                shell.mode != OccupationMode::Double || shell.electrons == 0,
                "d selector requires an empty reference shell"
            );
            ensure!(
                shell.mode != OccupationMode::Closed
                    || (shell.electrons == shell.orbital.capacity()
                        && core_orbitals.contains(&shell.orbital)),
                "closed shells must be full and consistent across references"
            );
            ensure!(
                slots.contains(&shell.orbital),
                "reference shell {} is outside the orbital input range",
                shell.orbital.label()
            );
        }
    }
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
    let mut charge = budget.cloned().map(EnumerationCharge::new).transpose()?;
    let mut lists = Vec::new();
    for reference in &request.references {
        let fields = fields_for_reference(reference, &slots, &active_max, &core_set)?;
        let list = enumerate_reference(&fields, request.max_excitations, charge.as_mut())?;
        lists.push(list);
    }
    let configurations = merge_lists(lists);
    Ok((
        EnumeratedOccupations {
            core_subshells,
            active_subshells,
            configurations,
        },
        charge,
    ))
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
            let excluded = match core {
                3 => (n, l) == (3, 2),
                4 => n == 4 && l >= 2,
                5 => (n, l) == (4, 3) || (n == 5 && l >= 2),
                6 => (n, l) == (5, 3) || (n == 6 && l >= 2),
                _ => false,
            };
            if !excluded {
                result.push(Orbital { n, l });
            }
        }
    }
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
    mut charge: Option<&mut EnumerationCharge>,
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
    enumerate_nonrel(
        &walk,
        Partial::default(),
        &mut occupations,
        &mut output,
        &mut charge,
    )?;
    output.sort_by(|left, right| right.key.cmp(&left.key));
    Ok(output)
}

fn enumerate_nonrel(
    walk: &Walk<'_>,
    partial: Partial,
    occupations: &mut [u8],
    output: &mut Vec<EnumeratedConfiguration>,
    charge: &mut Option<&mut EnumerationCharge>,
) -> Result<()> {
    if partial.varupp > walk.max_excitations || partial.varned > walk.max_excitations {
        return Ok(());
    }
    if partial.index == walk.fields.len() {
        // blanda.f90 drops a candidate whose parity differs from the reference
        // before it reaches GEN, so one reference configuration only ever
        // yields CSFs of its own parity.
        if partial.electrons == walk.electrons && partial.parity == walk.parity {
            split_configuration(walk.fields, occupations, output, charge)?;
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
                    charge,
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
            let capacity = if field.orbital.l >= 5 {
                4
            } else {
                field.orbital.capacity()
            };
            let start = (i16::from(field.reference) + i16::from(max_excitations)
                - i16::from(varupp))
            .clamp(0, i16::from(capacity)) as u8;
            let low = match field.mode {
                OccupationMode::Minimum(value) => field.reference.min(value),
                _ => 0,
            };
            let stop = (i16::from(field.reference) - i16::from(max_excitations) + i16::from(varned))
                .max(i16::from(low)) as u8;
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
    charge: &mut Option<&mut EnumerationCharge>,
) -> Result<()> {
    let key = fields
        .iter()
        .zip(occupations)
        .map(|(_, &occupation)| (occupation, 0, 0))
        .collect::<Vec<_>>();
    let mut branches = Vec::new();
    split_branches(
        fields,
        occupations,
        0,
        &mut branches,
        Vec::new(),
        key,
        charge,
    )
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
    charge: &mut Option<&mut EnumerationCharge>,
) -> Result<()> {
    if index == fields.len() {
        if let Some(charge) = charge.as_deref_mut() {
            charge.add_configuration(current.capacity(), key.capacity())?;
        }
        output.push(EnumeratedConfiguration {
            occupations: current,
            key,
        });
        return Ok(());
    }
    let field = fields[index];
    let occupation = occupations[index];
    if field.mode == OccupationMode::Closed {
        return split_branches(fields, occupations, index + 1, output, current, key, charge);
    }
    for (lower, upper) in split_occupation(field.orbital, occupation) {
        let mut next = current.clone();
        if field.orbital.l == 0 {
            if lower > 0 {
                next.push(SubshellOccupation {
                    subshell: Subshell::try_new(field.orbital.n, -1).expect("validated orbital"),
                    electrons: lower,
                });
            }
        } else {
            let lower_subshell = Subshell::try_new(field.orbital.n, field.orbital.l as i8)
                .expect("validated orbital");
            let upper_subshell = Subshell::try_new(field.orbital.n, -(field.orbital.l as i8) - 1)
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
        split_branches(
            fields,
            occupations,
            index + 1,
            output,
            next,
            key.clone(),
            charge,
        )?;
    }
    Ok(())
}

fn relativistic_partners(orbital: Orbital) -> Vec<Subshell> {
    if orbital.l == 0 {
        return vec![Subshell::try_new(orbital.n, -1).expect("validated orbital")];
    }
    vec![
        Subshell::try_new(orbital.n, orbital.l as i8).expect("validated orbital"),
        Subshell::try_new(orbital.n, -(orbital.l as i8) - 1).expect("validated orbital"),
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
        // BLANDA/BLANDB restrict the j=l+1/2 partner of every g shell
        // to at most two electrons. For h and higher, both partners are
        // restricted to two. These checks appear in the next orbital's loop
        // (or the final loop) in GRASP; apply them here before a configuration
        // reaches the CSF generator so counting and generation agree.
        if (orbital.l < 4 || upper <= 2) && (orbital.l < 5 || lower <= 2) {
            result.push((lower, upper));
        }
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
