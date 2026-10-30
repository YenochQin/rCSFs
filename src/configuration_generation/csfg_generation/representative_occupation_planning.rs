use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::configuration_generation::shared_generation::{
    ExcitationRequest, OccupationMode, Orbital, Subshell, SubshellOccupation, enumerate_occupations,
};

type Signature = Vec<(u8, i8, u8)>;

#[derive(Clone, Copy, Debug)]
pub(super) struct Channel {
    pub kappa: i8,
    pub first: u8,
    pub last: u8,
}

pub(super) struct Template {
    pub occupations: Vec<SubshellOccupation>,
    pub kind: usize,
    pub multiplicity: u64,
}

pub(super) struct Plan {
    pub core: Vec<Subshell>,
    pub labels: Vec<Subshell>,
    pub peel: Vec<Subshell>,
    pub templates: Vec<Template>,
    pub unique_occupations: usize,
}

fn limits(text: &str) -> Result<BTreeMap<u8, u8>> {
    let mut limits = BTreeMap::new();
    for value in text.split(',') {
        let orbital: Orbital = value.trim().parse()?;
        ensure!(
            limits.insert(orbital.l, orbital.n).is_none(),
            "duplicate labeling l channel"
        );
    }
    Ok(limits)
}

fn partners(n: u8, l: u8) -> Result<Vec<Subshell>> {
    if l == 0 {
        Ok(vec![Subshell::new(n, -1)?])
    } else {
        Ok(vec![
            Subshell::new(n, l as i8)?,
            Subshell::new(n, -(l as i8) - 1)?,
        ])
    }
}

fn signature(occupations: &[SubshellOccupation]) -> Signature {
    let mut key = occupations
        .iter()
        .map(|o| (o.subshell.n(), o.subshell.kappa(), o.electrons))
        .collect::<Vec<_>>();
    key.sort_unstable();
    key
}

// GRASPG TEST compares population triples in canonical n/l order, even though
// GEN couples occupied orbitals in its independently specified orbital order.
fn fortran_key(occupations: &[SubshellOccupation]) -> Vec<(u8, u8, u8)> {
    let mut key = Vec::new();
    for n in 1..=15 {
        for l in 0..n.min(11) {
            let mut lower = 0;
            let mut upper = 0;
            for o in occupations {
                if o.subshell.n() == n && o.subshell.l() == l {
                    if o.subshell.kappa() > 0 || l == 0 {
                        lower = o.electrons;
                    } else {
                        upper = o.electrons;
                    }
                }
            }
            key.push((lower + upper, upper, lower));
        }
    }
    key
}

impl Plan {
    pub(super) fn build(request: &ExcitationRequest, labeling: &str) -> Result<Self> {
        let label_limits = limits(labeling)?;
        let enumerated = enumerate_occupations(request)?;
        ensure!(
            !enumerated.configurations.is_empty(),
            "no occupation configurations generated"
        );
        let active: BTreeMap<_, _> = request.active_orbitals.iter().map(|o| (o.l, o.n)).collect();
        for (&l, &n) in &label_limits {
            ensure!(
                active.get(&l).is_some_and(|&max| max >= n),
                "labeling space must be within the active space"
            );
        }
        let mut all_labels = Vec::new();
        for n in 1..=15 {
            for (&l, &max) in &label_limits {
                if l < n && n <= max {
                    all_labels.extend(partners(n, l)?);
                }
            }
        }
        let label_set: HashSet<_> = all_labels.iter().copied().collect();
        ensure!(
            enumerated
                .core_subshells
                .iter()
                .all(|s| label_set.contains(s)),
            "labeling space must contain the complete closed core"
        );
        for reference in &request.references {
            for shell in &reference.shells {
                ensure!(
                    shell.mode != OccupationMode::Double,
                    "CSFG pair-only d selectors are not supported"
                );
                if !label_limits
                    .get(&shell.orbital.l)
                    .is_some_and(|&n| shell.orbital.n <= n)
                {
                    ensure!(
                        shell.electrons == 0 && shell.mode == OccupationMode::Active,
                        "reference occupations and occupation restrictions must be in the labeling space"
                    );
                }
            }
        }
        let core_set: HashSet<_> = enumerated.core_subshells.iter().copied().collect();
        let labels = all_labels
            .into_iter()
            .filter(|s| !core_set.contains(s))
            .collect::<Vec<_>>();
        let mut peel = labels.clone();
        let mut channels = HashMap::new();
        for (&l, &last) in &active {
            ensure!(
                last > l,
                "nonphysical active-space sentinels are not supported in CSFG mode"
            );
            let first = (label_limits.get(&l).copied().unwrap_or(l) + 1).max(l + 1);
            if first > last {
                continue;
            }
            for shell in partners(last, l)? {
                let channel = Channel {
                    kappa: shell.kappa(),
                    first,
                    last,
                };
                for n in first..=last {
                    peel.push(Subshell::new(n, channel.kappa)?);
                }
                channels.insert(channel.kappa, channel);
            }
        }
        ensure!(
            enumerated.core_subshells.len() + peel.len() <= 127,
            "CSFG output exceeds the default GRASPG capacity of 127 subshells"
        );
        let indices: HashMap<_, _> = peel.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let expected_electrons: u16 = enumerated.configurations[0]
            .occupations
            .iter()
            .map(|o| u16::from(o.electrons))
            .sum();
        let allowed: HashSet<_> = enumerated
            .configurations
            .iter()
            .map(|c| signature(&c.occupations))
            .collect();
        let mut templates = HashMap::new();
        for configuration in &enumerated.configurations {
            ensure!(
                configuration
                    .occupations
                    .iter()
                    .map(|o| u16::from(o.electrons))
                    .sum::<u16>()
                    == expected_electrons,
                "all references must have the same electron count"
            );
            let mut occupations = configuration.occupations.clone();
            let mut correlated = occupations
                .iter()
                .enumerate()
                .filter(|(_, o)| !label_set.contains(&o.subshell))
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let electrons: u8 = correlated.iter().map(|&i| occupations[i].electrons).sum();
            ensure!(
                electrons <= 2,
                "at most two correlation electrons are supported"
            );
            let kind = match (correlated.len(), electrons) {
                (0, 0) => 1,
                (1, 1) => 2,
                (1, 2) => 5,
                (2, 2)
                    if occupations[correlated[0]].subshell.kappa()
                        == occupations[correlated[1]].subshell.kappa() =>
                {
                    4
                }
                (2, 2) => 3,
                _ => anyhow::bail!("unsupported correlation occupation"),
            };
            correlated.sort_by_key(|&i| occupations[i].subshell.n());
            for (position, &i) in correlated.iter().enumerate() {
                let c = channels
                    .get(&occupations[i].subshell.kappa())
                    .context("occupation outside the declared orbital layout")?;
                let n = if kind == 4 && position == 0 {
                    ensure!(
                        c.last > c.first,
                        "same-kappa single occupations require two radial orbitals"
                    );
                    c.last - 1
                } else {
                    c.last
                };
                occupations[i].subshell = Subshell::new(n, c.kappa)?;
            }
            ensure!(
                occupations
                    .iter()
                    .all(|o| indices.contains_key(&o.subshell)),
                "occupation outside orbital layout"
            );
            occupations.sort_by_key(|o| indices[&o.subshell]);
            templates
                .entry(signature(&occupations))
                .or_insert(Template {
                    occupations,
                    kind,
                    multiplicity: 0,
                });
        }
        let mut templates = templates.into_values().collect::<Vec<_>>();
        // Verify exact, non-overlapping physical occupation coverage before
        // any expensive angular generation. Counting alone would miss errors.
        let mut covered = HashSet::new();
        for template in &mut templates {
            let expanded = expand_occupations(template, &label_set, &channels)?;
            template.multiplicity = expanded.len().try_into()?;
            for key in expanded {
                ensure!(
                    allowed.contains(&key),
                    "representative expansion adds an occupation excluded by the physical model"
                );
                ensure!(
                    covered.insert(key),
                    "overlapping CSFG occupation representatives"
                );
            }
        }
        ensure!(
            covered == allowed,
            "CSFG representatives do not cover all allowed occupations"
        );
        templates
            .sort_by_cached_key(|t| (t.kind != 1, std::cmp::Reverse(fortran_key(&t.occupations))));
        Ok(Self {
            core: enumerated.core_subshells,
            labels,
            peel,
            templates,
            unique_occupations: allowed.len(),
        })
    }
}

fn expand_occupations(
    t: &Template,
    labels: &HashSet<Subshell>,
    channels: &HashMap<i8, Channel>,
) -> Result<Vec<Signature>> {
    let positions = t
        .occupations
        .iter()
        .enumerate()
        .filter(|(_, o)| !labels.contains(&o.subshell))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if positions.is_empty() {
        return Ok(vec![signature(&t.occupations)]);
    }
    let a = positions[0];
    let ca = channels[&t.occupations[a].subshell.kappa()];
    let mut expanded = Vec::new();
    for na in ca.first..=ca.last {
        let mut o = t.occupations.clone();
        o[a].subshell = Subshell::new(na, ca.kappa)?;
        if positions.len() == 1 {
            expanded.push(signature(&o));
            continue;
        }
        let b = positions[1];
        let cb = channels[&o[b].subshell.kappa()];
        for nb in cb.first..=cb.last {
            if t.kind == 4 && na >= nb {
                continue;
            }
            o[b].subshell = Subshell::new(nb, cb.kappa)?;
            expanded.push(signature(&o));
        }
    }
    Ok(expanded)
}
