//! Descriptor transport contract plus producer-side record validation.
use crate::complete_csf::{IntermediateCoupling, OccupiedSubshell};
use crate::configuration_generation::Subshell;
pub use crate::descriptor_transport::*;
use crate::seniority::{physical_states, validate_physical_seniority};
use anyhow::{Context, Result, ensure};

pub(crate) const fn empty_slot(cumulative_code: i8) -> [i8; 4] {
    [0, 0, 1, cumulative_code]
}

/// Return physical total 2J and parity from an already validated row.
pub(crate) fn row_symmetry(row: &[i8]) -> (i8, i8) {
    (row[row.len() - 2] - 1, row[row.len() - 1])
}

pub(crate) fn valid_coupling(previous: u16, subshell: u16, cumulative: u16) -> bool {
    let lower = previous.abs_diff(subshell);
    (lower..=previous + subshell).contains(&cumulative) && (cumulative - lower).is_multiple_of(2)
}

/// Validate raw V3 channels with orbital labels resolved once per dataset.
pub(crate) struct DescriptorRowValidator {
    peel: Vec<Subshell>,
    core_electrons: usize,
}

impl DescriptorRowValidator {
    pub(crate) fn new(core: &[Subshell], peel: &[String]) -> Result<Self> {
        Ok(Self {
            peel: peel
                .iter()
                .map(|label| label.parse())
                .collect::<Result<_>>()?,
            core_electrons: core
                .iter()
                .map(|orbital| usize::from(orbital.capacity()))
                .sum(),
        })
    }

    pub(crate) fn validate(&self, row: &[i8], electron_count: &mut Option<usize>) -> Result<()> {
        let layout = DescriptorLayout::new(DescriptorVersion::V3, self.peel.len());
        ensure!(row.len() == layout.row_len(), "descriptor width mismatch");
        let total_j = decode_angular(row[row.len() - 2], "total 2J+1")?;
        let parity = row[row.len() - 1];
        let mut electrons = self.core_electrons;
        let mut parity_exponent = 0usize;
        let mut previous = 0;
        for (&slot, orbital) in row[..layout.feature_len()]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(&self.peel)
        {
            let [occupation, seniority, j_code, k_code] = slot;
            if occupation == 0 {
                ensure!(
                    slot == empty_slot(encode_angular(previous, "cumulative 2K")?),
                    "invalid empty subshell {orbital}: expected (0, 0, 1, previous cumulative code)"
                );
                continue;
            }
            let validate_slot = || -> Result<(u8, u16)> {
                let occupation = u8::try_from(occupation).context("negative occupation")?;
                let seniority = u8::try_from(seniority).context("negative physical seniority")?;
                let j = decode_angular(j_code, "subshell 2J+1")?;
                let k = decode_angular(k_code, "cumulative 2K+1")?;
                validate_physical_seniority(
                    u16::from(orbital.capacity() - 1),
                    occupation,
                    j,
                    seniority,
                )?;
                ensure!(
                    valid_coupling(previous, j, k),
                    "invalid angular momentum coupling from 2K={previous} through 2J={j} to 2K={k}"
                );
                Ok((occupation, k))
            };
            let (occupation, k) = validate_slot()
                .with_context(|| format!("invalid descriptor subshell {orbital}"))?;
            electrons += usize::from(occupation);
            parity_exponent += usize::from(occupation) * usize::from(orbital.l());
            previous = k;
        }
        ensure!(previous == total_j, "incorrect final angular momentum");
        ensure!(electrons > 0, "empty electron basis");
        ensure!(
            parity
                == if parity_exponent.is_multiple_of(2) {
                    1
                } else {
                    -1
                },
            "incorrect parity"
        );
        if let Some(expected) = *electron_count {
            ensure!(expected == electrons, "electron count mismatch");
        }
        *electron_count = Some(electrons);
        Ok(())
    }
}

/// Validate occupied indices, printed subshell state constraints and the
/// ordering of explicit coupling boundaries before resolving the chain.
pub fn validate_record(
    subshells: &[String],
    occupied: &[OccupiedSubshell],
    couplings: &[IntermediateCoupling],
) -> Result<()> {
    ensure!(
        !occupied.is_empty(),
        "a CSF record must occupy at least one subshell"
    );

    let mut previous_index: Option<u16> = None;
    for shell in occupied {
        let index = usize::from(shell.subshell_index);
        let label = subshells
            .get(index)
            .with_context(|| format!("occupied subshell index {index} exceeds peel table"))?;
        ensure!(
            previous_index.is_none_or(|previous| shell.subshell_index > previous),
            "occupied subshells are out of Peel order or duplicated at index {index}"
        );
        previous_index = Some(shell.subshell_index);

        let subshell: Subshell = label
            .parse()
            .with_context(|| format!("invalid peel subshell label {label:?}"))?;
        let capacity = subshell.capacity();
        ensure!(
            shell.occupation >= 1 && shell.occupation <= capacity,
            "occupation {} for {label:?} is outside 1..={capacity}",
            shell.occupation
        );
        if let Some(state) = shell.state {
            encode_angular(state.two_j, "subshell 2J")
                .with_context(|| format!("subshell {label}"))?;
            let allowed = physical_states(u16::from(capacity - 1), shell.occupation)?;
            ensure!(
                allowed
                    .iter()
                    .any(|candidate| candidate.matches_printed(state)),
                "invalid printed state {state:?} at subshell {label}"
            );
        }
    }

    let occupied_len = u16::try_from(occupied.len()).context("too many occupied subshells")?;
    let mut previous_boundary: Option<u16> = None;
    for coupling in couplings {
        ensure!(
            (2..occupied_len).contains(&coupling.boundary),
            "coupling boundary {} is invalid for {} occupied subshells",
            coupling.boundary,
            occupied.len()
        );
        ensure!(
            previous_boundary.is_none_or(|previous| coupling.boundary > previous),
            "coupling boundaries are out of order or duplicated at boundary {}",
            coupling.boundary
        );
        previous_boundary = Some(coupling.boundary);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complete_csf::SubshellState;

    #[test]
    fn validates_coupling_and_seniority_in_raw_v3_rows() {
        let peel = ["2s", "3s", "4s", "5s"].map(str::to_owned);
        let validator = DescriptorRowValidator::new(&[], &peel).unwrap();
        let row = [2, 0, 1, 1, 2, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1];
        let mut electron_count = None;
        validator.validate(&row, &mut electron_count).unwrap();
        assert_eq!(electron_count, Some(6));
        for (index, value, message) in [
            (1, 1, "physical seniority"),
            (3, 0, "code must be in 1..=127"),
            (7, 2, "invalid angular momentum coupling"),
            (10, 127, "illegal physical seniority"),
            (17, -1, "incorrect parity"),
        ] {
            let mut modified = row;
            modified[index] = value;
            let error = validator
                .validate(&modified, &mut electron_count)
                .unwrap_err();
            assert!(format!("{error:#}").contains(message), "{error:#}");
        }
    }

    #[test]
    fn v3_layout_adds_two_global_columns() {
        let layout = DescriptorLayout::new(DescriptorVersion::V3, 3);
        assert_eq!(layout.feature_len(), 12);
        assert_eq!(layout.row_len(), 14);
        assert_eq!(layout.total_two_j_index(), Some(12));
        assert_eq!(layout.parity_index(), Some(13));
        assert_eq!(
            layout.feature_column_names(),
            [
                "sub0_occupation",
                "sub0_seniority",
                "sub0_two_j_plus_one",
                "sub0_cumulative_two_k_plus_one",
                "sub1_occupation",
                "sub1_seniority",
                "sub1_two_j_plus_one",
                "sub1_cumulative_two_k_plus_one",
                "sub2_occupation",
                "sub2_seniority",
                "sub2_two_j_plus_one",
                "sub2_cumulative_two_k_plus_one",
            ]
        );
        assert_eq!(
            layout.global_column_names(),
            ["total_two_j_plus_one", "parity"]
        );
    }

    #[test]
    fn validate_record_accepts_a_legal_occupied_subshell() {
        let subshells = vec!["4f".to_owned()];
        let occupied = [OccupiedSubshell {
            subshell_index: 0,
            occupation: 4,
            state: Some(SubshellState {
                two_j: 4,
                seniority: Some(2),
            }),
        }];
        validate_record(&subshells, &occupied, &[]).unwrap();
    }

    #[test]
    fn validate_record_rejects_occupation_above_capacity() {
        let subshells = vec!["4f".to_owned()];
        let occupied = [OccupiedSubshell {
            subshell_index: 0,
            occupation: 20,
            state: None,
        }];
        let error = validate_record(&subshells, &occupied, &[]).unwrap_err();
        assert!(error.to_string().contains("outside 1"));
    }

    #[test]
    fn validate_record_rejects_out_of_order_subshells() {
        let subshells = vec!["4f-".to_owned(), "4f".to_owned()];
        let occupied = [
            OccupiedSubshell {
                subshell_index: 1,
                occupation: 4,
                state: None,
            },
            OccupiedSubshell {
                subshell_index: 0,
                occupation: 3,
                state: None,
            },
        ];
        let error = validate_record(&subshells, &occupied, &[]).unwrap_err();
        assert!(error.to_string().contains("out of Peel order"));
    }

    #[test]
    fn validate_record_rejects_duplicate_subshell() {
        let subshells = vec!["4f".to_owned()];
        let occupied = [
            OccupiedSubshell {
                subshell_index: 0,
                occupation: 2,
                state: None,
            },
            OccupiedSubshell {
                subshell_index: 0,
                occupation: 2,
                state: None,
            },
        ];
        let error = validate_record(&subshells, &occupied, &[]).unwrap_err();
        assert!(error.to_string().contains("out of Peel order"));
    }

    #[test]
    fn validate_record_rejects_boundary_zero_one_or_last() {
        let subshells = vec!["1s".to_owned(), "2s".to_owned(), "3s".to_owned()];
        let occupied = [
            OccupiedSubshell {
                subshell_index: 0,
                occupation: 1,
                state: None,
            },
            OccupiedSubshell {
                subshell_index: 1,
                occupation: 1,
                state: None,
            },
            OccupiedSubshell {
                subshell_index: 2,
                occupation: 1,
                state: None,
            },
        ];
        for boundary in [0, 1, 3] {
            let couplings = [IntermediateCoupling { boundary, two_j: 0 }];
            let error = validate_record(&subshells, &occupied, &couplings).unwrap_err();
            assert!(error.to_string().contains("coupling boundary"));
        }
    }

    #[test]
    fn validate_record_rejects_nonincreasing_boundaries() {
        let subshells = vec![
            "1s".to_owned(),
            "2s".to_owned(),
            "3s".to_owned(),
            "4s".to_owned(),
        ];
        let occupied: Vec<_> = (0..4)
            .map(|index| OccupiedSubshell {
                subshell_index: index,
                occupation: 1,
                state: None,
            })
            .collect();
        let couplings = [
            IntermediateCoupling {
                boundary: 2,
                two_j: 0,
            },
            IntermediateCoupling {
                boundary: 2,
                two_j: 0,
            },
        ];
        let error = validate_record(&subshells, &occupied, &couplings).unwrap_err();
        assert!(error.to_string().contains("out of order or duplicated"));
    }

    #[test]
    fn kv_metadata_round_trips_expected_keys() {
        let layout = DescriptorLayout::new(DescriptorVersion::V3, 2);
        let peel_subshells = vec!["4f-".to_owned(), "4f".to_owned()];
        let entries = output_kv_metadata(
            layout,
            &peel_subshells,
            Some("abc123"),
            Some("x_header.toml"),
        );
        let get = |key: &str| {
            entries
                .iter()
                .find(|entry| entry.key == key)
                .and_then(|entry| entry.value.clone())
        };
        assert_eq!(get("descriptor_version"), Some("3".to_owned()));
        assert_eq!(get("seniority_encoding"), Some("physical".to_owned()));
        assert_eq!(get("channels_per_subshell"), Some("4".to_owned()));
        assert_eq!(get("subshell_count"), Some("2".to_owned()));
        assert_eq!(get("peel_subshells"), Some("[\"4f-\",\"4f\"]".to_owned()));
        assert_eq!(get("missing_sentinel"), None);
        assert_eq!(get("normalized"), Some("false".to_owned()));
        assert_eq!(get("source_header_sha256"), Some("abc123".to_owned()));
        assert_eq!(
            get("source_header_filename"),
            Some("x_header.toml".to_owned())
        );
    }
}
