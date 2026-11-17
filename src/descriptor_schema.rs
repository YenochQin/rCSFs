//! Descriptor transport contract plus producer-side record validation.
use crate::complete_csf::{IntermediateCoupling, OccupiedSubshell};
use crate::configuration_generation::shared_generation::{Subshell, subshell_states};
pub use crate::descriptor_transport::*;
use anyhow::{Context, Result, ensure};

/// Validate that `occupied`/`couplings` are a legal CSF record over `subshells`.
///
/// This checks legality against GRASP's physical state tables, not
/// generator reachability: an imported file may legally print a state the
/// generator's own visibility rule (`mod.rs`'s `FIRST`-flag suppression)
/// would never emit, and that must still validate. Checks performed:
///
/// - every occupied subshell index resolves into `subshells`;
/// - occupied subshells appear in ascending (Peel) order with no duplicates;
/// - `1 <= occupation <= capacity`;
/// - a printed `(2J, seniority)` state is a member of the subshell's legal
///   state table for that occupation;
/// - each coupling boundary satisfies `2 <= boundary < occupied.len()`;
/// - coupling boundaries are strictly increasing with no duplicates.
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
            let max_two_j = u16::from(capacity - 1);
            let legal_states = subshell_states(max_two_j, shell.occupation).with_context(|| {
                format!("no legal state table for {label:?}({})", shell.occupation)
            })?;
            ensure!(
                legal_states.contains(&state),
                "printed state {state:?} for {label:?}({}) is not a legal subshell state",
                shell.occupation
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
    fn v2_layout_adds_two_global_columns() {
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 3);
        assert_eq!(layout.feature_len(), 12);
        assert_eq!(layout.row_len(), 14);
        assert_eq!(layout.total_two_j_index(), Some(12));
        assert_eq!(layout.parity_index(), Some(13));
        assert_eq!(
            layout.feature_column_names(),
            [
                "sub0_n", "sub0_2j", "sub0_v", "sub0_2k", "sub1_n", "sub1_2j", "sub1_v", "sub1_2k",
                "sub2_n", "sub2_2j", "sub2_v", "sub2_2k",
            ]
        );
        assert_eq!(layout.global_column_names(), ["total_two_j", "parity"]);
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
    fn validate_record_rejects_illegal_seniority_for_state() {
        let subshells = vec!["4f".to_owned()];
        // 4f(4) 2J=4 is only legal with seniority 2 or 4; seniority 9 does not exist.
        let occupied = [OccupiedSubshell {
            subshell_index: 0,
            occupation: 4,
            state: Some(SubshellState {
                two_j: 4,
                seniority: Some(9),
            }),
        }];
        let error = validate_record(&subshells, &occupied, &[]).unwrap_err();
        assert!(error.to_string().contains("not a legal subshell state"));
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
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 2);
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
        assert_eq!(get("descriptor_version"), Some("2".to_owned()));
        assert_eq!(get("seniority_encoding"), Some("physical".to_owned()));
        assert_eq!(get("channels_per_subshell"), Some("4".to_owned()));
        assert_eq!(get("subshell_count"), Some("2".to_owned()));
        assert_eq!(get("peel_subshells"), Some("[\"4f-\",\"4f\"]".to_owned()));
        assert_eq!(get("missing_sentinel"), Some("-1".to_owned()));
        assert_eq!(get("normalized"), Some("false".to_owned()));
        assert_eq!(get("source_header_sha256"), Some("abc123".to_owned()));
        assert_eq!(
            get("source_header_filename"),
            Some("x_header.toml".to_owned())
        );
    }
}
