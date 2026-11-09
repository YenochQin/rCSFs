//! Printed JKVANT/SENIOR states projected from the shared physical-state table.
//! `None` means no printed seniority label, not seniority zero.

use crate::complete_csf::SubshellState;
use anyhow::Result;

pub(crate) fn subshell_states(two_j: u16, electrons: u8) -> Result<Vec<SubshellState>> {
    crate::seniority::printed_subshell_states(two_j, electrons)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `2j_max` values a supported subshell can ask for: `capacity = 2|kappa|`
    /// with `1 <= |kappa| <= 11`, so every odd value up to 21 is reachable and
    /// nothing else is.
    fn reachable_two_j_max() -> impl Iterator<Item = u16> {
        (1..=21).step_by(2)
    }

    fn population_of(two_j: u16, electrons: u8) -> u16 {
        let capacity = two_j + 1;
        u16::from(electrons).min(capacity - u16::from(electrons))
    }

    /// `subshell_states` is called with `2j_max + 1 == capacity`, and a
    /// supported subshell holds at most 22 electrons.
    fn capacity_of(two_j: u16) -> u8 {
        u8::try_from(two_j + 1).expect("a reachable 2j_max fits one byte")
    }

    /// The tables that GRASP defines outside the generic arms. A new arm is a
    /// change to the premise that the P6a uniqueness argument rests on, so it
    /// has to be added here deliberately.
    const EXOTIC_TABLES: [(u16, u16); 6] = [(5, 3), (7, 3), (7, 4), (9, 3), (9, 4), (9, 5)];

    /// Two states of one subshell must never carry the same `(2J, seniority)`
    /// pair: such a pair is indistinguishable in a V2 row, so a subshell whose
    /// table repeated one would emit genuinely duplicate rows.
    ///
    /// The same walk pins which tables exist at all, so the uniqueness argument
    /// cannot silently acquire a new table with a repeated state.
    #[test]
    fn state_tables_are_injective_and_cover_the_documented_set() {
        let mut tables = 0usize;
        let mut states = 0usize;
        for two_j in reachable_two_j_max() {
            for electrons in 1..=capacity_of(two_j) {
                let population = population_of(two_j, electrons);
                let expected = population <= 2 || EXOTIC_TABLES.contains(&(two_j, population));
                match subshell_states(two_j, electrons) {
                    Ok(states_in_table) => {
                        assert!(
                            expected,
                            "2j={two_j} with electron/hole occupation {population} \
                             ({electrons} electrons) gained a table; document it before \
                             relying on it being injective"
                        );
                        for (index, state) in states_in_table.iter().enumerate() {
                            for other in &states_in_table[index + 1..] {
                                assert_ne!(
                                    (state.two_j, state.seniority),
                                    (other.two_j, other.seniority),
                                    "2j={two_j} with {electrons} electrons repeats the state \
                                     {state:?}; two such states are one V2 row"
                                );
                            }
                        }
                        tables += 1;
                        states += states_in_table.len();
                    }
                    Err(error) => {
                        assert!(
                            !expected,
                            "2j={two_j} with electron/hole occupation {population} \
                             ({electrons} electrons) lost its table: {error}"
                        );
                    }
                }
            }
        }
        // The walk must have covered the real tables, not an empty range.
        assert!(tables >= 60, "only {tables} state tables were checked");
        assert!(states >= 200, "only {states} subshell states were checked");
    }
}
