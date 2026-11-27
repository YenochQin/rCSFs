//! Spectroscopic labels shared by CSF text readers and generation inputs.

use anyhow::{Context, Result, ensure};

pub(crate) const ORBITAL_LETTERS: &str = "spdfghiklmn";

/// Highest principal quantum number supported by rCSFs generation and V3 descriptor processing.
pub const MAX_PRINCIPAL_QUANTUM_NUMBER: u8 = 30;

/// Parse a physical nonrelativistic orbital before applying any generator limits.
pub(crate) fn parse_nl(label: &str) -> Result<(u32, u8)> {
    let digit_count = label.bytes().take_while(u8::is_ascii_digit).count();
    let (digits, letter) = label.split_at(digit_count);
    ensure!(
        !digits.is_empty() && !digits.starts_with('0'),
        "invalid principal quantum number in {label:?}"
    );
    let l = ORBITAL_LETTERS.bytes().position(|symbol| letter.as_bytes() == [symbol])
        .with_context(|| format!("invalid orbital label {label:?}: expected an ASCII orbital letter from {ORBITAL_LETTERS}"))? as u8;
    let n: u32 = digits
        .parse()
        .with_context(|| format!("invalid principal quantum number in {label:?}"))?;
    ensure!(
        u32::from(l) < n,
        "invalid orbital {label:?}: l={l} must be less than n={n}"
    );
    Ok((n, l))
}

/// Parse a relativistic label into the principal quantum number and Dirac kappa.
pub(crate) fn parse_subshell(label: &str) -> Result<(u32, i8)> {
    let (orbital, lower) = label
        .strip_suffix('-')
        .map_or((label, false), |s| (s, true));
    let (n, l) = parse_nl(orbital)?;
    ensure!(
        !lower || l != 0,
        "invalid subshell {label:?}: an s orbital has only j=1/2"
    );
    let l = l as i8;
    Ok((n, if lower { l } else { -l - 1 }))
}
