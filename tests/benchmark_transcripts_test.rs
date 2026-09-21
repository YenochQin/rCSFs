//! Registered benchmark transcripts must be reproducible by hash.
//!
//! `tests/fixtures/transcripts.toml` binds each benchmark input to its exact
//! bytes and to the counts registered in `docs/benchmarks`.  A measurement is
//! only comparable with a recorded baseline while both sides of the comparison
//! still describe the same input, so a drifted fixture has to fail here rather
//! than quietly produce a different workload.
//!
//! This test covers the cheap half of the contract: the SHA-256 of the input
//! bytes and the number of enumerated occupation configurations.  The recorded
//! CSF totals are checked by `scripts/benchmark_v2_generation.py` against a
//! real run, because generating millions of records does not belong in the
//! default test suite.

use _rcsfs::csf_generation::{ExcitationRequest, enumerate_occupations};
use sha2::{Digest, Sha256};

const MANIFEST: &str = include_str!("fixtures/transcripts.toml");

/// Every fixture the manifest may reference, compiled in and also checked
/// against the file on disk so a build-time snapshot cannot hide a drift.
const FIXTURES: [(&str, &str); 2] = [
    (
        "b1_cc1_5spdfg_3exc.rcsfgenerate",
        include_str!("fixtures/b1_cc1_5spdfg_3exc.rcsfgenerate"),
    ),
    (
        "b2_cc1_fullas_2exc.rcsfgenerate",
        include_str!("fixtures/b2_cc1_fullas_2exc.rcsfgenerate"),
    ),
];

fn fixture(name: &str) -> &'static str {
    FIXTURES
        .iter()
        .find(|(file, _)| *file == name)
        .map(|(_, text)| *text)
        .unwrap_or_else(|| panic!("manifest references uncompiled fixture {name}"))
}

fn entry_integer(entry: &toml::Value, key: &str, file: &str) -> i64 {
    entry
        .get(key)
        .and_then(toml::Value::as_integer)
        .unwrap_or_else(|| panic!("manifest entry {file} is missing the integer {key}"))
}

#[test]
fn registered_benchmark_transcripts_match_their_manifest() {
    let manifest: toml::Value = toml::from_str(MANIFEST).expect("manifest is valid TOML");
    assert_eq!(
        manifest.get("version").and_then(toml::Value::as_integer),
        Some(1),
        "unexpected manifest version"
    );
    let entries = manifest
        .get("transcript")
        .and_then(toml::Value::as_array)
        .expect("manifest has a transcript table");
    assert_eq!(entries.len(), FIXTURES.len());

    let mut names = Vec::new();
    for entry in entries {
        let file = entry
            .get("file")
            .and_then(toml::Value::as_str)
            .expect("manifest entry has a file name");
        let name = entry
            .get("name")
            .and_then(toml::Value::as_str)
            .expect("manifest entry has a name");
        assert!(names.iter().all(|existing| *existing != name));
        names.push(name);

        // The fixture must exist and hash to the registered value.
        let on_disk = std::fs::read_to_string(format!("tests/fixtures/{file}"))
            .unwrap_or_else(|error| panic!("cannot read tests/fixtures/{file}: {error}"));
        assert_eq!(
            on_disk,
            fixture(file),
            "{file} differs from the text compiled into this test"
        );
        let digest = format!("{:x}", Sha256::digest(on_disk.as_bytes()));
        assert_eq!(
            Some(digest.as_str()),
            entry.get("sha256").and_then(toml::Value::as_str),
            "{file} drifted from its registered hash; re-register the fixture and its baseline together"
        );

        // The cheap structural counts must still hold.
        let request = ExcitationRequest::from_transcript(&on_disk)
            .unwrap_or_else(|error| panic!("{file} is not a valid transcript: {error:#}"));
        assert_eq!(request.min_two_j, 8);
        assert_eq!(request.max_two_j, 8);
        let occupations = enumerate_occupations(&request)
            .unwrap_or_else(|error| panic!("{file} failed to enumerate: {error:#}"));
        let registered = usize::try_from(entry_integer(entry, "unique_occupations", file)).unwrap();
        assert_eq!(
            occupations.configurations.len(),
            registered,
            "{file} enumerated a different number of configurations"
        );
        // The expensive totals are consumed by the benchmark script, but an
        // entry without them cannot serve as a baseline at all.
        assert!(entry_integer(entry, "records", file) > 0);
    }
}
