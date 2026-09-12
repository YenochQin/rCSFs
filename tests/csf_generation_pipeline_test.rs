//! Regression test for the shared transcript-to-file pipeline extracted from
//! `examples/generate_transcript_csfs.rs` into `csf_generation::pipeline`.
//! This is the same function backing the `rcsfs.generate_csfs_from_transcript`
//! PyO3 binding, so a correct result here means the Python binding's
//! generation logic is correct too.

use _rcsfs::csf_generation::generate_csfs_from_transcript;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn transcript_pipeline_matches_registered_e1_cc1as1() {
    let transcript = fs::read_to_string("tests/fixtures/e1_cc1as1.rcsfgenerate")
        .expect("fixture must be readable");
    let dir = std::env::temp_dir().join(format!(
        "rcsfs-pipeline-test-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).expect("failed to create temp dir");
    let output = dir.join("out.c");

    let stats = generate_csfs_from_transcript(&transcript, &output, None, None, false)
        .expect("generation must succeed for the registered fixture");

    assert_eq!(stats.record_count, 452_373);
    assert_eq!(stats.block_count, 7);
    assert!(stats.output_bytes > 0);
    assert!(stats.descriptor_count.is_none());
    assert!(output.exists());

    fs::remove_dir_all(&dir).ok();
}
