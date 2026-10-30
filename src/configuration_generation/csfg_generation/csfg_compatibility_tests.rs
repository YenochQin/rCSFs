use super::*;
use anyhow::Result;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "rcsfs-csfg-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn generate(&self, request: &CsfgRequest, threads: usize) -> Result<CsfgStats> {
        generate_csfgs(
            request,
            &self.0.join("rcsfg.out"),
            &self.0.join("rlabel.out"),
            Some(threads),
        )
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn request() -> CsfgRequest {
    CsfgRequest {
        inactive_core: 0,
        reference_configuration: vec!["1s(2,*)".into()],
        labeling_space: "1s".into(),
        active_space: "3s,3p".into(),
        j_min: 0,
        j_max: 4,
        excitations: 2,
    }
}
#[test]
fn csfg_all_five_types_match_independent_graspg_fixture() {
    let work = Work::new();
    let stats = work.generate(&request(), 1).unwrap();
    assert_eq!(
        fs::read_to_string(work.0.join("rcsfg.out")).unwrap(),
        include_str!("../../../tests/fixtures/csfg/rcsfg.txt")
    );
    assert_eq!(
        fs::read_to_string(work.0.join("rlabel.out")).unwrap(),
        include_str!("../../../tests/fixtures/csfg/rlabel.txt")
    );
    assert_eq!(stats.record_count, 16);
    assert!(stats.type_counts.iter().all(|&count| count > 0));
    assert_eq!(stats.label_csf_count, 1);
    assert_eq!(stats.expanded_csf_count, 28);
}
#[test]
fn csfg_parallel_and_duplicate_references_preserve_exact_order() {
    let a = Work::new();
    let b = Work::new();
    let mut req = request();
    a.generate(&req, 1).unwrap();
    req.reference_configuration
        .push(req.reference_configuration[0].clone());
    b.generate(&req, 2).unwrap();
    for name in ["rcsfg.out", "rlabel.out"] {
        assert_eq!(
            fs::read(a.0.join(name)).unwrap(),
            fs::read(b.0.join(name)).unwrap()
        );
    }
}
#[test]
fn csfg_single_radial_orbital_has_no_distinct_radial_pair() {
    let work = Work::new();
    let mut req = request();
    req.active_space = "2s,2p".into();
    let stats = work.generate(&req, 1).unwrap();
    assert_eq!(stats.type_counts[3], 0);
    assert_eq!(stats.expanded_csf_count, stats.record_count);
}
#[test]
fn csfg_core_is_in_label_count_and_prefix() {
    let work = Work::new();
    let mut req = request();
    req.inactive_core = 1;
    req.reference_configuration = vec!["2s(2,*)".into()];
    req.labeling_space = "2s".into();
    work.generate(&req, 1).unwrap();
    assert_eq!(
        fs::read_to_string(work.0.join("rlabel.out")).unwrap(),
        "           2   = nonsym\n  1s   2s\n"
    );
    let text = fs::read_to_string(work.0.join("rcsfg.out")).unwrap();
    assert!(
        text.starts_with("Core subshells:\n  1s\nPeel subshells:\n  2s   3s   2p-  3p-  2p   3p\n")
    );
}
#[test]
fn csfg_rejects_unsupported_models_without_outputs() {
    for (reference, label, excitations) in [
        ("1s(2,*)", "1s", 3),
        ("1λ(2,*)", "1s", 2),
        ("2s(2,*)", "1s", 2),
        ("1s(2,*)2s(0,i)", "1s", 2),
        ("1s(2,*)2s(0,d)", "2s", 2),
    ] {
        let work = Work::new();
        let mut req = request();
        req.reference_configuration = vec![reference.into()];
        req.labeling_space = label.into();
        req.excitations = excitations;
        assert!(work.generate(&req, 1).is_err());
        assert_eq!(fs::read_dir(&work.0).unwrap().count(), 0);
    }
}
#[test]
fn csfg_zero_excitation_and_output_collision() {
    let work = Work::new();
    let mut req = request();
    req.excitations = 0;
    req.active_space = "1s".into();
    let stats = work.generate(&req, 1).unwrap();
    assert_eq!(stats.record_count, 1);
    let before = fs::read(work.0.join("rcsfg.out")).unwrap();
    assert!(work.generate(&req, 1).is_err());
    assert_eq!(before, fs::read(work.0.join("rcsfg.out")).unwrap());
}

#[test]
fn csfg_preserves_minimum_occupation_and_distinct_references() {
    let work = Work::new();
    let mut req = request();
    req.reference_configuration = vec!["1s(2,1)".into()];
    let stats = work.generate(&req, 2).unwrap();
    assert_eq!(stats.expanded_csf_count, 5);
    assert!(stats.type_counts[2..].iter().all(|&n| n == 0));
    let work = Work::new();
    req.reference_configuration = vec!["1s(2,*)".into(), "2s(2,*)".into()];
    req.labeling_space = "2s".into();
    req.excitations = 0;
    let stats = work.generate(&req, 2).unwrap();
    assert_eq!(stats.label_csf_count, 2);
    assert_eq!(stats.expanded_csf_count, 2);
}

#[test]
fn csfg_rejects_inconsistent_closed_core_between_references() {
    let work = Work::new();
    let mut req = request();
    req.labeling_space = "2s".into();
    req.reference_configuration = vec!["1s(2,c)2s(2,*)".into(), "1s(2,*)2s(2,*)".into()];
    let error = work.generate(&req, 1).unwrap_err();
    assert!(error.to_string().contains("core"), "{error:#}");
    assert_eq!(fs::read_dir(&work.0).unwrap().count(), 0);
}
