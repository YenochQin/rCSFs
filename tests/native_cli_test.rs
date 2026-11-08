//! Exercise the installed-style executable with no Python command available.
#![cfg(feature = "cli")]
use _rcsfs::complete_csf::CompleteCsfFile;
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

const MODEL: &str = "inactive_core = 0\nreference_configuration = [\"1s(2,*)\"]\nactive_space = \"2s\"\nj_min = 0\nj_max = 0\nexcitations = 2\nthreads = 1\n";
const TRANSCRIPT: &str = "* ! Orbital order\n0\n1s(2,*)\n\n2s\n0,0\n2\nn\n";
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
fn invoke(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsfs"))
        .args(args)
        .current_dir(dir)
        .env("PATH", "")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .output()
        .unwrap()
}
fn success(dir: &Path, args: &[&str]) -> Output {
    let output = invoke(dir, args);
    assert!(
        output.status.success(),
        "{args:?}: {}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    output
}
fn json(dir: &Path, args: &[&str]) -> Value {
    let output = success(dir, args);
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&output.stdout)))
}
fn prepare_descriptor(dir: &Path) {
    fs::write(dir.join("input.log"), TRANSCRIPT).unwrap();
    success(
        dir,
        &[
            "csfsgenerate",
            "input.c",
            "--transcript",
            "input.log",
            "--threads",
            "1",
        ],
    );
    json(
        dir,
        &[
            "convert",
            "input.c",
            "input.parquet",
            "--num-workers",
            "1",
            "--json",
        ],
    );
    json(
        dir,
        &[
            "gen-descriptors",
            "input.parquet",
            "features.parquet",
            "--header",
            "input_header.toml",
            "--num-workers",
            "1",
            "--json",
        ],
    );
}
#[test]
fn help_version_and_each_command_run_without_python() {
    let dir = tempfile::tempdir().unwrap();
    assert!(
        String::from_utf8(success(dir.path(), &["--version"]).stdout)
            .unwrap()
            .contains("native Rust")
    );
    for command in [
        "convert",
        "info",
        "csfsgenerate",
        "csfggenerate",
        "gen-descriptors",
        "zero-first",
        "csfs-split",
        "jsplit",
        "interacting",
        "restore-csfs",
    ] {
        success(dir.path(), &[command, "--help"]);
        success(dir.path(), &[command, "init-config"]);
    }
    let contents = fs::read_to_string(dir.path().join("rcsfs.toml")).unwrap();
    assert!(toml::from_str::<toml::Table>(&contents).unwrap().is_empty());
    success(dir.path(), &["csfsgenerate", "init-config"]);
    assert_eq!(
        contents,
        fs::read_to_string(dir.path().join("rcsfs.toml")).unwrap()
    );
}
#[test]
fn descriptor_roundtrip_preserves_states_headers_and_selected_rows() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    prepare_descriptor(p);
    let info = json(p, &["info", "features.parquet", "--json"]);
    assert_eq!(info["num_rows"], 3);
    let meta: toml::Table =
        toml::from_str(&fs::read_to_string(p.join("features.toml")).unwrap()).unwrap();
    assert_eq!(meta["format_version"].as_integer(), Some(2));
    assert_eq!(meta["normalized"].as_bool(), Some(false));
    fs::write(p.join("restored.c"), "previous complete output").unwrap();
    json(
        p,
        &[
            "restore-csfs",
            "--descriptors",
            "features.parquet",
            "--header",
            "input_header.toml",
            "--output",
            "restored.c",
            "--json",
        ],
    );
    assert_eq!(
        CompleteCsfFile::parse_path(&p.join("input.c")).unwrap(),
        CompleteCsfFile::parse_path(&p.join("restored.c")).unwrap()
    );
    let stats = json(
        p,
        &[
            "restore-csfs",
            "--descriptors",
            "features.parquet",
            "--header",
            "input_header.toml",
            "--output",
            "selected.c",
            "--indices",
            "1",
            "--json",
        ],
    );
    assert_eq!(stats["record_count"], 1);
    let selected = CompleteCsfFile::parse_path(&p.join("selected.c")).unwrap();
    assert_eq!(selected.records[0].total_two_j, 0);
}
#[test]
fn invalid_header_and_path_aliases_preserve_existing_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    prepare_descriptor(p);
    let before = fs::read(p.join("input.parquet")).unwrap();
    let bad = invoke(
        p,
        &[
            "gen-descriptors",
            "input.parquet",
            "input.parquet",
            "--header",
            "input_header.toml",
            "--json",
        ],
    );
    assert!(!bad.status.success());
    assert_eq!(before, fs::read(p.join("input.parquet")).unwrap());
    fs::write(p.join("saved.c"), "keep me").unwrap();
    let mut header = fs::read_to_string(p.join("input_header.toml")).unwrap();
    header.push_str("\n# changed header\n");
    fs::write(p.join("wrong.toml"), header).unwrap();
    let bad = invoke(
        p,
        &[
            "restore-csfs",
            "--descriptors",
            "features.parquet",
            "--header",
            "wrong.toml",
            "--output",
            "saved.c",
            "--json",
        ],
    );
    assert!(!bad.status.success());
    assert_eq!(fs::read_to_string(p.join("saved.c")).unwrap(), "keep me");
    assert_eq!(
        serde_json::from_slice::<Value>(&bad.stdout).unwrap()["success"],
        false
    );
    assert!(!fs::read_dir(p).unwrap().any(|e| {
        e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("rcsfs-native-")
    }));
}
#[test]
fn generation_memory_disk_and_transcript_outputs_agree() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(p.join("input.log"), TRANSCRIPT).unwrap();
    let baseline = json(
        p,
        &[
            "csfsgenerate",
            "memory.c",
            "--transcript",
            "input.log",
            "--threads",
            "1",
            "--json",
        ],
    );
    fs::write(p.join("model.toml"), format!("[csfsgenerate]\n{MODEL}rcsfs_out = 'disk.c'\ngenerate_descriptors = true\ngeneration_storage = 'disk'\nmemory_budget_mib = 32\n")).unwrap();
    let disk = json(p, &["csfsgenerate", "-c", "model.toml", "--json"]);
    assert_eq!(baseline["record_count"], disk["record_count"]);
    assert_eq!(
        fs::read(p.join("memory.c")).unwrap(),
        fs::read(p.join("disk.c")).unwrap()
    );
    json(
        p,
        &[
            "restore-csfs",
            "--descriptors",
            "disk_descriptors.parquet",
            "--header",
            "disk_header.toml",
            "--output",
            "roundtrip.c",
            "--json",
        ],
    );
    assert_eq!(
        fs::read(p.join("memory.c")).unwrap(),
        fs::read(p.join("roundtrip.c")).unwrap()
    );
}
#[test]
fn estimate_creates_no_outputs_and_cli_overrides_config() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(
        p.join("model.toml"),
        format!("[csfsgenerate]\n{MODEL}rcsfs_out = 'unused.c'\ngeneration_storage = 'disk'\n"),
    )
    .unwrap();
    let stats = json(
        p,
        &[
            "csfsgenerate",
            "-c",
            "model.toml",
            "--estimate-only",
            "--json",
            "--threads",
            "2",
        ],
    );
    assert_eq!(stats["estimate_only"], true);
    assert!(stats["pre_deduplication_records"].as_u64().unwrap() > 0);
    assert_eq!(fs::read_dir(p).unwrap().count(), 1);
    success(
        p,
        &[
            "csfsgenerate",
            "chosen.c",
            "-c",
            "model.toml",
            "--generation-storage",
            "memory",
        ],
    );
    assert!(p.join("chosen.c").exists());
    assert!(!p.join("unused.c").exists());
}
#[test]
fn batch_generation_split_uses_conf_naming_and_writes_log() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(p.join("model.toml"), format!("conf = 'helium'\n[csfsgenerate]\nas = 2\n{MODEL}\n[csfs-split.active_spaces]\nAS1 = '1s'\n")).unwrap();
    success(p, &["-c", "model.toml"]);
    assert!(p.join("helium_as2raw.parquet").exists());
    assert!(p.join("helium_as1raw.c").exists());
    let log = fs::read_to_string(p.join("rcsfs_helium.log")).unwrap();
    let records: Vec<Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.last().unwrap()["event"], "complete");
    assert_eq!(
        CompleteCsfFile::parse_path(&p.join("helium_as1raw.c"))
            .unwrap()
            .records
            .len(),
        1
    );
}
#[test]
fn batch_is_validated_before_any_generation() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(p.join("bad.toml"), format!("[[csfsgenerate]]\n{MODEL}rcsfs_out = 'same.c'\n[[csfsgenerate]]\n{MODEL}rcsfs_out = 'same.c'\n")).unwrap();
    let bad = invoke(p, &["-c", "bad.toml"]);
    assert_eq!(bad.status.code(), Some(2));
    assert_eq!(fs::read_dir(p).unwrap().count(), 1);
    fs::write(
        p.join("bad.toml"),
        format!("[csfsgenerate]\n{MODEL}rcsfs_out = 'out.c'\n[gen-descriptors]\nunknown = true\n"),
    )
    .unwrap();
    let bad = invoke(p, &["-c", "bad.toml"]);
    assert_eq!(bad.status.code(), Some(2));
    assert!(!p.join("out.c").exists());
}
#[test]
fn continuation_lists_are_merged_in_disk_mode() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    let list = "reference_configuration = ['1s(2,*)']\nactive_space = '2s'\nj_min = 0\nj_max = 0\nexcitations = 2\n";
    fs::write(p.join("model.toml"), format!("[csfsgenerate]\ninactive_core = 0\nthreads = 1\nrcsfs_out = 'union.c'\n[[csfsgenerate.lists]]\n{list}[[csfsgenerate.lists]]\n{list}")).unwrap();
    let stats = json(p, &["csfsgenerate", "-c", "model.toml", "--json"]);
    assert!(stats["duplicate_count"].as_u64().unwrap() > 0);
    assert_eq!(
        stats["generated_count"].as_u64().unwrap(),
        stats["record_count"].as_u64().unwrap() * 2
    );
}
#[test]
fn csfg_matches_independent_fixture_and_protects_existing_files() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(p.join("model.toml"), "[csfggenerate]\ninactive_core = 0\nreference_configuration = ['1s(2,*)']\nlabeling_space = '1s'\nactive_space = '3s,3p'\nj_min = 0\nj_max = 4\nexcitations = 2\nthreads = 1\n").unwrap();
    let stats = json(p, &["csfggenerate", "-c", "model.toml", "--json"]);
    assert_eq!(stats["record_count"], 16);
    assert_eq!(
        fs::read(p.join("rcsfg.out")).unwrap(),
        fs::read(fixture("csfg/rcsfg.txt")).unwrap()
    );
    assert_eq!(
        fs::read(p.join("rlabel.out")).unwrap(),
        fs::read(fixture("csfg/rlabel.txt")).unwrap()
    );
    assert!(
        !invoke(p, &["csfggenerate", "-c", "model.toml"])
            .status
            .success()
    );
    success(p, &["csfggenerate", "-c", "model.toml", "--overwrite"]);
}
#[test]
fn csfg_numbered_layers_are_published_only_after_all_models_succeed() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    let model = "conf = 'he'\n[[csfggenerate]]\nas = 2\ninactive_core = 0\nreference_configuration = ['1s(2,*)']\nlabeling_space = '1s'\nactive_space = '3s,3p'\nj_min = 0\nj_max = 4\nexcitations = 2\nthreads = 1\n[csfg-split.active_spaces]\nAS1 = '2s,2p'\n";
    fs::write(p.join("model.toml"), model).unwrap();
    success(p, &["-c", "model.toml"]);
    for name in ["he_as2raw.g", "he_as2raw.l", "he_as1raw.g", "he_as1raw.l"] {
        assert!(p.join(name).exists());
    }
    let failing = model.replace("'he'", "'bad'").replace("'2s,2p'", "'2p'");
    fs::write(p.join("bad.toml"), failing).unwrap();
    assert!(!invoke(p, &["-c", "bad.toml"]).status.success());
    assert!(!p.join("bad_as2raw.g").exists());
}
#[test]
fn jsplit_copies_orbitals_and_aliases_accept_config() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    let input = fixture("multi_j.c");
    let stats = json(
        p,
        &[
            "split-j",
            input.to_str().unwrap(),
            "--prefix",
            "block",
            "--json",
        ],
    );
    assert_eq!(stats["block_count"], 3);
    for block in stats["outputs"].as_array().unwrap() {
        assert_eq!(
            fs::read(p.join(block["w_file"].as_str().unwrap())).unwrap(),
            fs::read(fixture("multi_j.w")).unwrap()
        );
    }
    fs::write(
        p.join("config.toml"),
        format!(
            "[jsplit]\ninput_csf = {:?}\nprefix = 'nocopy'\ncopy_w = false\n",
            input.to_str().unwrap()
        ),
    )
    .unwrap();
    let stats = json(p, &["rasfsplit", "-c", "config.toml", "--json"]);
    assert!(
        stats["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|b| b["w_file"].is_null())
    );
}
#[test]
fn zero_first_and_interaction_use_shared_algorithms() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::copy(fixture("complete.csf"), p.join("reference.c")).unwrap();
    fs::copy(fixture("complete.csf"), p.join("full.c")).unwrap();
    let stats = json(
        p,
        &[
            "zero-first",
            "reference.c",
            "full.c",
            "zf.c",
            "--num-workers",
            "1",
            "--json",
        ],
    );
    assert_eq!(stats["first_order_count"], 0);
    assert_eq!(
        CompleteCsfFile::parse_path(&p.join("full.c")).unwrap(),
        CompleteCsfFile::parse_path(&p.join("zf.c")).unwrap()
    );
    let stats = json(
        p,
        &[
            "interacting",
            "reference.c",
            "full.c",
            "-o",
            "selected.c",
            "--threads",
            "1",
            "--json",
        ],
    );
    assert_eq!(stats["exact"], false);
    assert_eq!(stats["hamiltonian"], "dirac_coulomb");
    assert_eq!(stats["exact_reference_skipped"], 2);
}
#[test]
fn malformed_arguments_and_output_config_collision_are_errors() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    for args in [
        vec!["unknown"],
        vec!["jsplit", "missing.c", "--bad"],
        vec!["csfsgenerate", "--threads", "0"],
        vec![
            "gen-descriptors",
            "in.parquet",
            "out.parquet",
            "--header",
            "header.toml",
            "--descriptor-version",
            "1",
        ],
    ] {
        assert_eq!(invoke(p, &args).status.code(), Some(2), "{args:?}");
    }
    fs::write(
        p.join("model.toml"),
        format!("[csfsgenerate]\n{MODEL}rcsfs_out = 'model.toml'\n"),
    )
    .unwrap();
    let before = fs::read(p.join("model.toml")).unwrap();
    assert!(
        !invoke(p, &["csfsgenerate", "-c", "model.toml"])
            .status
            .success()
    );
    assert_eq!(before, fs::read(p.join("model.toml")).unwrap());
}
#[test]
fn interactive_generation_saves_reusable_configuration() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsfs"))
        .args(["csfsgenerate", "interactive.c", "--threads", "1", "--json"])
        .current_dir(p)
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"*\n0\n1s(2,*)\n\n2s\n0,0\n2\nn\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stats: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(stats["success"], true);
    success(p, &["csfsgenerate", "repeat.c", "-c"]);
    assert_eq!(
        fs::read(p.join("interactive.c")).unwrap(),
        fs::read(p.join("repeat.c")).unwrap()
    );
}
