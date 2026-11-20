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
            "--output",
            "features_descriptors.parquet",
            "--transcript",
            "input.log",
            "--threads",
            "1",
        ],
    );
}
fn restore(dir: &Path, descriptor: &str, header: &str, output: &str) -> Value {
    json(
        dir,
        &[
            "restore-csfs",
            "--descriptors",
            descriptor,
            "--header",
            header,
            "--output",
            output,
            "--json",
        ],
    )
}
#[test]
fn output_paths_accept_current_directory_and_reject_other_directories() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(p.join("input.log"), TRANSCRIPT).unwrap();
    let absolute = p.join("absolute_descriptors.parquet");
    let canonical = p
        .canonicalize()
        .unwrap()
        .join("canonical_descriptors.parquet");
    for output in [
        "relative_descriptors.parquet",
        "./explicit_descriptors.parquet",
        absolute.to_str().unwrap(),
        canonical.to_str().unwrap(),
    ] {
        success(
            p,
            &[
                "csfsgenerate",
                "--output",
                output,
                "--transcript",
                "input.log",
                "--threads",
                "1",
            ],
        );
        assert!(p.join(output).is_file());
    }
    fs::create_dir(p.join("child")).unwrap();
    let outside = tempfile::tempdir().unwrap();
    let external = outside.path().join("outside_descriptors.parquet");
    for output in [
        "child/nested_descriptors.parquet",
        external.to_str().unwrap(),
    ] {
        let result = invoke(
            p,
            &[
                "csfsgenerate",
                "--output",
                output,
                "--transcript",
                "input.log",
            ],
        );
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr)
                .contains("output must name a file in the current directory")
        );
        assert!(!p.join(output).exists());
    }
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
    let info = json(p, &["info", "features_descriptors.parquet", "--json"]);
    assert_eq!(info["num_rows"], 3);
    assert_eq!(info["key_value_metadata"]["descriptor_version"], "2");
    assert!(!p.join("input.c").exists());
    fs::write(p.join("restored.c"), "previous complete output").unwrap();
    restore(
        p,
        "features_descriptors.parquet",
        "features_header.toml",
        "restored.c",
    );
    let imported = json(
        p,
        &[
            "convert",
            "restored.c",
            "copy_descriptors.parquet",
            "--json",
        ],
    );
    assert_eq!(imported["block_lengths"], serde_json::json!([3]));
    restore(p, "copy_descriptors.parquet", "copy_header.toml", "copy.c");
    assert_eq!(
        CompleteCsfFile::parse_path(&p.join("restored.c")).unwrap(),
        CompleteCsfFile::parse_path(&p.join("copy.c")).unwrap()
    );
    let stats = json(
        p,
        &[
            "restore-csfs",
            "--descriptors",
            "features_descriptors.parquet",
            "--header",
            "features_header.toml",
            "--output",
            "selected.c",
            "--indices",
            "1",
            "--json",
        ],
    );
    assert_eq!(stats["record_count"], 1);
    assert_eq!(
        CompleteCsfFile::parse_path(&p.join("selected.c"))
            .unwrap()
            .records[0]
            .total_two_j,
        0
    );
}

#[test]
fn invalid_header_and_path_aliases_preserve_existing_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    prepare_descriptor(p);
    let before = fs::read(p.join("features_descriptors.parquet")).unwrap();
    let bad = invoke(
        p,
        &[
            "convert",
            "features_descriptors.parquet",
            "features_descriptors.parquet",
            "--json",
        ],
    );
    assert!(!bad.status.success());
    assert_eq!(
        before,
        fs::read(p.join("features_descriptors.parquet")).unwrap()
    );
    fs::write(p.join("saved.c"), "keep me").unwrap();
    let mut header = fs::read_to_string(p.join("features_header.toml")).unwrap();
    header.push_str("\n# changed header\n");
    fs::write(p.join("wrong.toml"), header).unwrap();
    let bad = invoke(
        p,
        &[
            "restore-csfs",
            "--descriptors",
            "features_descriptors.parquet",
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
            "--output",
            "memory_descriptors.parquet",
            "--transcript",
            "input.log",
            "--threads",
            "1",
            "--generation-storage",
            "memory",
            "--json",
        ],
    );
    fs::write(p.join("model.toml"), format!("[csfsgenerate]\n{MODEL}output = 'disk_descriptors.parquet'\ngeneration_storage = 'disk'\nmemory_budget_mib = 32\n")).unwrap();
    let disk = json(p, &["csfsgenerate", "-c", "model.toml", "--json"]);
    assert_eq!(baseline["record_count"], disk["record_count"]);
    assert_eq!(disk["record_count"], 3);
    assert!(!p.join("memory.c").exists() && !p.join("disk.c").exists());
    restore(
        p,
        "memory_descriptors.parquet",
        "memory_header.toml",
        "expected.c",
    );
    restore(
        p,
        "disk_descriptors.parquet",
        "disk_header.toml",
        "actual.c",
    );
    assert_eq!(
        fs::read(p.join("expected.c")).unwrap(),
        fs::read(p.join("actual.c")).unwrap()
    );
}

#[test]
fn estimate_creates_no_outputs_and_cli_overrides_config() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(
        p.join("model.toml"),
        format!("[csfsgenerate]\n{MODEL}output = 'unused_descriptors.parquet'\ngeneration_storage = 'disk'\n"),
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
    // An estimate writes only its run log, never calculation artifacts.
    assert_eq!(fs::read_dir(p).unwrap().count(), 2);
    assert!(p.join("rcsfs.log").is_file());
    success(
        p,
        &[
            "csfsgenerate",
            "--output",
            "chosen_descriptors.parquet",
            "-c",
            "model.toml",
            "--generation-storage",
            "memory",
        ],
    );
    assert!(p.join("chosen_descriptors.parquet").exists());
    assert!(!p.join("unused_descriptors.parquet").exists());
}
#[test]
fn batch_generation_split_uses_conf_naming_and_writes_log() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(p.join("model.toml"), format!("conf = 'helium'\n[csfsgenerate]\nas = 2\n{MODEL}\n[csfs-split.active_spaces]\nAS1 = '1s'\n")).unwrap();
    success(p, &["-c", "model.toml"]);
    assert!(p.join("helium_as2raw_desc.parquet").exists());
    assert!(p.join("helium_as1raw_desc.parquet").exists());
    let log = fs::read_to_string(p.join("rcsfs_helium.log")).unwrap();
    let records: Vec<Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.last().unwrap()["event"], "complete");
    assert_eq!(
        json(p, &["info", "helium_as1raw_desc.parquet", "--json"])["num_rows"],
        1
    );
}
#[test]
fn batch_is_validated_before_any_generation() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::write(p.join("bad.toml"), format!("[[csfsgenerate]]\n{MODEL}output = 'same_descriptors.parquet'\n[[csfsgenerate]]\n{MODEL}output = 'same_descriptors.parquet'\n")).unwrap();
    let bad = invoke(p, &["-c", "bad.toml"]);
    assert_eq!(bad.status.code(), Some(2));
    assert_eq!(fs::read_dir(p).unwrap().count(), 1);
    fs::write(
        p.join("bad.toml"),
        format!("[csfsgenerate]\n{MODEL}output = 'out.c'\n[gen-descriptors]\nunknown = true\n"),
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
    fs::write(p.join("model.toml"), format!("[csfsgenerate]\ninactive_core = 0\nthreads = 1\noutput = 'union_descriptors.parquet'\n[[csfsgenerate.lists]]\n{list}[[csfsgenerate.lists]]\n{list}")).unwrap();
    let stats = json(p, &["csfsgenerate", "-c", "model.toml", "--json"]);
    assert!(stats["duplicate_count"].as_u64().unwrap() > 0);
    assert_eq!(
        stats["generated_count"].as_u64().unwrap(),
        stats["record_count"].as_u64().unwrap() * 2
    );
    assert_eq!(
        stats["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["record_count"].as_u64().unwrap())
            .sum::<u64>(),
        stats["record_count"].as_u64().unwrap()
    );
}

#[test]
fn human_console_is_concise_and_full_details_are_logged() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    let source = fs::read_to_string(fixture("descriptor_multiblock.c")).unwrap();
    let lines: Vec<_> = source.lines().collect();
    let text = format!(
        "{}\n{}",
        lines[..5].join("\n"),
        format!("{}\n", lines[5..8].join("\n")).repeat(1234)
    );
    fs::write(p.join("many.c"), text).unwrap();
    let converted = success(p, &["convert", "many.c", "many.parquet"]);
    let stdout = String::from_utf8(converted.stdout).unwrap();
    let stderr = String::from_utf8(converted.stderr).unwrap();
    assert!(stdout.contains("1,234"), "{stdout}");
    assert!(stdout.contains("KB)"), "{stdout}");
    assert!(
        !stdout.contains("csf_count") && !stdout.contains("{"),
        "{stdout}"
    );
    assert_eq!(stderr.trim(), "Run log: rcsfs.log");
    let log = fs::read_to_string(p.join("rcsfs.log")).unwrap();
    let events: Vec<Value> = log
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(events.iter().any(|event| event["event"] == "step_start"));
    assert_eq!(
        events
            .iter()
            .find(|event| event["event"] == "result")
            .unwrap()["stats"]["record_count"],
        1234
    );
    assert_eq!(events.last().unwrap()["success"], true);
    let info = success(p, &["info", "many.parquet"]);
    let stdout = String::from_utf8(info.stdout).unwrap();
    assert!(
        stdout.contains("1,234") && stdout.contains("KB"),
        "{stdout}"
    );
    assert!(!stdout.contains("key_value_metadata"), "{stdout}");
    let result = json(p, &["info", "many.parquet", "--json"]);
    assert_eq!(result["num_rows"], 1234);
    assert_eq!(
        result["file_size"],
        fs::metadata(p.join("many.parquet")).unwrap().len()
    );
}

#[test]

fn malformed_import_reports_error_and_preserves_complete_output() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    let mut source = fs::read_to_string(fixture("descriptor_multiblock.c")).unwrap();
    source.push_str("incomplete trailing record\n");
    fs::write(p.join("invalid.c"), source).unwrap();
    fs::write(p.join("out.parquet"), b"previous").unwrap();
    let bad = invoke(p, &["convert", "invalid.c", "out.parquet", "--json"]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("truncated CSF state line"));
    assert_eq!(fs::read(p.join("out.parquet")).unwrap(), b"previous");
    assert!(!p.join("out_header.toml").exists());
    let log = fs::read_to_string(p.join("rcsfs.log")).unwrap();
    let last: Value = serde_json::from_str(log.lines().last().unwrap()).unwrap();
    assert_eq!(last["success"], false);
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
    let log = fs::read_to_string(p.join("rcsfs_he.log")).unwrap();
    let events: Vec<Value> = log
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let result = events
        .iter()
        .find(|event| event["event"] == "result")
        .unwrap();
    assert_eq!(result["stats"]["results"].as_array().unwrap().len(), 2);
    assert_eq!(events.last().unwrap()["success"], true);
    let failing = model.replace("'he'", "'bad'").replace("'2s,2p'", "'2p'");
    fs::write(p.join("bad.toml"), failing).unwrap();
    assert!(!invoke(p, &["-c", "bad.toml"]).status.success());
    assert!(!p.join("bad_as2raw.g").exists());
}
#[test]

fn jsplit_copies_orbitals_and_accepts_config() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    success(
        p,
        &[
            "convert",
            fixture("descriptor_multiblock.c").to_str().unwrap(),
            "basis_descriptors.parquet",
        ],
    );
    fs::copy(fixture("multi_j.w"), p.join("basis.w")).unwrap();
    let stats = json(
        p,
        &[
            "jsplit",
            "basis_descriptors.parquet",
            "--prefix",
            "block",
            "--json",
        ],
    );
    let blocks = stats["outputs"].as_array().unwrap();
    assert_eq!(blocks.len(), 2);
    for block in blocks {
        assert_eq!(
            fs::read(p.join(block["w_file"].as_str().unwrap())).unwrap(),
            fs::read(fixture("multi_j.w")).unwrap()
        );
        assert_eq!(block["record_count"], 1);
    }
    fs::write(
        p.join("config.toml"),
        "[jsplit]\ninput_csf = 'basis_descriptors.parquet'\nprefix = 'nocopy'\ncopy_w = false\n",
    )
    .unwrap();
    let stats = json(p, &["jsplit", "-c", "config.toml", "--json"]);
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
    for output in ["reference_descriptors.parquet", "full_descriptors.parquet"] {
        success(
            p,
            &[
                "convert",
                fixture("descriptor_multiblock.c").to_str().unwrap(),
                output,
            ],
        );
    }
    let stats = json(
        p,
        &[
            "zero-first",
            "reference_descriptors.parquet",
            "full_descriptors.parquet",
            "zf_descriptors.parquet",
            "--json",
        ],
    );
    assert_eq!(stats["first_order_count"], 0);
    let blocks = stats["blocks"].as_array().unwrap();
    assert_eq!(blocks.len(), 2);
    assert!(blocks.iter().all(|block| block["zero_count"] == 1));
    assert!(!p.join("zf.c").exists());
    restore(p, "full_descriptors.parquet", "full_header.toml", "full.c");
    restore(p, "zf_descriptors.parquet", "zf_header.toml", "zf.c");
    let full = CompleteCsfFile::parse_path(&p.join("full.c")).unwrap();
    assert_eq!(full, CompleteCsfFile::parse_path(&p.join("zf.c")).unwrap());
    let stats = json(
        p,
        &[
            "interacting",
            "reference_descriptors.parquet",
            "full_descriptors.parquet",
            "-o",
            "selected_descriptors.parquet",
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
        format!("[csfsgenerate]\n{MODEL}output = 'model.toml'\n"),
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
        .args([
            "csfsgenerate",
            "--output",
            "interactive_descriptors.parquet",
            "--threads",
            "1",
            "--json",
        ])
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
    success(
        p,
        &[
            "csfsgenerate",
            "--output",
            "repeat_descriptors.parquet",
            "-c",
        ],
    );
    restore(
        p,
        "interactive_descriptors.parquet",
        "interactive_header.toml",
        "interactive.c",
    );
    restore(
        p,
        "repeat_descriptors.parquet",
        "repeat_header.toml",
        "repeat.c",
    );
    assert_eq!(
        fs::read(p.join("interactive.c")).unwrap(),
        fs::read(p.join("repeat.c")).unwrap()
    );
}
