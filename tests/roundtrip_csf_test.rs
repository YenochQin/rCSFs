#[allow(dead_code)]
#[path = "../examples/roundtrip_csf.rs"]
mod example;

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "rcsfs-roundtrip-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const CSF: &[u8] = include_bytes!("fixtures/complete.csf");

#[test]
fn refuses_existing_outputs_and_input_aliases_without_overwriting() {
    let directory = TestDirectory::new();
    let input = directory.0.join("input.c");
    fs::write(&input, CSF).unwrap();
    let hard_link = directory.0.join("hard-link.c");
    fs::hard_link(&input, &hard_link).unwrap();
    let existing = directory.0.join("existing.c");
    fs::write(&existing, b"preserve existing output").unwrap();
    let mut outputs = vec![input.clone(), hard_link, existing.clone()];
    #[cfg(unix)]
    {
        let symlink = directory.0.join("symlink.c");
        std::os::unix::fs::symlink(&input, &symlink).unwrap();
        outputs.push(symlink);
    }
    for output in outputs {
        let error = example::roundtrip(&input, &output).unwrap_err();
        assert!(error.to_string().contains("output must be a new file"));
        assert_eq!(fs::read(&input).unwrap(), CSF);
        assert_eq!(fs::read(&existing).unwrap(), b"preserve existing output");
    }
}

#[test]
fn writes_byte_identical_text_to_a_new_output() {
    let directory = TestDirectory::new();
    let input = directory.0.join("input.c");
    let output = directory.0.join("output.c");
    fs::write(&input, CSF).unwrap();
    example::roundtrip(&input, &output).unwrap();
    assert_eq!(fs::read(&input).unwrap(), CSF);
    assert_eq!(fs::read(&output).unwrap(), CSF);
}
