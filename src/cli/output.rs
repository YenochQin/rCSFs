//! Human summaries are deliberately separate from the full JSON result contract.
use serde_json::Value;
use std::{fmt::Write, fs, path::Path};

const REPORT_WIDTH: usize = 72;

pub(super) fn count(value: u64) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(digit);
    }
    output
}

/// SI units: 1 MB = 1,000,000 bytes, 1 GB = 1,000,000,000 bytes.
pub(super) fn size(bytes: u64) -> String {
    if bytes < 1_000 {
        return format!("{bytes} B");
    }
    let mut amount = bytes as f64;
    let units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    let mut unit = 0;
    while amount >= 1_000.0 && unit + 1 < units.len() {
        amount /= 1_000.0;
        unit += 1;
    }
    format!("{amount:.2} {}", units[unit])
}

fn field(out: &mut String, label: &str, value: impl std::fmt::Display) {
    writeln!(out, "  {label:<23} {value}").unwrap();
}

fn section(out: &mut String, title: &str) {
    writeln!(out, "\n  {title}").unwrap();
    writeln!(out, "  {}", "-".repeat(REPORT_WIDTH - 2)).unwrap();
}

fn number(out: &mut String, stats: &Value, key: &str, label: &str) {
    if let Some(value) = stats[key].as_u64() {
        field(out, label, count(value));
    }
}

fn artifact(out: &mut String, label: &str, path: &str) {
    let suffix = fs::metadata(Path::new(path))
        .map(|metadata| format!("  ({})", size(metadata.len())))
        .unwrap_or_default();
    field(out, label, format_args!("{path}{suffix}"));
}

fn symmetry(block: &Value) -> Option<String> {
    let two_j = block["total_two_j"]
        .as_u64()
        .or_else(|| block["two_j"].as_u64())?;
    let j = if two_j.is_multiple_of(2) {
        (two_j / 2).to_string()
    } else {
        format!("{two_j}/2")
    };
    let parity = block["parity"].as_str().unwrap_or("");
    Some(format!("J = {j} {parity}"))
}

pub(super) fn summary(command: &str, stats: &Value, elapsed: f64) -> String {
    let mut out = String::new();
    let status = if stats["estimate_only"] == true {
        "estimate"
    } else {
        "completed"
    };
    // Each result starts one blank line after the log notice or previous result.
    writeln!(out, "\n{}", "=".repeat(REPORT_WIDTH)).unwrap();
    writeln!(out, "{command}: {status} ({elapsed:.2} s)").unwrap();
    writeln!(out, "{}", "-".repeat(REPORT_WIDTH)).unwrap();
    if let Some(results) = stats["results"].as_array() {
        for (index, result) in results.iter().enumerate() {
            section(&mut out, &format!("Model {}", count(index as u64 + 1)));
            details(&mut out, command, result);
        }
    } else {
        details(&mut out, command, stats);
    }
    out
}

fn details(out: &mut String, command: &str, stats: &Value) {
    number(
        out,
        stats,
        "record_count",
        if command == "csfggenerate" {
            "CSFG records"
        } else {
            "CSFs"
        },
    );
    for (key, label) in [
        ("csf_count", "CSFs"),
        ("input_csf_count", "Input CSFs"),
        ("output_csf_count", "Output CSFs"),
        ("output_count", "Output CSFs"),
        ("block_count", "Symmetry blocks"),
        ("descriptor_count", "Descriptors"),
        ("duplicate_count", "Duplicates removed"),
        ("label_csf_count", "Label CSFs"),
        ("generator_count", "Generators"),
        ("expanded_csf_count", "Expanded CSFs"),
        ("zero_csf_count", "Zero-order CSFs"),
        ("full_csf_count", "Complete-space CSFs"),
        ("first_order_count", "First-order CSFs"),
        ("reference_count", "Reference CSFs"),
        ("candidate_count", "Candidate CSFs"),
        ("selected_count", "Selected candidates"),
        ("rejected_count", "Rejected candidates"),
        ("num_rows", "Rows"),
        ("num_columns", "Columns"),
    ] {
        number(out, stats, key, label);
    }
    if let Some(value) = stats["compression"].as_str() {
        field(out, "Compression", value);
    }
    if stats["estimate_only"] == true {
        number(out, stats, "pre_deduplication_records", "CSFs before dedup");
        field(
            out,
            "Estimate scope",
            "before deduplication; no CSF outputs written",
        );
        for (key, label) in [
            ("required_scratch", "Required scratch"),
            ("required_output", "Required output"),
            ("csf_text", "CSF text estimate"),
            ("descriptor", "Descriptor estimate"),
        ] {
            if let Some(bytes) = stats["bytes"][key].as_u64() {
                field(out, label, size(bytes));
            }
        }
        if let Some(checks) = stats["space_checks"].as_array() {
            for check in checks {
                let status = match check["sufficient"].as_bool() {
                    Some(true) => "OK",
                    Some(false) => "INSUFFICIENT",
                    None => "UNCHECKED",
                };
                field(
                    out,
                    "Disk space",
                    format_args!("{}: {status}", check["path"].as_str().unwrap_or("?")),
                );
            }
        }
    }
    if let Some(blocks) = stats["blocks"]
        .as_array()
        .filter(|blocks| !blocks.is_empty())
    {
        writeln!(
            out,
            "\n  {:<16} {:>18}",
            "J / parity",
            if command == "csfggenerate" {
                "CSFG records"
            } else {
                "CSFs"
            }
        )
        .unwrap();
        writeln!(out, "  {}+{}", "-".repeat(16), "-".repeat(18)).unwrap();
        for block in blocks {
            let count_value = block["record_count"]
                .as_u64()
                .or_else(|| block["output_count"].as_u64());
            if let (Some(label), Some(value)) = (symmetry(block), count_value) {
                writeln!(out, "  {label:<16} {:>18}", count(value)).unwrap();
            }
        }
    }
    let artifacts = [
        ("output_file", "Output"),
        ("parquet_file", "CSF Parquet"),
        ("header_file", "Header"),
        ("descriptor_parquet_file", "Descriptors"),
        ("descriptor_metadata_file", "Descriptor metadata"),
        ("rcsfg_out", "CSFG file"),
        ("rlabel_out", "Label file"),
    ];
    if artifacts.iter().any(|(key, _)| stats[key].is_string()) {
        section(out, "Output files");
    }
    for (key, label) in artifacts {
        if let Some(path) = stats[key].as_str() {
            artifact(out, label, path);
        }
    }
    if stats["file_path"].is_string() || stats["file_size"].is_u64() {
        section(out, "File information");
    }
    if let Some(path) = stats["file_path"].as_str() {
        field(out, "File", path);
    }
    if let Some(bytes) = stats["file_size"].as_u64() {
        field(out, "File size", size(bytes));
    }
    if let Some(outputs) = stats["outputs"]
        .as_array()
        .filter(|outputs| !outputs.is_empty())
    {
        section(out, "Split outputs");
        for (index, item) in outputs.iter().enumerate() {
            if index > 0 {
                writeln!(out).unwrap();
            }
            let label = symmetry(item)
                .or_else(|| item["maximum_orbitals"].as_str().map(str::to_owned))
                .unwrap_or_default();
            if let Some(value) = item["csf_count"].as_u64() {
                field(out, &label, format_args!("{} CSFs", count(value)));
            }
            if let Some(path) = item["output_file"].as_str() {
                artifact(out, "  CSF file", path);
            }
            if let Some(path) = item["w_file"].as_str() {
                artifact(out, "  Orbitals", path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_units_and_counts_cover_boundaries() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1_000), "1,000");
        assert_eq!(count(u64::MAX), "18,446,744,073,709,551,615");
        assert_eq!(size(999), "999 B");
        assert_eq!(size(1_000), "1.00 KB");
        assert_eq!(size(999_000), "999.00 KB");
        assert_eq!(size(1_000_000), "1.00 MB");
        assert_eq!(size(999_000_000), "999.00 MB");
        assert_eq!(size(1_250_000_000), "1.25 GB");
        assert_eq!(size(1_000_000_000_000), "1.00 TB");
    }
}
