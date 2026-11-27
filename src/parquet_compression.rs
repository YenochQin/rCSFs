//! Compression settings for descriptor Parquet writers.
use anyhow::Result;
use parquet::basic::{Compression, ZstdLevel};

pub(crate) fn parse(compression: Option<&str>) -> Result<Compression> {
    let Some(spec) = compression else {
        return Ok(Compression::ZSTD(
            ZstdLevel::try_new(3).expect("zstd level 3 is always valid"),
        ));
    };

    let lower = spec.trim().to_ascii_lowercase();
    match lower.as_str() {
        "none" | "uncompressed" => Ok(Compression::UNCOMPRESSED),
        "snappy" => Ok(Compression::SNAPPY),
        "zstd" => Ok(Compression::ZSTD(
            ZstdLevel::try_new(3).expect("zstd level 3 is always valid"),
        )),
        other if other.starts_with("zstd-") => {
            let level_str = &other["zstd-".len()..];
            let level: i32 = level_str.parse().map_err(|_| {
                anyhow::anyhow!(
                    "invalid zstd level '{}': expected integer 1..=22",
                    level_str
                )
            })?;
            let zstd_level = ZstdLevel::try_new(level).map_err(|_| {
                anyhow::anyhow!("zstd level {} out of range (expected 1..=22)", level)
            })?;
            Ok(Compression::ZSTD(zstd_level))
        }
        other => Err(anyhow::anyhow!(
            "unknown compression '{}': expected one of none/uncompressed/snappy/zstd/zstd-N",
            other
        )),
    }
}
