//! Read V2 descriptor batches without restoring text or invoking Python.
//! Buffers contain original channels; physical validation belongs to the consumer.
use anyhow::{Context, Result, ensure};
use arrow_array::{Array, Int8Array};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use sha2::{Digest, Sha256};
use std::{fs::File, path::Path};

use crate::descriptor_schema::{DescriptorLayout, DescriptorVersion};

pub struct DescriptorReader {
    reader: ParquetRecordBatchReader,
    pub header_lines: [String; 5],
    pub block_lengths: Vec<u64>,
    pub layout: DescriptorLayout,
    remaining: u64,
    failed: bool,
    pub core_count: usize,
    pub orbitals: Vec<(i32, i32)>,
    pub block_symmetries: Vec<(i32, i32)>,
    pub electron_count: i32,
    block_index: usize,
    block_rows: u64,
}

fn orbital(label: &str) -> Result<(i32, i32)> {
    ensure!(label.is_ascii(), "non-ASCII orbital label");
    let (base, lower) = label
        .strip_suffix('-')
        .map_or((label, false), |s| (s, true));
    ensure!(base.len() >= 2, "invalid orbital label");
    let (n, letter) = base.split_at(base.len() - 1);
    let l = "spdfghiklmno"
        .find(letter)
        .context("invalid orbital letter")? as i32;
    let n: i32 = n.parse()?;
    ensure!(
        n > l && (!lower || l > 0),
        "invalid orbital quantum numbers"
    );
    Ok((n, if lower { l } else { -l - 1 }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatch;
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
    use std::sync::Arc;

    fn fixture(
        root: &Path,
        mutate: impl FnOnce(&mut Vec<parquet::file::metadata::KeyValue>),
    ) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
        let header = root.join("basis_header.toml");
        let path = root.join("basis_descriptors.parquet");
        std::fs::write(
            &header,
            "[header_info]\nheader_lines = ['Core subshells:', '1s', 'Peel subshells:', '2s 2p', 'CSF(s):']\nelectron_count=4\n[block_info]\nblock_count=2\nblock_lengths=[1,1]\n",
        )?;
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 2);
        let schema = crate::descriptor_schema::output_schema(layout)?;
        let hash = crate::descriptor_schema::hash_header_file(&header)?;
        let mut metadata = crate::descriptor_schema::output_kv_metadata(
            layout,
            &["2s".into(), "2p".into()],
            Some(&hash),
            None,
        );
        mutate(&mut metadata);
        let properties = WriterProperties::builder()
            .set_key_value_metadata(Some(metadata))
            .set_max_row_group_size(1)
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path)?, schema.clone(), Some(properties))?;
        let rows = [
            [2, -1, 0, -1, 0, -1, 0, -1, 0, 1],
            [0, -1, 0, -1, 2, 0, 0, -1, 0, 1],
        ];
        let columns = (0..10)
            .map(|c| {
                Arc::new(Int8Array::from(vec![rows[0][c], rows[1][c]])) as arrow_array::ArrayRef
            })
            .collect();
        writer.write(&RecordBatch::try_new(schema, columns)?)?;
        writer.close()?;
        Ok((path, header))
    }

    #[test]
    fn strict_footer_keys_and_ordered_batches() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let required = [
            "descriptor_version",
            "descriptor_dtype",
            "max_two_j",
            "seniority_encoding",
            "channels_per_subshell",
            "subshell_count",
            "missing_sentinel",
            "normalized",
            "source_header_sha256",
            "peel_subshells",
            "feature_columns",
            "global_columns",
        ];
        for key in required {
            let (path, header) = fixture(directory.path(), |metadata| {
                metadata.retain(|entry| entry.key != key)
            })?;
            assert!(
                DescriptorReader::open(&path, &header, 1).is_err(),
                "missing {key}"
            );
            let (path, header) = fixture(directory.path(), |metadata| {
                let duplicate = metadata
                    .iter()
                    .find(|entry| entry.key == key)
                    .unwrap()
                    .clone();
                metadata.push(duplicate);
            })?;
            assert!(
                DescriptorReader::open(&path, &header, 1).is_err(),
                "duplicate {key}"
            );
            let (path, header) = fixture(directory.path(), |metadata| {
                metadata
                    .iter_mut()
                    .find(|entry| entry.key == key)
                    .unwrap()
                    .value = Some("invalid".into());
            })?;
            assert!(
                DescriptorReader::open(&path, &header, 1).is_err(),
                "invalid {key}"
            );
        }
        let (path, header) = fixture(directory.path(), |_| {})?;
        let mut first = DescriptorReader::open(&path, &header, 1)?;
        let mut second = DescriptorReader::open(&path, &header, 2)?;
        assert_eq!(first.block_lengths, [1, 1]);
        assert_eq!(first.block_symmetries, [(0, 1), (0, 1)]);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        assert_eq!(first.read_batch(&mut a)?, 1);
        assert_eq!(second.read_batch(&mut b)?, 2);
        assert_eq!(a, b[..10]);
        assert_eq!(first.read_batch(&mut a)?, 1);
        assert_eq!(a, b[10..]);
        assert_eq!(first.read_batch(&mut a)?, 0);
        Ok(())
    }

    #[test]
    fn rejects_corrupt_payload_and_header_semantics() -> Result<()> {
        let directory = tempfile::tempdir()?;
        for replacement in [
            "block_lengths=[0,2]",
            "block_lengths=[1]",
            "block_lengths=[-1,3]",
        ] {
            let (path, header) = fixture(directory.path(), |_| {})?;
            let text =
                std::fs::read_to_string(&header)?.replace("block_lengths=[1,1]", replacement);
            std::fs::write(&header, text)?;
            assert!(DescriptorReader::open(&path, &header, 1).is_err());
        }
        let (path, header) = fixture(directory.path(), |_| {})?;
        let original = std::fs::read(&path)?;
        for bytes in [
            &original[..original.len() - 1],
            &original[..original.len() / 2],
            b"GCSDESC1",
        ] {
            std::fs::write(&path, bytes)?;
            assert!(DescriptorReader::open(&path, &header, 1).is_err());
        }
        Ok(())
    }

    #[cfg(feature = "descriptor-ffi")]
    #[test]
    fn ffi_buffers_errors_and_independent_handles() -> Result<()> {
        use crate::descriptor_ffi::*;
        use std::ffi::{CString, c_char, c_void};
        let directory = tempfile::tempdir()?;
        let (path, header) = fixture(directory.path(), |_| {})?;
        let path = CString::new(path.to_str().unwrap())?;
        let header = CString::new(header.to_str().unwrap())?;
        let mut handles = [std::ptr::null_mut::<c_void>(); 2];
        let mut error = [0 as c_char; 512];
        unsafe {
            assert_eq!(rcsfs_descriptor_abi_version(), 1);
            for handle in &mut handles {
                assert_eq!(
                    rcsfs_descriptor_open(
                        path.as_ptr(),
                        header.as_ptr(),
                        handle,
                        error.as_mut_ptr(),
                        512
                    ),
                    0
                );
            }
            let original = handles[0];
            assert_ne!(
                rcsfs_descriptor_open(
                    path.as_ptr(),
                    header.as_ptr(),
                    &mut handles[0],
                    error.as_mut_ptr(),
                    512
                ),
                0
            );
            assert_eq!(handles[0], original);
            let mut info = [0i64; 5];
            assert_eq!(
                rcsfs_descriptor_info(handles[0], info.as_mut_ptr(), error.as_mut_ptr(), 512),
                0
            );
            assert_eq!(info, [1, 2, 2, 2, 4]);
            let mut orbitals = [-99i32; 6];
            let mut lengths = [0i64; 2];
            let mut symmetries = [0i32; 4];
            assert_ne!(
                rcsfs_descriptor_tables(
                    handles[0],
                    orbitals.as_mut_ptr(),
                    1,
                    lengths.as_mut_ptr(),
                    symmetries.as_mut_ptr(),
                    2,
                    error.as_mut_ptr(),
                    512
                ),
                0
            );
            assert_eq!(orbitals, [-99; 6]);
            assert_eq!(
                rcsfs_descriptor_tables(
                    handles[0],
                    orbitals.as_mut_ptr(),
                    3,
                    lengths.as_mut_ptr(),
                    symmetries.as_mut_ptr(),
                    2,
                    error.as_mut_ptr(),
                    512
                ),
                0
            );
            assert_eq!(orbitals, [1, -1, 2, -1, 2, -2]);
            assert_eq!(lengths, [1, 1]);
            assert_eq!(symmetries, [0, 1, 0, 1]);
            let mut rows = [-99i8; 20];
            assert_ne!(
                rcsfs_descriptor_read(handles[0], rows.as_mut_ptr(), 1, 1, error.as_mut_ptr(), 512),
                0
            );
            assert_eq!(rows, [-99; 20]);
            assert_eq!(
                rcsfs_descriptor_read(
                    handles[0],
                    rows.as_mut_ptr(),
                    20,
                    2,
                    error.as_mut_ptr(),
                    512
                ),
                0
            );
            assert_eq!(
                rows,
                [
                    2, -1, 0, -1, 0, -1, 0, -1, 0, 1, 0, -1, 0, -1, 2, 0, 0, -1, 0, 1
                ]
            );
            let mut other = [0i8; 10];
            assert_eq!(
                rcsfs_descriptor_read(
                    handles[1],
                    other.as_mut_ptr(),
                    10,
                    1,
                    error.as_mut_ptr(),
                    512
                ),
                0
            );
            assert_eq!(other, rows[..10]);
            assert_ne!(
                rcsfs_descriptor_read(
                    handles[0],
                    other.as_mut_ptr(),
                    10,
                    1,
                    error.as_mut_ptr(),
                    512
                ),
                0
            );
            for handle in &mut handles {
                rcsfs_descriptor_close(handle);
                rcsfs_descriptor_close(handle);
                assert!(handle.is_null());
            }
            assert_ne!(
                rcsfs_descriptor_info(handles[0], info.as_mut_ptr(), error.as_mut_ptr(), 512),
                0
            );
        }
        Ok(())
    }

    #[test]
    fn closed_core_batches_and_header_binding() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let header = dir.path().join("header.toml");
        let path = dir.path().join("descriptors.parquet");
        let text = "[header_info]\nheader_lines = ['Core subshells:', '1s', 'Peel subshells:', '', 'CSF(s):']\n[block_info]\nblock_lengths = [1]\n";
        std::fs::write(&header, text)?;
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 0);
        let schema = crate::descriptor_schema::output_schema(layout)?;
        let hash = crate::descriptor_schema::hash_header_file(&header)?;
        let kv = crate::descriptor_schema::output_kv_metadata(layout, &[], Some(&hash), None);
        let properties = WriterProperties::builder()
            .set_key_value_metadata(Some(kv))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path)?, schema.clone(), Some(properties))?;
        writer.write(&RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int8Array::from(vec![0])),
                Arc::new(Int8Array::from(vec![1])),
            ],
        )?)?;
        writer.close()?;
        let mut reader = DescriptorReader::open(&path, &header, 1)?;
        let mut values = Vec::new();
        assert_eq!(reader.read_batch(&mut values)?, 1);
        assert_eq!(values, [0, 1]);
        assert_eq!(reader.read_batch(&mut values)?, 0);
        assert!(values.is_empty());
        std::fs::write(&header, format!("{text}\n"))?;
        assert!(DescriptorReader::open(&path, &header, 1).is_err());
        assert!(DescriptorReader::open(&path, &header, 0).is_err());
        Ok(())
    }
}

impl DescriptorReader {
    /// Open with a bounded output batch size. Header hashing and parsing use
    /// the same byte snapshot. No application global state is modified.
    pub fn open(path: &Path, header: &Path, batch_size: usize) -> Result<Self> {
        ensure!(
            batch_size > 0 && batch_size <= 8192,
            "batch size must be in 1..=8192"
        );
        let bytes = std::fs::read(header)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let document: toml::Value = toml::from_str(std::str::from_utf8(&bytes)?)?;
        let lines = document
            .get("header_info")
            .and_then(|v| v.get("header_lines"))
            .and_then(toml::Value::as_array)
            .context("missing header lines")?;
        let header_lines: [String; 5] = lines
            .iter()
            .map(|v| {
                Ok(v.as_str()
                    .context("header line is not a string")?
                    .to_owned())
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected five header lines"))?;
        ensure!(
            header_lines
                .iter()
                .all(|line| !line.contains(['\n', '\r', '\0'])),
            "header lines must not contain embedded newline or NUL"
        );
        for (i, expected) in [
            (0, "Core subshells:"),
            (2, "Peel subshells:"),
            (4, "CSF(s):"),
        ] {
            ensure!(header_lines[i].trim() == expected, "invalid header label");
        }
        let peel: Vec<_> = header_lines[3].split_whitespace().collect();
        let core: Vec<_> = header_lines[1].split_whitespace().collect();
        let orbitals = core
            .iter()
            .chain(peel.iter())
            .map(|s| orbital(s))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!orbitals.is_empty(), "empty orbital table");
        let unique: std::collections::HashSet<_> = orbitals.iter().copied().collect();
        ensure!(
            unique.len() == orbitals.len(),
            "duplicate core/peel orbital"
        );
        let core_count = core.len();
        let layout = DescriptorLayout::new(DescriptorVersion::V2, peel.len());
        let lengths = document
            .get("block_info")
            .and_then(|v| v.get("block_lengths"))
            .and_then(toml::Value::as_array)
            .context("missing block lengths")?;
        let block_lengths = lengths
            .iter()
            .map(|v| {
                let n = v.as_integer().context("block length is not an integer")?;
                ensure!(n > 0, "empty or negative block");
                Ok(n as u64)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(!block_lengths.is_empty(), "no blocks");
        let total = block_lengths.iter().try_fold(0u64, |sum, n| {
            sum.checked_add(*n).context("row count overflow")
        })?;
        for (section, key, expected) in [
            ("block_info", "block_count", block_lengths.len() as u64),
            ("conversion_stats", "csf_count", total),
        ] {
            if let Some(value) = document.get(section).and_then(|v| v.get(key)) {
                ensure!(
                    value.as_integer().and_then(|n| u64::try_from(n).ok()) == Some(expected),
                    "inconsistent {key}"
                );
            }
        }
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file.try_clone()?)?;
        let metadata = builder.metadata().file_metadata();
        ensure!(
            u64::try_from(metadata.num_rows()).ok() == Some(total),
            "row count mismatch"
        );
        let kv = metadata
            .key_value_metadata()
            .context("missing descriptor metadata")?;
        let get = |key: &str| -> Result<&str> {
            let mut entries = kv.iter().filter(|v| v.key == key);
            let value = entries
                .next()
                .and_then(|v| v.value.as_deref())
                .context(format!("missing {key}"))?;
            ensure!(entries.next().is_none(), "duplicate {key}");
            Ok(value)
        };
        for (key, expected) in [
            ("descriptor_version", "2"),
            ("channels_per_subshell", "4"),
            ("missing_sentinel", "-1"),
            ("normalized", "false"),
            ("source_header_sha256", digest.as_str()),
        ] {
            ensure!(get(key)? == expected, "invalid {key}");
        }
        ensure!(
            get("subshell_count")?.parse::<usize>()? == peel.len(),
            "peel count mismatch"
        );
        crate::descriptor_schema::validate_seniority_encoding(kv)?;
        crate::descriptor_schema::validate_descriptor_storage(kv)?;
        crate::descriptor_schema::validate_output_schema(builder.schema(), layout)?;
        for (key, expected) in [
            (
                "peel_subshells",
                peel.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            ),
            ("feature_columns", layout.feature_column_names()),
            ("global_columns", layout.global_column_names()),
        ] {
            ensure!(
                serde_json::from_str::<Vec<String>>(get(key)?)? == expected,
                "invalid {key}"
            );
        }
        let declared_electrons = document
            .get("header_info")
            .and_then(|v| v.get("electron_count"))
            .map(|v| {
                v.as_integer()
                    .and_then(|n| i32::try_from(n).ok())
                    .filter(|n| *n > 0)
                    .context("invalid electron_count")
            })
            .transpose()?;
        let reader = builder.with_batch_size(batch_size).build()?;
        let mut source = Self {
            reader,
            header_lines,
            block_lengths,
            layout,
            remaining: total,
            failed: false,
            core_count,
            orbitals,
            block_symmetries: Vec::new(),
            electron_count: declared_electrons.unwrap_or(0),
            block_index: 0,
            block_rows: 0,
        };
        let mut values = Vec::new();
        while source.read_batch(&mut values)? != 0 {}
        ensure!(
            source.remaining == 0
                && source.block_index == source.block_lengths.len()
                && source.block_rows == 0
                && source.block_symmetries.len() == source.block_lengths.len(),
            "incomplete descriptor block table"
        );
        source.reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_batch_size(batch_size)
            .build()?;
        source.remaining = total;
        source.block_index = 0;
        source.block_rows = 0;
        Ok(source)
    }

    /// Returns rows copied into caller-owned channel-major Fortran storage
    /// (`channel + row_width * row`). An error invalidates this reader.
    pub fn read_batch(&mut self, output: &mut Vec<i8>) -> Result<usize> {
        output.clear();
        ensure!(!self.failed, "reader failed; close and reopen it");
        self.failed = true;
        let batch = loop {
            let Some(batch) = self.reader.next() else {
                ensure!(self.remaining == 0, "premature end of descriptors");
                self.failed = false;
                return Ok(0);
            };
            let batch = batch?;
            if batch.num_rows() > 0 {
                break batch;
            }
        };
        ensure!(
            batch.num_columns() == self.layout.row_len(),
            "column count mismatch"
        );
        let columns = batch
            .columns()
            .iter()
            .map(|column| {
                ensure!(column.null_count() == 0, "null descriptor channel");
                column
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .context("non-Int8 descriptor channel")
            })
            .collect::<Result<Vec<_>>>()?;
        let count = batch.num_rows();
        self.remaining = self
            .remaining
            .checked_sub(count as u64)
            .context("too many rows")?;
        let size = count
            .checked_mul(self.layout.row_len())
            .context("batch size overflow")?;
        output.try_reserve(size)?;
        for row in 0..count {
            for column in &columns {
                output.push(column.value(row));
            }
        }
        for row in output.chunks_exact(self.layout.row_len()) {
            ensure!(
                self.block_index < self.block_lengths.len(),
                "row outside blocks"
            );
            let symmetry = (i32::from(row[row.len() - 2]), i32::from(row[row.len() - 1]));
            ensure!(
                (0..=126).contains(&symmetry.0) && symmetry.1.abs() == 1,
                "invalid symmetry"
            );
            if self.block_symmetries.len() == self.block_index {
                self.block_symmetries.push(symmetry);
            }
            ensure!(
                self.block_symmetries[self.block_index] == symmetry,
                "mixed block symmetry"
            );
            let mut electrons: i32 = self.orbitals[..self.core_count]
                .iter()
                .map(|(_, k)| 2 * k.abs())
                .sum();
            let mut parity_sum = 0;
            for (shell, &(_, k)) in self.orbitals[self.core_count..].iter().enumerate() {
                let q = i32::from(row[4 * shell]);
                ensure!(
                    (0..=2 * k.abs()).contains(&q),
                    "occupation outside capacity"
                );
                ensure!(row[4 * shell + 2] >= 0, "negative seniority");
                ensure!(
                    (-1..=126).contains(&row[4 * shell + 1])
                        && (-1..=126).contains(&row[4 * shell + 3]),
                    "invalid J/K"
                );
                if q == 0 {
                    ensure!(
                        row[4 * shell..4 * shell + 4] == [0, -1, 0, -1],
                        "noncanonical empty shell"
                    );
                }
                electrons += q;
                parity_sum += q * if k > 0 { k } else { -k - 1 };
            }
            ensure!(electrons > 0, "empty electron basis");
            if self.electron_count == 0 {
                self.electron_count = electrons;
            }
            ensure!(electrons == self.electron_count, "electron count mismatch");
            ensure!(
                symmetry.1 == if parity_sum % 2 == 0 { 1 } else { -1 },
                "incorrect parity"
            );
            if self.layout.subshell_count() == 0 {
                ensure!(
                    self.block_lengths == [1] && symmetry == (0, 1),
                    "invalid closed-core basis"
                );
            }
            self.block_rows += 1;
            if self.block_rows == self.block_lengths[self.block_index] {
                self.block_index += 1;
                self.block_rows = 0;
            }
        }
        self.failed = false;
        Ok(count)
    }
}
