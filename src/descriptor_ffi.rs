//! C ABI v1. All buffers and live handles must be valid for the stated sizes.
//! Each handle is single-threaded; separate handles are independent.
use crate::descriptor_reader::DescriptorReader;
use anyhow::{Result, ensure};
use std::{
    ffi::{CStr, c_char, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    ptr,
};

struct Handle {
    reader: DescriptorReader,
    batch: Vec<i8>,
    offset: usize,
}

unsafe fn invoke(error: *mut c_char, capacity: usize, action: impl FnOnce() -> Result<()>) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(action));
    let message = match result {
        Ok(Ok(())) => String::new(),
        Ok(Err(e)) => format!("{e:#}"),
        Err(_) => "Rust descriptor reader panicked".into(),
    };
    if !error.is_null() && capacity > 0 {
        let bytes = message.as_bytes();
        let mut length = bytes.len().min(capacity - 1);
        while !message.is_char_boundary(length) {
            length -= 1;
        }
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), error.cast(), length);
            *error.add(length) = 0;
        }
    }
    if message.is_empty() { 0 } else { 1 }
}

#[unsafe(no_mangle)]
pub extern "C" fn rcsfs_descriptor_abi_version() -> i32 {
    1
}

/// # Safety
/// Paths must be NUL terminated; output/error buffers must be writable.
/// Output must point to an initialized NULL handle slot. A live slot is rejected
/// unchanged; close it before opening another file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rcsfs_descriptor_open(
    path: *const c_char,
    header: *const c_char,
    output: *mut *mut c_void,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        invoke(error, capacity, || {
            ensure!(!output.is_null(), "null output handle");
            ensure!(
                (*output).is_null(),
                "close the existing handle before reopening"
            );
            ensure!(!path.is_null() && !header.is_null(), "null input path");
            let reader = DescriptorReader::open(
                Path::new(CStr::from_ptr(path).to_str()?),
                Path::new(CStr::from_ptr(header).to_str()?),
                8192,
            )?;
            *output = Box::into_raw(Box::new(Handle {
                reader,
                batch: Vec::new(),
                offset: 0,
            }))
            .cast();
            Ok(())
        })
    }
}

/// # Safety
/// Handle must be live; info has five writable i64 entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rcsfs_descriptor_info(
    handle: *mut c_void,
    info: *mut i64,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        invoke(error, capacity, || {
            ensure!(!handle.is_null() && !info.is_null(), "null info argument");
            let r = &(*(handle as *mut Handle)).reader;
            let values = [
                r.core_count as i64,
                r.layout.subshell_count() as i64,
                r.block_lengths.len() as i64,
                i64::try_from(r.block_lengths.iter().sum::<u64>())?,
                i64::from(r.electron_count),
            ];
            ptr::copy_nonoverlapping(values.as_ptr(), info, 5);
            Ok(())
        })
    }
}

/// # Safety
/// Arrays must match orbital and block counts returned by info. Orbital and
/// symmetry arrays are interleaved pairs, lengths has one entry per block.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rcsfs_descriptor_tables(
    handle: *mut c_void,
    orbitals: *mut i32,
    orbital_capacity: usize,
    lengths: *mut i64,
    symmetries: *mut i32,
    block_capacity: usize,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        invoke(error, capacity, || {
            ensure!(
                !handle.is_null()
                    && !orbitals.is_null()
                    && !lengths.is_null()
                    && !symmetries.is_null(),
                "null tables argument"
            );
            let r = &(*(handle as *mut Handle)).reader;
            ensure!(
                orbital_capacity >= r.orbitals.len() && block_capacity >= r.block_lengths.len(),
                "table buffer too small"
            );
            for (i, &(n, k)) in r.orbitals.iter().enumerate() {
                *orbitals.add(2 * i) = n;
                *orbitals.add(2 * i + 1) = k;
            }
            for (i, (&length, &(j, p))) in
                r.block_lengths.iter().zip(&r.block_symmetries).enumerate()
            {
                *lengths.add(i) = i64::try_from(length)?;
                *symmetries.add(2 * i) = j;
                *symmetries.add(2 * i + 1) = p;
            }
            Ok(())
        })
    }
}

/// # Safety
/// Handle must be live and exclusively borrowed. Output has `size` writable
/// bytes. Reads exactly rows; a failure invalidates the output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rcsfs_descriptor_read(
    handle: *mut c_void,
    output: *mut i8,
    size: usize,
    rows: usize,
    error: *mut c_char,
    capacity: usize,
) -> i32 {
    unsafe {
        invoke(error, capacity, || {
            ensure!(!handle.is_null() && !output.is_null(), "null read argument");
            let h = &mut *(handle as *mut Handle);
            let count = rows
                .checked_mul(h.reader.layout.row_len())
                .ok_or_else(|| anyhow::anyhow!("size overflow"))?;
            ensure!(size >= count, "output buffer too small");
            let mut written = 0;
            while written < count {
                if h.offset == h.batch.len() {
                    ensure!(h.reader.read_batch(&mut h.batch)? > 0, "unexpected EOF");
                    h.offset = 0;
                }
                let n = (count - written).min(h.batch.len() - h.offset);
                ptr::copy_nonoverlapping(h.batch.as_ptr().add(h.offset), output.add(written), n);
                written += n;
                h.offset += n;
            }
            Ok(())
        })
    }
}

/// # Safety
/// Pointer must address a live handle or NULL. Clears the caller's handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rcsfs_descriptor_close(handle: *mut *mut c_void) {
    if !handle.is_null() {
        unsafe {
            if !(*handle).is_null() {
                drop(Box::from_raw(*handle as *mut Handle));
                *handle = ptr::null_mut();
            }
        }
    }
}
