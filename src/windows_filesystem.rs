//! Windows volume space and file identity shared by preflight and alias checks.

use std::ffi::c_void;
use std::fs::OpenOptions;
use std::io;
use std::os::windows::{ffi::OsStrExt, fs::OpenOptionsExt, io::AsRawHandle};
use std::path::Path;

// FILE_ID_INFO from winbase.h. FileIdInfo is FILE_INFO_BY_HANDLE_CLASS 18.
#[repr(C)]
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct FileIdentity {
    pub(crate) volume: u64,
    id: [u8; 16],
}

#[link(name = "Kernel32")]
unsafe extern "system" {
    fn GetDiskFreeSpaceExW(
        directory: *const u16,
        available_to_caller: *mut u64,
        total: *mut u64,
        free: *mut u64,
    ) -> i32;
    fn GetFileInformationByHandleEx(
        file: *mut c_void,
        information_class: i32,
        information: *mut c_void,
        size: u32,
    ) -> i32;
}

pub(crate) fn free_bytes(path: &Path) -> io::Result<u64> {
    let directory = path.canonicalize()?;
    let mut wide: Vec<_> = directory.as_os_str().encode_wide().collect();
    // UNC directories require a trailing backslash. Canonicalization also
    // resolves relative paths and rejects nonexistent paths before the query.
    if wide.last() != Some(&u16::from(b'\\')) {
        wide.push(u16::from(b'\\'));
    }
    wide.push(0);
    let mut available = 0_u64;
    // SAFETY: `wide` is NUL-terminated and `available` is a writable ULARGE_INTEGER.
    // Optional outputs are null. The caller-available count respects disk quotas.
    let success = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if success == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(available)
    }
}

pub(crate) fn file_identity(path: &Path) -> io::Result<FileIdentity> {
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_ID_INFO_CLASS: i32 = 18;
    // Metadata access is sufficient; BACKUP_SEMANTICS permits directory handles.
    // OpenOptions follows symlinks/junctions, identifying the actual target volume.
    let file = OpenOptions::new()
        .access_mode(0)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    let mut identity = std::mem::MaybeUninit::<FileIdentity>::uninit();
    // SAFETY: the live File owns the handle, and the buffer has the C layout
    // and size required by FILE_ID_INFO. It is read only after a successful call.
    let success = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FILE_ID_INFO_CLASS,
            identity.as_mut_ptr().cast(),
            std::mem::size_of::<FileIdentity>() as u32,
        )
    };
    if success == 0 {
        Err(io::Error::last_os_error())
    } else {
        // SAFETY: GetFileInformationByHandleEx initialized FILE_ID_INFO.
        Ok(unsafe { identity.assume_init() })
    }
}
