//! Standalone native reader. These path modules are shared source, not copies.
//! Build from the complete rCSFs source distribution; no Python/CLI/generator dependency.
#[path = "../src/descriptor_ffi.rs"]
#[cfg(feature = "descriptor-ffi")]
pub mod descriptor_ffi;
#[path = "../src/descriptor_reader.rs"]
pub mod descriptor_reader;
#[path = "../src/descriptor_transport.rs"]
pub mod descriptor_schema;
