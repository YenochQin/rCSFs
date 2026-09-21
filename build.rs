//! Supplies the macOS link arguments an extension module needs.
//!
//! An extension module deliberately leaves the CPython symbols unresolved at
//! link time; the host interpreter provides them when the module is loaded.
//! Maturin passes the matching `-undefined dynamic_lookup` flags itself, so a
//! wheel build is unaffected, but a plain
//! `cargo build --release --features pyo3/extension-module` does not, and
//! fails on macOS with undefined `_Py*` symbols. Emitting the flags from this
//! script keeps the documented in-tree build working.
//!
//! The arguments are scoped to the `cdylib` artifact, so test and example
//! binaries keep link-time checking of their own symbols.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
