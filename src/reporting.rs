//! Route library diagnostics without tying algorithms to the CLI or its log format.
use std::{fmt, sync::OnceLock};

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    Detail,
    Warning,
    LegacyTable,
}
type DiagnosticSink = fn(Kind, fmt::Arguments<'_>);
static SINK: OnceLock<DiagnosticSink> = OnceLock::new();

#[cfg(feature = "cli")]
pub(crate) fn set_sink(sink: DiagnosticSink) {
    let _ = SINK.set(sink);
}

pub(crate) fn emit(kind: Kind, message: fmt::Arguments<'_>) {
    if let Some(sink) = SINK.get() {
        sink(kind, message);
    } else {
        // Library callers retain their diagnostics; only the CLI owns a run log.
        eprintln!("{message}");
    }
}

macro_rules! detail {
    ($($arg:tt)*) => { $crate::reporting::emit($crate::reporting::Kind::Detail, format_args!($($arg)*)) };
}
macro_rules! warning {
    ($($arg:tt)*) => { $crate::reporting::emit($crate::reporting::Kind::Warning, format_args!($($arg)*)) };
}
macro_rules! legacy_table {
    ($($arg:tt)*) => { $crate::reporting::emit($crate::reporting::Kind::LegacyTable, format_args!($($arg)*)) };
}
pub(crate) use {detail, legacy_table, warning};
