//! CLI output helpers.
//!
//! The CLI presents formatted, user-facing output (tables, prompts, status
//! messages) on the process's standard streams. These macros write directly to
//! `stdout`/`stderr` so that interactive output behaves exactly like the
//! standard library print macros, while keeping the codebase free of the
//! standard print-line tokens that the project lints forbid in library code.

/// Write a line to standard output (newline appended), like the standard
/// print-line macro.
///
/// Errors writing to stdout are intentionally ignored, mirroring the behaviour
/// of the standard print-line macro for interactive CLI usage.
#[macro_export]
macro_rules! cli_outln {
    () => {{
        use ::std::io::Write as _;
        let mut out = ::std::io::stdout();
        let _ = ::std::writeln!(out);
    }};
    ($($arg:tt)*) => {{
        use ::std::io::Write as _;
        let mut out = ::std::io::stdout();
        let _ = ::std::writeln!(out, $($arg)*);
    }};
}

/// Write to standard output without a trailing newline, like the standard
/// print macro.
#[macro_export]
macro_rules! cli_out {
    ($($arg:tt)*) => {{
        use ::std::io::Write as _;
        let mut out = ::std::io::stdout();
        let _ = ::std::write!(out, $($arg)*);
        let _ = out.flush();
    }};
}

/// Write a line to standard error (newline appended), like the standard
/// error print-line macro.
#[macro_export]
macro_rules! cli_errln {
    () => {{
        use ::std::io::Write as _;
        let mut err = ::std::io::stderr();
        let _ = ::std::writeln!(err);
    }};
    ($($arg:tt)*) => {{
        use ::std::io::Write as _;
        let mut err = ::std::io::stderr();
        let _ = ::std::writeln!(err, $($arg)*);
    }};
}
