# Temporary test code

The maintained test suite is responsible only for this repository's own code:
its Rust algorithms, Python APIs, CLI behavior, and file formats. Tests must be
self-contained, using repository fixtures and normal project dependencies; they
must not require an external GRASP source checkout, executable, or private
baseline dataset. Stored fixtures may encode expected compatibility behavior.

Put temporary test code, exploratory probes, and one-off external comparisons
under `temp/`, not `tests/`, `src/`, or `examples/`. Temporary code is outside the
maintained suite and must not be added to Cargo or pytest test discovery. Keep
local inputs and generated outputs there untracked. This rule concerns temporary
test code; maintained tests may still use standard temporary directories for I/O.
