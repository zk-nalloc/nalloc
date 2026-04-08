# Changelog

All notable changes to `zk-nalloc` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.2] - 2026-04-08

### Fixed
- Add `[package.metadata.docs.rs]` to Cargo.toml so docs.rs builds with
  all features enabled and correct rustdoc flags (`--cfg docsrs`).
  Targets: `x86_64-unknown-linux-gnu`, `x86_64-apple-darwin`,
  `x86_64-pc-windows-msvc`, `aarch64-apple-darwin`.
- `is_recycled()` memory ordering: `Relaxed` → `Acquire` to establish a
  proper happens-before edge with the `Release` store in `reset()`.

### Security
- Document TOCTOU limitation in `WitnessArena::alloc` and `secure_wipe`.
- Document that `static NAlloc` does not run `Drop` — witness data is NOT
  automatically wiped on program exit in the global-allocator use-case.
  Users must call `secure_wipe()` explicitly before prover shutdown.
- Document why `ArenaManager::Drop` ref-count check is safe despite the
  apparent TOCTOU between check and dealloc.

## [Unreleased]

### Added
- `impl Drop for NAlloc`: `NAlloc` now properly frees the heap-allocated
  `ArenaManager` on drop. Previously, non-`static` uses of `NAlloc`
  (e.g., `NAlloc::try_new()` in tests or short-lived proof contexts) silently
  leaked ~1.4 GB of virtual address space per instance.
- `test_drop_deallocates_arena_manager`: regression test that creates and drops
  a fully-initialized `NAlloc`, then verifies the heap is still healthy.
- `volatile_zero` helper in `witness.rs`: recycled-memory zeroing in
  `WitnessArena::alloc` and `alloc_zeroed` now uses volatile word-sized writes
  to prevent dead-store elimination, matching the security guarantee of
  `secure_reset`.
- `#[must_use]` attributes on `NAlloc`, `NAlloc::try_new`, `is_initialized`,
  `is_fallback_mode`, `try_witness`, `try_polynomial`, `try_scratch`, `stats`,
  and `stats_or_default` — the compiler now warns if callers discard these
  values.
- `.github/workflows/ci.yml`: GitHub Actions CI pipeline with:
  - Test matrix: stable + nightly × Linux + macOS
  - Feature variants: default, no-default-features, all-features (Linux)
  - `cargo fmt --check` + `cargo clippy -D warnings`
  - `cargo doc` with broken-intra-doc-link detection
  - `cargo audit` for dependency vulnerability scanning

### Fixed
- `SPIN_ITERATIONS` (config constant) was defined but never used. The
  initialization spin-wait loop in `NAlloc::init_arenas` now executes
  `SPIN_ITERATIONS` `hint::spin_loop()` calls between each state-load,
  reducing memory bus pressure on contended cache lines.
- `benches/alloc_benchmark.rs`: system-allocator benchmarks (`Small Alloc` and
  `Large Alloc (1MB)`) leaked memory on every iteration because there was no
  corresponding `dealloc`. Each bench closure now pairs `System.alloc` with
  `System.dealloc`.

---

## [0.2.0] — Initial public release

### Added
- Arena-based ZK-optimized allocator (`NAlloc`) with three specialized arenas:
  - **Witness Arena** (`WitnessArena`): secure zeroing, volatile wipe on reset,
    `mlock` support.
  - **Polynomial Arena** (`PolynomialArena`): 64-byte aligned allocations for
    FFT/NTT SIMD operations.
  - **Scratch Arena** (`BumpAlloc`): temporary computation buffers.
- Lock-free, atomic bump-pointer allocation (`BumpAlloc`).
- Platform-specific secure memory primitives (`sys` module):
  - `mmap`/`munmap` on Linux/macOS/Unix.
  - `VirtualAlloc`/`VirtualFree` on Windows.
  - `explicit_bzero` (Linux), `memset_s` (Apple), `RtlSecureZeroMemory`
    (Windows) for compiler-resilient zeroing.
- Optional features: `fallback` (default), `huge-pages`, `guard-pages`,
  `mlock`.
- `GlobalAlloc` implementation: large allocations (>1 MB) routed to
  Polynomial Arena; small allocations to Scratch Arena.
- Graceful fallback to system allocator on arena exhaustion or init failure.
- Comprehensive test suite: 52 unit, integration, and doc-tests.
- Benchmarks comparing `nalloc` vs system allocator for small/large/batch
  allocation patterns.