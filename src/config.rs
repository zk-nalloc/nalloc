//! Configuration constants for nalloc.
//!
//! This module centralizes all tunable parameters and magic numbers
//! to make the allocator easily configurable.

// ============================================================================
// Arena Sizes
// ============================================================================

/// Size of the Witness Arena in bytes.
/// Used for private ZK inputs requiring secure wiping.
pub const WITNESS_ARENA_SIZE: usize = 128 * 1024 * 1024; // 128 MB

/// Size of the Polynomial Arena in bytes.
/// Used for FFT/NTT coefficient vectors - needs to be large for complex circuits.
pub const POLY_ARENA_SIZE: usize = 1024 * 1024 * 1024; // 1 GB

/// Size of the Scratch Arena in bytes.
/// Used for temporary computation buffers.
pub const SCRATCH_ARENA_SIZE: usize = 256 * 1024 * 1024; // 256 MB

// ============================================================================
// Allocation Thresholds
// ============================================================================

/// Allocations larger than this threshold go to the Polynomial Arena.
/// Smaller allocations go to the Scratch Arena via GlobalAlloc.
pub const LARGE_ALLOC_THRESHOLD: usize = 1024 * 1024; // 1 MB

// ============================================================================
// Alignment Constants
// ============================================================================

/// Cache line size for SIMD-friendly FFT/NTT operations.
/// 64 bytes is optimal for AVX-512 and most modern CPUs.
pub const CACHE_LINE_ALIGN: usize = 64;

/// Page alignment for huge vector allocations.
/// 4KB works across Linux, macOS, and Windows.
pub const PAGE_ALIGN: usize = 4096;

/// Default minimum alignment for all allocations.
pub const DEFAULT_ALIGN: usize = 8;

/// 2MB huge page size (Linux).
#[cfg(feature = "huge-pages")]
pub const HUGE_PAGE_2MB: usize = 2 * 1024 * 1024;

/// 1GB huge page size (Linux, requires explicit kernel config).
#[cfg(feature = "huge-pages")]
pub const HUGE_PAGE_1GB: usize = 1024 * 1024 * 1024;

// ============================================================================
// Security Constants
// ============================================================================

/// Pattern used for memory poisoning in debug builds.
/// 0xDE is a distinctive pattern that helps identify use-after-free.
#[cfg(debug_assertions)]
pub const POISON_PATTERN: u8 = 0xDE;

/// Secure wipe pattern (zero is standard for cryptographic applications).
pub const SECURE_WIPE_PATTERN: u8 = 0x00;

// ============================================================================
// Timeout and Retry Constants
// ============================================================================

/// Maximum number of CAS retries before giving up (for extreme contention).
pub const MAX_CAS_RETRIES: usize = 1000;

/// Spin loop hint iterations before yielding.
pub const SPIN_ITERATIONS: usize = 32;

// ============================================================================
// Guard Page Constants
// ============================================================================

/// Size of guard pages (usually one page).
#[cfg(feature = "guard-pages")]
pub const GUARD_PAGE_SIZE: usize = PAGE_ALIGN;
