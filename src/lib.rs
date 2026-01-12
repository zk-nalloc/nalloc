//! nalloc: A ZK-Proof optimized memory allocator.
//!
//! This crate provides a high-performance, deterministic memory allocator
//! specifically designed for Zero-Knowledge proof systems. It is framework-agnostic
//! and works with any ZK system: Halo2, Plonky2, Risc0, SP1, Miden, Cairo, Arkworks, etc.
//!
//! # Features
//!
//! - **Arena-based allocation**: Pre-reserved memory pools for different workload types
//! - **Bump allocation**: O(1) allocation via atomic pointer increment
//! - **Security-first**: Volatile secure wiping for witness data
//! - **Cache-optimized**: 64-byte alignment for FFT/NTT SIMD operations
//! - **Cross-platform**: Linux, macOS, Windows, and Unix support
//! - **Zero ZK dependencies**: Pure memory primitive, no framework lock-in
//! - **Fallback support**: Gracefully falls back to system allocator when arena exhausted
//!
//! # Cargo Features
//!
//! - `fallback` (default): Fall back to system allocator when arena is exhausted
//! - `huge-pages`: Enable Linux 2MB/1GB huge page support
//! - `guard-pages`: Add guard pages at arena boundaries for overflow detection
//! - `mlock`: Lock witness memory to prevent swapping (security)
//!
//! # Usage
//!
//! As a global allocator:
//! ```rust,no_run
//! use zk_nalloc::NAlloc;
//!
//! #[global_allocator]
//! static ALLOC: NAlloc = NAlloc::new();
//!
//! fn main() {
//!     let data = vec![0u64; 1000];
//!     println!("Allocated {} elements", data.len());
//! }
//! ```
//!
//! Using specialized arenas directly:
//! ```rust
//! use zk_nalloc::NAlloc;
//!
//! let alloc = NAlloc::new();
//! let witness = alloc.witness();
//! let ptr = witness.alloc(1024, 8);
//! assert!(!ptr.is_null());
//!
//! // Securely wipe when done
//! unsafe { witness.secure_wipe(); }
//! ```

pub mod arena;
pub mod bump;
pub mod config;
pub mod platform;
pub mod polynomial;
pub mod witness;

pub use arena::{ArenaManager, ArenaStats};
pub use bump::BumpAlloc;
pub use config::*;
pub use platform::sys;
#[cfg(feature = "guard-pages")]
pub use platform::GuardedAlloc;
#[cfg(feature = "huge-pages")]
pub use platform::HugePageSize;
pub use platform::{AllocErrorKind, AllocFailed};
pub use polynomial::PolynomialArena;
pub use witness::WitnessArena;

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::{copy_nonoverlapping, null_mut};
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

/// Initialization state for NAlloc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum InitState {
    /// Not yet initialized
    Uninitialized = 0,
    /// Currently being initialized by another thread
    Initializing = 1,
    /// Successfully initialized with arenas
    Initialized = 2,
    /// Failed to initialize, using system allocator fallback
    Fallback = 3,
}

/// The global ZK-optimized allocator.
///
/// `NAlloc` provides a drop-in replacement for the standard Rust global allocator,
/// with special optimizations for ZK-Proof workloads.
///
/// # Memory Strategy
///
/// - **Large allocations (>1MB)**: Routed to Polynomial Arena (FFT vectors)
/// - **Small allocations**: Routed to Scratch Arena (temporary buffers)
/// - **Witness data**: Use `NAlloc::witness()` for security-critical allocations
///
/// # Thread Safety
///
/// This allocator uses lock-free atomic operations for initialization and
/// allocation. It's safe to use from multiple threads concurrently.
///
/// # Fallback Behavior
///
/// If arena initialization fails (e.g., out of memory), NAlloc gracefully
/// falls back to the system allocator rather than panicking. This ensures
/// your application continues to function even under memory pressure.
pub struct NAlloc {
    /// Pointer to the ArenaManager (null until initialized)
    arenas: AtomicPtr<ArenaManager>,
    /// Initialization state
    init_state: AtomicU8,
}

impl NAlloc {
    /// Create a new `NAlloc` instance.
    ///
    /// The arenas are lazily initialized on the first allocation.
    pub const fn new() -> Self {
        Self {
            arenas: AtomicPtr::new(null_mut()),
            init_state: AtomicU8::new(InitState::Uninitialized as u8),
        }
    }

    /// Try to create NAlloc and initialize arenas immediately.
    ///
    /// Returns an error if arena allocation fails, allowing the caller
    /// to handle the failure gracefully.
    pub fn try_new() -> Result<Self, AllocFailed> {
        let nalloc = Self::new();
        nalloc.try_init()?;
        Ok(nalloc)
    }

    /// Try to initialize arenas.
    ///
    /// Returns Ok if initialization succeeds or was already done.
    /// Returns Err if initialization fails.
    fn try_init(&self) -> Result<(), AllocFailed> {
        let state = self.init_state.load(Ordering::Acquire);

        match state {
            s if s == InitState::Initialized as u8 => Ok(()),
            s if s == InitState::Fallback as u8 => {
                Err(AllocFailed::with_kind(0, AllocErrorKind::OutOfMemory))
            }
            _ => {
                let ptr = self.init_arenas();
                if ptr.is_null() {
                    Err(AllocFailed::with_kind(0, AllocErrorKind::OutOfMemory))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Initialize the arenas if not already done.
    ///
    /// This uses a spin-lock pattern with atomic state to prevent
    /// recursive allocation issues and handle initialization failures gracefully.
    #[cold]
    #[inline(never)]
    fn init_arenas(&self) -> *mut ArenaManager {
        // Fast path: already initialized
        let state = self.init_state.load(Ordering::Acquire);
        if state == InitState::Initialized as u8 {
            return self.arenas.load(Ordering::Acquire);
        }
        if state == InitState::Fallback as u8 {
            return null_mut();
        }

        // Try to acquire initialization lock
        if self
            .init_state
            .compare_exchange(
                InitState::Uninitialized as u8,
                InitState::Initializing as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // We won the race - initialize
            match ArenaManager::new() {
                Ok(manager) => {
                    // Use system allocator to avoid recursive allocation
                    let layout = Layout::new::<ArenaManager>();
                    let raw = unsafe { System.alloc(layout) as *mut ArenaManager };

                    if raw.is_null() {
                        // Failed to allocate manager struct - enter fallback mode
                        eprintln!("[nalloc] Warning: Failed to allocate ArenaManager struct, using system allocator");
                        self.init_state
                            .store(InitState::Fallback as u8, Ordering::Release);
                        return null_mut();
                    }

                    unsafe {
                        std::ptr::write(raw, manager);
                    }
                    self.arenas.store(raw, Ordering::Release);
                    self.init_state
                        .store(InitState::Initialized as u8, Ordering::Release);
                    return raw;
                }
                Err(e) => {
                    // Arena allocation failed - enter fallback mode
                    eprintln!(
                        "[nalloc] Warning: Arena initialization failed ({}), using system allocator",
                        e
                    );
                    self.init_state
                        .store(InitState::Fallback as u8, Ordering::Release);
                    return null_mut();
                }
            }
        }

        // Another thread is initializing - spin wait
        loop {
            std::hint::spin_loop();
            let state = self.init_state.load(Ordering::Acquire);

            match state {
                s if s == InitState::Initialized as u8 => {
                    return self.arenas.load(Ordering::Acquire);
                }
                s if s == InitState::Fallback as u8 => {
                    return null_mut();
                }
                _ => continue,
            }
        }
    }

    /// Check if NAlloc is operating in fallback mode (using system allocator).
    #[inline]
    pub fn is_fallback_mode(&self) -> bool {
        self.init_state.load(Ordering::Relaxed) == InitState::Fallback as u8
    }

    /// Check if NAlloc is fully initialized with arenas.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.init_state.load(Ordering::Relaxed) == InitState::Initialized as u8
    }

    #[inline(always)]
    fn get_arenas(&self) -> Option<&ArenaManager> {
        let state = self.init_state.load(Ordering::Acquire);

        if state == InitState::Initialized as u8 {
            let ptr = self.arenas.load(Ordering::Acquire);
            if !ptr.is_null() {
                return Some(unsafe { &*ptr });
            }
        }

        if state == InitState::Uninitialized as u8 || state == InitState::Initializing as u8 {
            let ptr = self.init_arenas();
            if !ptr.is_null() {
                return Some(unsafe { &*ptr });
            }
        }

        None
    }

    /// Access the witness arena directly.
    ///
    /// Use this for allocating sensitive private inputs that need
    /// zero-initialization and secure wiping.
    ///
    /// # Panics
    ///
    /// Panics if arena initialization failed. Use `try_witness()` for
    /// fallible access.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zk_nalloc::NAlloc;
    ///
    /// let alloc = NAlloc::new();
    /// let witness = alloc.witness();
    /// let secret_ptr = witness.alloc(256, 8);
    /// assert!(!secret_ptr.is_null());
    ///
    /// // Securely wipe when done
    /// unsafe { witness.secure_wipe(); }
    /// ```
    #[inline]
    pub fn witness(&self) -> WitnessArena {
        self.try_witness()
            .expect("Arena initialization failed - use try_witness() for fallible access")
    }

    /// Try to access the witness arena.
    ///
    /// Returns `None` if arena initialization failed.
    #[inline]
    pub fn try_witness(&self) -> Option<WitnessArena> {
        self.get_arenas().map(|a| WitnessArena::new(a.witness()))
    }

    /// Access the polynomial arena directly.
    ///
    /// Use this for FFT/NTT-friendly polynomial coefficient vectors.
    /// Provides 64-byte alignment by default for SIMD operations.
    ///
    /// # Panics
    ///
    /// Panics if arena initialization failed. Use `try_polynomial()` for
    /// fallible access.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zk_nalloc::NAlloc;
    ///
    /// let alloc = NAlloc::new();
    /// let poly = alloc.polynomial();
    /// let coeffs = poly.alloc_fft_friendly(1024); // 1K coefficients
    /// assert!(!coeffs.is_null());
    /// assert_eq!((coeffs as usize) % 64, 0); // 64-byte aligned
    /// ```
    #[inline]
    pub fn polynomial(&self) -> PolynomialArena {
        self.try_polynomial()
            .expect("Arena initialization failed - use try_polynomial() for fallible access")
    }

    /// Try to access the polynomial arena.
    ///
    /// Returns `None` if arena initialization failed.
    #[inline]
    pub fn try_polynomial(&self) -> Option<PolynomialArena> {
        self.get_arenas()
            .map(|a| PolynomialArena::new(a.polynomial()))
    }

    /// Access the scratch arena directly.
    ///
    /// Use this for temporary computation space.
    ///
    /// # Panics
    ///
    /// Panics if arena initialization failed. Use `try_scratch()` for
    /// fallible access.
    #[inline]
    pub fn scratch(&self) -> std::sync::Arc<BumpAlloc> {
        self.try_scratch()
            .expect("Arena initialization failed - use try_scratch() for fallible access")
    }

    /// Try to access the scratch arena.
    ///
    /// Returns `None` if arena initialization failed.
    #[inline]
    pub fn try_scratch(&self) -> Option<std::sync::Arc<BumpAlloc>> {
        self.get_arenas().map(|a| a.scratch())
    }

    /// Reset all arenas, freeing all allocated memory.
    ///
    /// The witness arena is securely wiped before reset.
    ///
    /// # Safety
    /// This will invalidate all previously allocated memory.
    ///
    /// # Note
    /// Does nothing if operating in fallback mode.
    pub unsafe fn reset_all(&self) {
        if let Some(arenas) = self.get_arenas() {
            arenas.reset_all();
        }
    }

    /// Get statistics about arena usage.
    ///
    /// Returns `None` if operating in fallback mode.
    ///
    /// Useful for monitoring memory consumption and tuning arena sizes.
    pub fn stats(&self) -> Option<ArenaStats> {
        self.get_arenas().map(|a| a.stats())
    }

    /// Get statistics, returning default stats if in fallback mode.
    pub fn stats_or_default(&self) -> ArenaStats {
        self.stats().unwrap_or(ArenaStats {
            witness_used: 0,
            witness_capacity: 0,
            polynomial_used: 0,
            polynomial_capacity: 0,
            scratch_used: 0,
            scratch_capacity: 0,
            #[cfg(feature = "fallback")]
            witness_fallback_bytes: 0,
            #[cfg(feature = "fallback")]
            polynomial_fallback_bytes: 0,
            #[cfg(feature = "fallback")]
            scratch_fallback_bytes: 0,
        })
    }
}

impl Default for NAlloc {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: NAlloc uses atomic operations for all shared state
unsafe impl Send for NAlloc {}
unsafe impl Sync for NAlloc {}

unsafe impl GlobalAlloc for NAlloc {
    #[inline(always)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        debug_assert!(layout.size() > 0);
        debug_assert!(layout.align() > 0);
        debug_assert!(layout.align().is_power_of_two());

        // Try to use arenas
        if let Some(arenas) = self.get_arenas() {
            // Strategy:
            // 1. Large allocations (>threshold) go to Polynomial Arena (likely vectors)
            // 2. Smaller allocations go to Scratch Arena
            // 3. User can explicitly use Witness Arena via NAlloc::witness()

            if layout.size() > LARGE_ALLOC_THRESHOLD {
                arenas.polynomial().alloc(layout.size(), layout.align())
            } else {
                arenas.scratch().alloc(layout.size(), layout.align())
            }
        } else {
            // Fallback to system allocator
            System.alloc(layout)
        }
    }

    #[inline(always)]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // In fallback mode, we need to actually deallocate
        if self.is_fallback_mode() {
            System.dealloc(ptr, layout);
            return;
        }

        // For arena allocations, deallocation is a no-op.
        // Memory is reclaimed by calling reset() on the arena.
        // However, if fallback feature is enabled and this was a fallback
        // allocation, it would have been allocated from System.
        // Currently we don't track which allocations are from fallback,
        // so we can't distinguish. For now, treat as no-op.
        // TODO: Track fallback allocations for proper deallocation
    }

    #[inline(always)]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        debug_assert!(!ptr.is_null());
        debug_assert!(layout.size() > 0);
        debug_assert!(new_size > 0);

        let old_size = layout.size();

        // If the new size is smaller or equal, just return the same pointer.
        // (The bump allocator doesn't shrink.)
        if new_size <= old_size {
            return ptr;
        }

        // Allocate a new block
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let new_ptr = self.alloc(new_layout);

        if new_ptr.is_null() {
            return null_mut();
        }

        // Copy the old data
        copy_nonoverlapping(ptr, new_ptr, old_size);

        // Dealloc the old pointer (no-op for bump allocator, but semantically correct)
        self.dealloc(ptr, layout);

        new_ptr
    }

    #[inline(always)]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = self.alloc(layout);
        if !ptr.is_null() {
            // Note: mmap'd memory is already zeroed, but we zero anyway for
            // recycled memory or if user specifically requested zeroed allocation.
            std::ptr::write_bytes(ptr, 0, layout.size());
        }
        ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::GlobalAlloc;

    #[test]
    fn test_global_alloc_api() {
        let alloc = NAlloc::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();
        unsafe {
            let ptr = alloc.alloc(layout);
            assert!(!ptr.is_null());
            // Check that we can write to it
            ptr.write(42);
            assert_eq!(ptr.read(), 42);
        }
    }

    #[test]
    fn test_try_new() {
        // This should succeed on any reasonable system
        let result = NAlloc::try_new();
        assert!(result.is_ok());

        let alloc = result.unwrap();
        assert!(alloc.is_initialized());
        assert!(!alloc.is_fallback_mode());
    }

    #[test]
    fn test_fallback_mode_detection() {
        let alloc = NAlloc::new();
        // Force initialization
        let _ = alloc.stats();

        // Should be initialized (not fallback) on a normal system
        assert!(alloc.is_initialized() || alloc.is_fallback_mode());
    }

    #[test]
    fn test_try_accessors() {
        let alloc = NAlloc::new();

        // These should return Some on a normal system
        assert!(alloc.try_witness().is_some());
        assert!(alloc.try_polynomial().is_some());
        assert!(alloc.try_scratch().is_some());
    }

    #[test]
    fn test_realloc() {
        let alloc = NAlloc::new();
        let layout = Layout::from_size_align(64, 8).unwrap();
        unsafe {
            let ptr = alloc.alloc(layout);
            assert!(!ptr.is_null());

            // Write some data
            for i in 0..64 {
                ptr.add(i).write(i as u8);
            }

            // Realloc to a larger size
            let new_ptr = alloc.realloc(ptr, layout, 128);
            assert!(!new_ptr.is_null());

            // Verify data was copied
            for i in 0..64 {
                assert_eq!(new_ptr.add(i).read(), i as u8);
            }
        }
    }

    #[test]
    fn test_alloc_zeroed() {
        let alloc = NAlloc::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();
        unsafe {
            let ptr = alloc.alloc_zeroed(layout);
            assert!(!ptr.is_null());

            // Verify memory is zeroed
            for i in 0..1024 {
                assert_eq!(*ptr.add(i), 0);
            }
        }
    }

    #[test]
    fn test_stats() {
        let alloc = NAlloc::new();

        // Trigger arena initialization with an allocation
        let layout = Layout::from_size_align(1024, 8).unwrap();
        unsafe {
            let _ = alloc.alloc(layout);
        }

        let stats = alloc.stats();
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert!(stats.scratch_used >= 1024);
        assert!(stats.total_capacity() > 0);
    }

    #[test]
    fn test_stats_or_default() {
        let alloc = NAlloc::new();

        // Should work even before initialization
        let stats = alloc.stats_or_default();
        assert!(stats.total_capacity() >= 0);
    }

    #[test]
    fn test_large_allocation_routing() {
        let alloc = NAlloc::new();

        // Small allocation (<1MB) should go to scratch
        let small_layout = Layout::from_size_align(1024, 8).unwrap();
        unsafe {
            let _ = alloc.alloc(small_layout);
        }

        let stats_after_small = alloc.stats().unwrap();
        assert!(stats_after_small.scratch_used >= 1024);

        // Large allocation (>1MB) should go to polynomial
        let large_layout = Layout::from_size_align(2 * 1024 * 1024, 64).unwrap();
        unsafe {
            let _ = alloc.alloc(large_layout);
        }

        let stats_after_large = alloc.stats().unwrap();
        assert!(stats_after_large.polynomial_used >= 2 * 1024 * 1024);
    }

    #[test]
    fn test_concurrent_init() {
        use std::sync::Arc;
        use std::thread;

        let alloc = Arc::new(NAlloc::new());
        let mut handles = vec![];

        // Spawn multiple threads that try to initialize simultaneously
        for _ in 0..8 {
            let alloc = Arc::clone(&alloc);
            handles.push(thread::spawn(move || {
                let layout = Layout::from_size_align(64, 8).unwrap();
                unsafe {
                    let ptr = alloc.alloc(layout);
                    assert!(!ptr.is_null());
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // After all threads complete, should be in a consistent state
        assert!(alloc.is_initialized() || alloc.is_fallback_mode());
    }
}
