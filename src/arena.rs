//! Arena Manager for nalloc.
//!
//! The `ArenaManager` pre-allocates large, specialized memory pools
//! during initialization. This avoids system call overhead during
//! hot proof computation paths.

use crate::bump::BumpAlloc;
use crate::config::{POLY_ARENA_SIZE, SCRATCH_ARENA_SIZE, WITNESS_ARENA_SIZE};
use crate::sys;

use std::sync::Arc;

/// Manages multiple specialized memory arenas.
///
/// Each arena is optimized for a specific purpose:
/// - **Witness Arena**: For private ZK inputs, with secure wiping.
/// - **Polynomial Arena**: For FFT/NTT coefficient vectors.
/// - **Scratch Arena**: For temporary computation buffers.
///
/// # Drop Safety
///
/// The `ArenaManager` tracks the number of outstanding arena handles.
/// On drop, it verifies that all handles have been released before
/// deallocating memory. If handles are still in use, the memory is
/// intentionally leaked to prevent use-after-free (with a warning).
pub struct ArenaManager {
    witness: Arc<BumpAlloc>,
    polynomial: Arc<BumpAlloc>,
    scratch: Arc<BumpAlloc>,
    /// Raw pointers for deallocation (since we can't get them after Arc is dropped)
    witness_ptr: *mut u8,
    poly_ptr: *mut u8,
    scratch_ptr: *mut u8,
    /// Sizes for deallocation
    witness_size: usize,
    poly_size: usize,
    scratch_size: usize,
    /// Flag to indicate this manager uses guard pages
    #[cfg(feature = "guard-pages")]
    #[allow(dead_code)]
    has_guard_pages: bool,
}

impl ArenaManager {
    /// Create a new ArenaManager with default sizes.
    ///
    /// This will allocate a total of ~1.4 GB of virtual memory.
    /// Note: On modern OSes, virtual memory is cheap; physical pages
    /// are only allocated when touched.
    pub fn new() -> Result<Self, crate::platform::AllocFailed> {
        Self::with_sizes(WITNESS_ARENA_SIZE, POLY_ARENA_SIZE, SCRATCH_ARENA_SIZE)
    }

    /// Create a new ArenaManager with custom sizes.
    ///
    /// Use this for fine-tuned configurations based on your circuit size.
    pub fn with_sizes(
        witness_size: usize,
        poly_size: usize,
        scratch_size: usize,
    ) -> Result<Self, crate::platform::AllocFailed> {
        let witness_ptr = sys::alloc(witness_size)?;
        let poly_ptr = sys::alloc(poly_size)?;
        let scratch_ptr = sys::alloc(scratch_size)?;

        Ok(Self {
            witness: Arc::new(unsafe { BumpAlloc::new(witness_ptr, witness_size) }),
            polynomial: Arc::new(unsafe { BumpAlloc::new(poly_ptr, poly_size) }),
            scratch: Arc::new(unsafe { BumpAlloc::new(scratch_ptr, scratch_size) }),
            witness_ptr,
            poly_ptr,
            scratch_ptr,
            witness_size,
            poly_size,
            scratch_size,
            #[cfg(feature = "guard-pages")]
            has_guard_pages: false,
        })
    }

    /// Create arenas with guard pages for buffer overflow protection.
    #[cfg(feature = "guard-pages")]
    pub fn with_guard_pages(
        witness_size: usize,
        poly_size: usize,
        scratch_size: usize,
    ) -> Result<Self, crate::platform::AllocFailed> {
        let witness_guarded = sys::alloc_with_guards(witness_size)?;
        let poly_guarded = sys::alloc_with_guards(poly_size)?;
        let scratch_guarded = sys::alloc_with_guards(scratch_size)?;

        Ok(Self {
            witness: Arc::new(unsafe { BumpAlloc::new(witness_guarded.ptr, witness_size) }),
            polynomial: Arc::new(unsafe { BumpAlloc::new(poly_guarded.ptr, poly_size) }),
            scratch: Arc::new(unsafe { BumpAlloc::new(scratch_guarded.ptr, scratch_size) }),
            witness_ptr: witness_guarded.base_ptr,
            poly_ptr: poly_guarded.base_ptr,
            scratch_ptr: scratch_guarded.base_ptr,
            witness_size: witness_guarded.total_size,
            poly_size: poly_guarded.total_size,
            scratch_size: scratch_guarded.total_size,
            has_guard_pages: true,
        })
    }

    /// Lock witness memory to prevent swapping (important for sensitive data).
    #[cfg(feature = "mlock")]
    pub fn lock_witness(&self) -> Result<(), crate::platform::AllocFailed> {
        sys::mlock(self.witness.base_ptr(), self.witness.capacity())
    }

    /// Unlock previously locked witness memory.
    #[cfg(feature = "mlock")]
    pub fn unlock_witness(&self) -> Result<(), crate::platform::AllocFailed> {
        sys::munlock(self.witness.base_ptr(), self.witness.capacity())
    }

    /// Get a handle to the witness arena.
    #[inline]
    pub fn witness(&self) -> Arc<BumpAlloc> {
        self.witness.clone()
    }

    /// Get a handle to the polynomial arena.
    #[inline]
    pub fn polynomial(&self) -> Arc<BumpAlloc> {
        self.polynomial.clone()
    }

    /// Get a handle to the scratch arena.
    #[inline]
    pub fn scratch(&self) -> Arc<BumpAlloc> {
        self.scratch.clone()
    }

    /// Reset all arenas.
    ///
    /// The witness arena is securely wiped (zeroed) before reset.
    ///
    /// # Safety
    /// This will invalidate all memory previously allocated from these arenas.
    pub unsafe fn reset_all(&self) {
        self.witness.secure_reset();
        self.polynomial.reset();
        self.scratch.reset();
    }

    /// Get statistics about arena usage.
    pub fn stats(&self) -> ArenaStats {
        ArenaStats {
            witness_used: self.witness.used(),
            witness_capacity: self.witness.capacity(),
            polynomial_used: self.polynomial.used(),
            polynomial_capacity: self.polynomial.capacity(),
            scratch_used: self.scratch.used(),
            scratch_capacity: self.scratch.capacity(),
            #[cfg(feature = "fallback")]
            witness_fallback_bytes: self.witness.fallback_bytes(),
            #[cfg(feature = "fallback")]
            polynomial_fallback_bytes: self.polynomial.fallback_bytes(),
            #[cfg(feature = "fallback")]
            scratch_fallback_bytes: self.scratch.fallback_bytes(),
        }
    }

    /// Check if all arena handles have been released.
    ///
    /// Returns true if this ArenaManager is the sole owner of all arenas.
    pub fn is_sole_owner(&self) -> bool {
        Arc::strong_count(&self.witness) == 1
            && Arc::strong_count(&self.polynomial) == 1
            && Arc::strong_count(&self.scratch) == 1
    }

    /// Get the reference counts for each arena (for debugging).
    pub fn ref_counts(&self) -> (usize, usize, usize) {
        (
            Arc::strong_count(&self.witness),
            Arc::strong_count(&self.polynomial),
            Arc::strong_count(&self.scratch),
        )
    }
}

/// Statistics about arena memory usage.
#[derive(Debug, Clone, Copy)]
pub struct ArenaStats {
    pub witness_used: usize,
    pub witness_capacity: usize,
    pub polynomial_used: usize,
    pub polynomial_capacity: usize,
    pub scratch_used: usize,
    pub scratch_capacity: usize,
    #[cfg(feature = "fallback")]
    pub witness_fallback_bytes: usize,
    #[cfg(feature = "fallback")]
    pub polynomial_fallback_bytes: usize,
    #[cfg(feature = "fallback")]
    pub scratch_fallback_bytes: usize,
}

impl ArenaStats {
    /// Total memory currently in use.
    pub fn total_used(&self) -> usize {
        self.witness_used + self.polynomial_used + self.scratch_used
    }

    /// Total memory capacity across all arenas.
    pub fn total_capacity(&self) -> usize {
        self.witness_capacity + self.polynomial_capacity + self.scratch_capacity
    }

    /// Total bytes allocated via fallback (only with `fallback` feature).
    #[cfg(feature = "fallback")]
    pub fn total_fallback_bytes(&self) -> usize {
        self.witness_fallback_bytes + self.polynomial_fallback_bytes + self.scratch_fallback_bytes
    }
}

impl Drop for ArenaManager {
    fn drop(&mut self) {
        // SAFETY CHECK: Verify we are the sole owner of all arenas
        // If not, we cannot safely deallocate the memory as it may still be in use

        let (witness_refs, poly_refs, scratch_refs) = self.ref_counts();

        if witness_refs > 1 || poly_refs > 1 || scratch_refs > 1 {
            // CRITICAL: Other references exist! We must leak the memory to prevent
            // use-after-free. This is a bug in the caller's code but we handle it safely.
            eprintln!(
                "[nalloc] WARNING: ArenaManager dropped with outstanding references! \
                 witness={}, polynomial={}, scratch={}. Memory will be leaked to prevent \
                 use-after-free. This is a bug in your code - ensure all arena handles \
                 are dropped before the ArenaManager.",
                witness_refs - 1,
                poly_refs - 1,
                scratch_refs - 1
            );

            // Intentionally leak by not deallocating
            return;
        }

        // We are the sole owner - safe to deallocate
        // First, securely wipe witness data
        unsafe {
            self.witness.secure_reset();
        }

        // Best-effort deallocation - log errors but don't panic
        if let Err(e) = sys::dealloc(self.witness_ptr, self.witness_size) {
            eprintln!(
                "[nalloc] Warning: Failed to deallocate witness arena: {}",
                e
            );
        }
        if let Err(e) = sys::dealloc(self.poly_ptr, self.poly_size) {
            eprintln!(
                "[nalloc] Warning: Failed to deallocate polynomial arena: {}",
                e
            );
        }
        if let Err(e) = sys::dealloc(self.scratch_ptr, self.scratch_size) {
            eprintln!(
                "[nalloc] Warning: Failed to deallocate scratch arena: {}",
                e
            );
        }
    }
}

// Safety: ArenaManager uses Arc internally for thread-safe sharing
unsafe impl Send for ArenaManager {}
unsafe impl Sync for ArenaManager {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_manager_creation() {
        // Use smaller sizes for testing
        let manager = ArenaManager::with_sizes(1024 * 1024, 2 * 1024 * 1024, 1024 * 1024).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.witness_capacity, 1024 * 1024);
        assert_eq!(stats.polynomial_capacity, 2 * 1024 * 1024);
        assert_eq!(stats.scratch_capacity, 1024 * 1024);
        assert_eq!(stats.total_used(), 0);
    }

    #[test]
    fn test_arena_stats() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 2 * 1024 * 1024, 1024 * 1024).unwrap();

        // Allocate some memory
        let _ = manager.witness().alloc(1024, 8);
        let _ = manager.polynomial().alloc(2048, 64);
        let _ = manager.scratch().alloc(512, 8);

        let stats = manager.stats();
        assert!(stats.witness_used >= 1024);
        assert!(stats.polynomial_used >= 2048);
        assert!(stats.scratch_used >= 512);
    }

    #[test]
    fn test_drop_deallocates() {
        // This test verifies that Drop runs without panicking
        {
            let _manager = ArenaManager::with_sizes(1024 * 1024, 1024 * 1024, 1024 * 1024).unwrap();
            // manager goes out of scope here, triggering Drop
        }
        // If we get here without crashing, deallocation worked
    }

    #[test]
    fn test_sole_owner_check() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 1024 * 1024, 1024 * 1024).unwrap();

        // Initially we should be sole owner
        assert!(manager.is_sole_owner());

        // Take a reference
        let _witness_handle = manager.witness();

        // Now we're not the sole owner
        assert!(!manager.is_sole_owner());

        // Drop the handle
        drop(_witness_handle);

        // We're sole owner again
        assert!(manager.is_sole_owner());
    }

    #[test]
    fn test_ref_counts() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 1024 * 1024, 1024 * 1024).unwrap();

        let (w, p, s) = manager.ref_counts();
        assert_eq!((w, p, s), (1, 1, 1));

        let _w1 = manager.witness();
        let _w2 = manager.witness();
        let _p1 = manager.polynomial();

        let (w, p, s) = manager.ref_counts();
        assert_eq!((w, p, s), (3, 2, 1));
    }

    #[test]
    #[cfg(all(target_os = "linux", feature = "guard-pages"))]
    fn test_guard_pages_creation() {
        let manager =
            ArenaManager::with_guard_pages(1024 * 1024, 1024 * 1024, 1024 * 1024).unwrap();

        // Verify we can allocate
        let ptr = manager.witness().alloc(1024, 8);
        assert!(!ptr.is_null());

        // Write to verify it's accessible
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, 1024);
        }
    }

    #[test]
    #[cfg(feature = "mlock")]
    fn test_mlock() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 1024 * 1024, 1024 * 1024).unwrap();

        // This may fail on systems without mlock permissions, so we just
        // check that it doesn't panic
        let _ = manager.lock_witness();
        let _ = manager.unlock_witness();
    }
}
