//! Witness Arena for nalloc.
//!
//! The `WitnessArena` provides a security-hardened interface for allocating
//! private ZK inputs (witnesses). Key features:
//!
//! - **Conditional zero on allocation**: Only zeroes recycled memory.
//! - **Secure wipe on reset**: Zeroes all memory before recycling using volatile writes.

use crate::bump::BumpAlloc;
use std::sync::Arc;

/// Specialized handle for Witness memory.
///
/// Ensures zeroing on allocation (for recycled memory) and secure wiping on reset.
pub struct WitnessArena {
    inner: Arc<BumpAlloc>,
}

impl WitnessArena {
    /// Create a new `WitnessArena` wrapping a `BumpAlloc`.
    #[inline]
    pub fn new(inner: Arc<BumpAlloc>) -> Self {
        Self { inner }
    }

    /// Allocate witness data.
    ///
    /// The returned memory is **zero-initialized** for security:
    /// - Fresh memory from `mmap` is already zeroed by the OS.
    /// - Recycled memory (after `secure_wipe`) is explicitly zeroed here.
    ///
    /// This optimization avoids redundant zeroing on first use while
    /// maintaining security guarantees for recycled memory.
    #[inline]
    pub fn alloc(&self, size: usize, align: usize) -> *mut u8 {
        debug_assert!(size > 0);
        debug_assert!(align > 0);

        // Issue #15: Read is_recycled BEFORE calling alloc to prevent race condition.
        // If we read after alloc, another thread could call secure_wipe() between
        // our allocation and the is_recycled check.
        let was_recycled = self.inner.is_recycled();

        let ptr = self.inner.alloc(size, align);
        if !ptr.is_null() && was_recycled {
            // Only zero if this memory has been recycled.
            // Fresh mmap'd memory is already zero (OS guarantee on Linux/macOS/Windows).
            unsafe {
                std::ptr::write_bytes(ptr, 0, size);
            }
        }
        ptr
    }

    /// Allocate witness data with explicit zero guarantee.
    ///
    /// Use this when you need a hard guarantee of zero-initialization,
    /// regardless of whether the memory has been recycled.
    #[inline]
    pub fn alloc_zeroed(&self, size: usize, align: usize) -> *mut u8 {
        debug_assert!(size > 0);
        debug_assert!(align > 0);

        let ptr = self.inner.alloc(size, align);
        if !ptr.is_null() {
            unsafe {
                std::ptr::write_bytes(ptr, 0, size);
            }
        }
        ptr
    }

    /// Securely wipe all witness data and reset the arena.
    ///
    /// Uses platform-specific secure zeroing (volatile writes) to ensure
    /// the data is actually erased and cannot be recovered.
    ///
    /// # Safety
    /// All previously allocated witness memory becomes invalid.
    #[inline]
    pub unsafe fn secure_wipe(&self) {
        self.inner.secure_reset();
    }

    /// Get the remaining capacity in bytes.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.inner.remaining()
    }

    /// Get the number of bytes currently allocated.
    #[inline]
    pub fn used(&self) -> usize {
        self.inner.used()
    }

    /// Get the total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Check if the arena has been recycled (wiped and reset).
    #[inline]
    pub fn is_recycled(&self) -> bool {
        self.inner.is_recycled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaManager;

    #[test]
    fn test_fresh_memory_not_double_zeroed() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 1024 * 1024, 1024 * 1024).unwrap();
        let witness = WitnessArena::new(manager.witness());

        // First allocation - should not trigger zeroing (fresh mmap is already zero)
        assert!(!witness.is_recycled());

        let ptr = witness.alloc(1024, 8);
        assert!(!ptr.is_null());

        // Verify it's zero (from OS)
        unsafe {
            for i in 0..1024 {
                assert_eq!(*ptr.add(i), 0);
            }
        }
    }

    #[test]
    fn test_recycled_memory_is_zeroed() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 1024 * 1024, 1024 * 1024).unwrap();
        let witness = WitnessArena::new(manager.witness());

        // Allocate and write secret data
        let ptr = witness.alloc(1024, 8);
        unsafe {
            std::ptr::write_bytes(ptr, 0xFF, 1024);
        }

        // Secure wipe
        unsafe { witness.secure_wipe() };

        // Now arena is recycled
        assert!(witness.is_recycled());

        // New allocation should be zeroed
        let ptr2 = witness.alloc(1024, 8);
        assert!(!ptr2.is_null());

        unsafe {
            for i in 0..1024 {
                assert_eq!(*ptr2.add(i), 0);
            }
        }
    }

    #[test]
    fn test_alloc_zeroed_always_zeroes() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 1024 * 1024, 1024 * 1024).unwrap();
        let witness = WitnessArena::new(manager.witness());

        // Even without recycling, alloc_zeroed should zero
        let ptr = witness.alloc_zeroed(1024, 8);
        assert!(!ptr.is_null());

        unsafe {
            for i in 0..1024 {
                assert_eq!(*ptr.add(i), 0);
            }
        }
    }
}
