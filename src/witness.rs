//! Witness Arena for nalloc.
//!
//! The `WitnessArena` provides a security-hardened interface for allocating
//! private ZK inputs (witnesses). Key features:
//!
//! - **Conditional zero on allocation**: Only zeroes recycled memory.
//! - **Secure wipe on reset**: Zeroes all memory before recycling using volatile writes.
//!
//! # Security Limitations
//!
//! - **Concurrent `secure_wipe()` is unsafe.** The caller must ensure no other
//!   thread is allocating from this arena when `secure_wipe()` is called.
//! - **Stack/register copies are not covered.** Values copied out of arena memory
//!   into local Rust variables are not erased by `secure_wipe()`. Use `zeroize`
//!   for stack-allocated secrets.
//! - **`static NAlloc` does not wipe on program exit.** Rust does not run `Drop`
//!   for `static` items. Call `secure_wipe()` explicitly before the prover exits.

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
    /// - Recycled memory (after `secure_wipe`) is explicitly zeroed with volatile
    ///   writes to prevent dead-store elimination by the compiler.
    ///
    /// This optimization avoids redundant zeroing on first use while
    /// maintaining security guarantees for recycled memory.
    ///
    /// # Security Notes
    ///
    /// **Concurrent `secure_wipe()` is unsafe.** Calling `secure_wipe()` while
    /// another thread may still be allocating from this arena is undefined
    /// behaviour: the wipe resets the arena cursor to base, invalidating all
    /// outstanding pointers. The caller is responsible for ensuring all arena
    /// users have finished before calling `secure_wipe()`.
    ///
    /// **Stack and register copies are not covered.** When caller code copies
    /// witness values into local variables (e.g. `let x = *ptr;`), those copies
    /// live in CPU registers and on the stack. Wiping the arena does **not**
    /// erase those copies. Use a crate such as `zeroize` for stack-allocated
    /// secrets, and prefer keeping field elements in arena memory for as long
    /// as possible.
    #[inline]
    pub fn alloc(&self, size: usize, align: usize) -> *mut u8 {
        debug_assert!(size > 0);
        debug_assert!(align > 0);

        // Read is_recycled with Acquire ordering BEFORE calling alloc.
        //
        // Acquire pairs with the Release store in BumpAlloc::reset(), establishing
        // a happens-before edge: if we observe is_recycled=true we are guaranteed
        // to see all writes (including the volatile zeroing) done by the resetting
        // thread before we return memory to the caller.
        //
        // We read BEFORE alloc so that the check and the allocation are ordered:
        // if a reset races with this alloc (which is a caller bug — see Safety above),
        // we conservatively zero the returned pointer rather than skipping zeroing.
        let was_recycled = self.inner.is_recycled();

        let ptr = self.inner.alloc(size, align);
        if !ptr.is_null() && was_recycled {
            // Only zero if this memory has been recycled.
            // Fresh mmap'd memory is already zero (OS guarantee on Linux/macOS/Windows).
            //
            // Use volatile writes so the compiler cannot elide the zeroing via
            // dead-store elimination — the same reason secure_reset() uses volatile.
            unsafe {
                volatile_zero(ptr, size);
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
                volatile_zero(ptr, size);
            }
        }
        ptr
    }

    /// Securely wipe all witness data and reset the arena.
    ///
    /// Uses platform-specific secure zeroing (`explicit_bzero` / `memset_s` /
    /// `RtlSecureZeroMemory` / volatile-write fallback) followed by a
    /// `SeqCst` fence to ensure the zeroing is visible across threads before
    /// the cursor is reset.
    ///
    /// # Safety
    ///
    /// 1. **All previously allocated witness memory becomes invalid.** Any
    ///    pointer obtained from `alloc` or `alloc_zeroed` on this arena must
    ///    not be read or written after this call.
    ///
    /// 2. **Must be called only when no other thread is allocating.** This
    ///    method is not safe to call concurrently with `alloc` on the same
    ///    arena. Use external synchronization (e.g. a barrier or channel) to
    ///    ensure all allocating threads have finished before calling this.
    ///
    /// 3. **Does not cover stack/register copies.** Values copied out of arena
    ///    memory into local variables are not erased. Use `zeroize` or
    ///    explicit `write_volatile` on stack buffers for those.
    ///
    /// 4. **`static NAlloc` does not call this automatically on program exit.**
    ///    When `NAlloc` is used as `#[global_allocator] static ALLOC: NAlloc`,
    ///    Rust does not run `Drop` for statics. You must call
    ///    `alloc.witness().secure_wipe()` explicitly before the prover exits.
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

/// Zero `len` bytes at `ptr` using volatile word-sized writes.
///
/// Volatile writes cannot be removed by the compiler's dead-store elimination
/// pass, which is critical when zeroing memory that will be re-used but whose
/// previous values are no longer read (from the compiler's perspective).
#[inline(never)]
unsafe fn volatile_zero(ptr: *mut u8, len: usize) {
    let word_size = std::mem::size_of::<usize>();
    let full_words = len / word_size;
    let remainder = len % word_size;

    let ptr_usize = ptr as *mut usize;
    for i in 0..full_words {
        std::ptr::write_volatile(ptr_usize.add(i), 0usize);
    }

    let tail = ptr.add(full_words * word_size);
    for i in 0..remainder {
        std::ptr::write_volatile(tail.add(i), 0u8);
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