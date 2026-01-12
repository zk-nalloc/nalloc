//! Core bump allocator for nalloc.
//!
//! A bump allocator is the fastest possible allocator: it simply increments
//! a pointer. This module provides a thread-safe, atomic bump allocator
//! optimized for ZK prover workloads with fallback support.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::NonNull;
use std::sync::atomic::{compiler_fence, AtomicBool, AtomicUsize, Ordering};

use crate::config::SECURE_WIPE_PATTERN;

/// A fast, lock-free bump allocator with fallback support.
///
/// Thread-safety is achieved via atomic compare-and-swap on the cursor.
/// This allows multiple threads to allocate concurrently without locks,
/// though there may be occasional retries on contention.
///
/// When the arena is exhausted and the `fallback` feature is enabled,
/// allocations fall back to the system allocator.
pub struct BumpAlloc {
    /// Base pointer of the memory region (never changes after init).
    base: NonNull<u8>,
    /// End pointer of the memory region (never changes after init).
    limit: NonNull<u8>,
    /// Current allocation cursor (atomically updated).
    cursor: AtomicUsize,
    /// Tracks whether the arena has been recycled (reset after use).
    /// Used to optimize zero-initialization in WitnessArena.
    is_recycled: AtomicBool,
    /// Counter for fallback allocations (for monitoring).
    #[cfg(feature = "fallback")]
    fallback_count: AtomicUsize,
    /// Total bytes allocated via fallback.
    #[cfg(feature = "fallback")]
    fallback_bytes: AtomicUsize,
}

impl BumpAlloc {
    /// Create a new bump allocator from a raw memory block.
    ///
    /// # Safety
    /// The memory block `[base, base+size)` must be valid and writable.
    #[inline]
    pub unsafe fn new(base: *mut u8, size: usize) -> Self {
        debug_assert!(!base.is_null());
        debug_assert!(size > 0);

        let base_nn = NonNull::new_unchecked(base);
        let limit_nn = NonNull::new_unchecked(base.add(size));

        Self {
            base: base_nn,
            limit: limit_nn,
            cursor: AtomicUsize::new(base as usize),
            is_recycled: AtomicBool::new(false),
            #[cfg(feature = "fallback")]
            fallback_count: AtomicUsize::new(0),
            #[cfg(feature = "fallback")]
            fallback_bytes: AtomicUsize::new(0),
        }
    }

    /// Get the base pointer of this allocator.
    #[inline]
    pub fn base_ptr(&self) -> *mut u8 {
        self.base.as_ptr()
    }

    /// Allocate memory with the given size and alignment.
    ///
    /// Returns a null pointer if there is not enough space and fallback is disabled.
    /// With the `fallback` feature, falls back to system allocator.
    #[inline(always)]
    pub fn alloc(&self, size: usize, align: usize) -> *mut u8 {
        debug_assert!(size > 0);
        debug_assert!(align > 0);
        debug_assert!(align.is_power_of_two());

        loop {
            let current = self.cursor.load(Ordering::Relaxed);
            let aligned = (current + align - 1) & !(align - 1);
            let next = aligned + size;

            if next > self.limit.as_ptr() as usize {
                // Arena exhausted
                return self.handle_exhaustion(size, align);
            }

            if self
                .cursor
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return aligned as *mut u8;
            }
            // Contention: another thread allocated concurrently. Retry.
        }
    }

    /// Handle arena exhaustion - either fallback or return null.
    #[cold]
    #[inline(never)]
    fn handle_exhaustion(&self, size: usize, align: usize) -> *mut u8 {
        #[cfg(debug_assertions)]
        {
            eprintln!(
                "[nalloc] Arena exhausted: requested {} bytes (align {}), remaining {} bytes",
                size,
                align,
                self.remaining()
            );
        }

        #[cfg(feature = "fallback")]
        {
            // Fall back to system allocator
            let layout = match Layout::from_size_align(size, align) {
                Ok(l) => l,
                Err(_) => return std::ptr::null_mut(),
            };

            let ptr = unsafe { System.alloc(layout) };

            if !ptr.is_null() {
                self.fallback_count.fetch_add(1, Ordering::Relaxed);
                self.fallback_bytes.fetch_add(size, Ordering::Relaxed);

                #[cfg(debug_assertions)]
                eprintln!("[nalloc] Fallback allocation: {} bytes", size);
            }

            ptr
        }

        #[cfg(not(feature = "fallback"))]
        {
            std::ptr::null_mut()
        }
    }

    /// Check if this arena has been recycled (reset after initial use).
    #[inline]
    pub fn is_recycled(&self) -> bool {
        self.is_recycled.load(Ordering::Relaxed)
    }

    /// Get the number of fallback allocations (only with `fallback` feature).
    #[cfg(feature = "fallback")]
    #[inline]
    pub fn fallback_count(&self) -> usize {
        self.fallback_count.load(Ordering::Relaxed)
    }

    /// Get the total bytes allocated via fallback (only with `fallback` feature).
    #[cfg(feature = "fallback")]
    #[inline]
    pub fn fallback_bytes(&self) -> usize {
        self.fallback_bytes.load(Ordering::Relaxed)
    }

    /// Reset the bump pointer to the base.
    ///
    /// # Safety
    /// All previously allocated memory becomes invalid after this call.
    /// Note: Fallback allocations are NOT freed by reset - they must be
    /// individually deallocated or will be freed when the program exits.
    #[inline]
    pub unsafe fn reset(&self) {
        self.cursor
            .store(self.base.as_ptr() as usize, Ordering::SeqCst);
        self.is_recycled.store(true, Ordering::Release);

        #[cfg(feature = "fallback")]
        {
            // Reset fallback counters
            self.fallback_count.store(0, Ordering::Relaxed);
            self.fallback_bytes.store(0, Ordering::Relaxed);
        }
    }

    /// Zero out all memory in the arena and reset the cursor.
    ///
    /// This is critical for security-sensitive applications like ZK provers,
    /// where witness data must be wiped after use to prevent leakage.
    ///
    /// Uses volatile writes to prevent the compiler from optimizing away
    /// the zeroing operation (dead store elimination).
    ///
    /// # Safety
    /// All previously allocated memory becomes invalid after this call.
    #[inline]
    pub unsafe fn secure_reset(&self) {
        let base = self.base.as_ptr();
        let size = self.limit.as_ptr() as usize - base as usize;

        // Use volatile writes to prevent dead store elimination.
        // This ensures the memory is actually zeroed even if it's never read again.
        Self::volatile_memset(base, SECURE_WIPE_PATTERN, size);

        // Compiler fence to ensure the wipe completes before any subsequent operations.
        compiler_fence(Ordering::SeqCst);

        self.reset();
    }

    /// Volatile memset implementation that cannot be optimized away.
    ///
    /// This is critical for cryptographic security - we need to guarantee
    /// that sensitive data is actually erased from memory.
    #[inline(never)]
    unsafe fn volatile_memset(ptr: *mut u8, value: u8, len: usize) {
        // Method 1: Use platform-specific secure zeroing where available
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            // explicit_bzero is guaranteed not to be optimized away
            extern "C" {
                fn explicit_bzero(s: *mut libc::c_void, n: libc::size_t);
            }
            if value == 0 {
                explicit_bzero(ptr as *mut libc::c_void, len);
                return;
            }
        }

        #[cfg(target_vendor = "apple")]
        {
            // memset_s is guaranteed not to be optimized away (C11)
            extern "C" {
                fn memset_s(
                    s: *mut libc::c_void,
                    smax: libc::size_t,
                    c: libc::c_int,
                    n: libc::size_t,
                ) -> libc::c_int;
            }
            let _ = memset_s(ptr as *mut libc::c_void, len, value as libc::c_int, len);
            return;
        }

        #[cfg(target_os = "windows")]
        {
            // RtlSecureZeroMemory is guaranteed not to be optimized away
            extern "system" {
                fn RtlSecureZeroMemory(ptr: *mut u8, len: usize);
            }
            if value == 0 {
                RtlSecureZeroMemory(ptr, len);
                return;
            }
        }

        // Fallback: Volatile write loop (works everywhere, used when platform-specific not available)
        #[cfg(not(any(
            target_os = "linux",
            target_os = "android",
            target_vendor = "apple",
            target_os = "windows"
        )))]
        {
            // Using usize-sized writes for better performance
            let ptr_usize = ptr as *mut usize;
            let pattern_usize = if value == 0 {
                0usize
            } else {
                let mut p = 0usize;
                for i in 0..std::mem::size_of::<usize>() {
                    p |= (value as usize) << (i * 8);
                }
                p
            };

            let full_words = len / std::mem::size_of::<usize>();
            let remainder = len % std::mem::size_of::<usize>();

            // Write full usize words
            for i in 0..full_words {
                std::ptr::write_volatile(ptr_usize.add(i), pattern_usize);
            }

            // Write remaining bytes
            let remainder_ptr = ptr.add(full_words * std::mem::size_of::<usize>());
            for i in 0..remainder {
                std::ptr::write_volatile(remainder_ptr.add(i), value);
            }
        }
    }

    /// Returns the total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.limit.as_ptr() as usize - self.base.as_ptr() as usize
    }

    /// Returns the number of bytes currently allocated.
    #[inline]
    pub fn used(&self) -> usize {
        self.cursor.load(Ordering::Relaxed) - self.base.as_ptr() as usize
    }

    /// Returns the number of bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity() - self.used()
    }
}

// Safety: BumpAlloc can be shared across threads because:
// - `base` and `limit` are never modified after construction
// - `cursor` uses atomic operations for thread-safe updates
// - `is_recycled` uses atomic operations
unsafe impl Send for BumpAlloc {}
unsafe impl Sync for BumpAlloc {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonnull_safety() {
        let mut buffer = vec![0u8; 1024];
        let alloc = unsafe { BumpAlloc::new(buffer.as_mut_ptr(), buffer.len()) };

        assert_eq!(alloc.capacity(), 1024);
        assert_eq!(alloc.used(), 0);
        assert_eq!(alloc.remaining(), 1024);
        assert!(!alloc.is_recycled());
    }

    #[test]
    fn test_recycled_flag() {
        let mut buffer = vec![0u8; 1024];
        let alloc = unsafe { BumpAlloc::new(buffer.as_mut_ptr(), buffer.len()) };

        assert!(!alloc.is_recycled());

        let _ = alloc.alloc(64, 8);
        assert!(!alloc.is_recycled());

        unsafe { alloc.reset() };
        assert!(alloc.is_recycled());
    }

    #[test]
    fn test_secure_reset_zeroes_memory() {
        let mut buffer = vec![0xFFu8; 1024];
        let alloc = unsafe { BumpAlloc::new(buffer.as_mut_ptr(), buffer.len()) };

        // Allocate and write data
        let ptr = alloc.alloc(512, 8);
        assert!(!ptr.is_null());
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, 512);
        }

        // Secure reset
        unsafe { alloc.secure_reset() };

        // Verify memory is zeroed
        for i in 0..1024 {
            assert_eq!(buffer[i], 0, "Byte {} not zeroed", i);
        }
    }

    #[test]
    fn test_alignment() {
        let mut buffer = vec![0u8; 4096];
        let alloc = unsafe { BumpAlloc::new(buffer.as_mut_ptr(), buffer.len()) };

        // Test various alignments
        for align_pow in 0..8 {
            let align = 1usize << align_pow;
            let ptr = alloc.alloc(64, align);
            assert!(!ptr.is_null());
            assert_eq!((ptr as usize) % align, 0, "Alignment {} failed", align);
        }
    }

    #[test]
    #[cfg(feature = "fallback")]
    fn test_fallback_allocation() {
        // Create a tiny arena that will exhaust quickly
        let mut buffer = vec![0u8; 256];
        let alloc = unsafe { BumpAlloc::new(buffer.as_mut_ptr(), buffer.len()) };

        // Fill the arena
        let _ = alloc.alloc(256, 1);

        // This should trigger fallback
        let ptr = alloc.alloc(64, 8);
        assert!(!ptr.is_null(), "Fallback allocation should succeed");

        assert!(alloc.fallback_count() > 0, "Fallback count should increase");
        assert!(alloc.fallback_bytes() >= 64, "Fallback bytes should track");

        // Don't forget to free the fallback allocation
        unsafe {
            System.dealloc(ptr, Layout::from_size_align(64, 8).unwrap());
        }
    }

    #[test]
    #[cfg(not(feature = "fallback"))]
    fn test_exhaustion_returns_null() {
        let mut buffer = vec![0u8; 256];
        let alloc = unsafe { BumpAlloc::new(buffer.as_mut_ptr(), buffer.len()) };

        // Fill the arena
        let _ = alloc.alloc(256, 1);

        // This should return null without fallback
        let ptr = alloc.alloc(64, 8);
        assert!(
            ptr.is_null(),
            "Should return null when exhausted without fallback"
        );
    }
}
