//! Polynomial Arena for nalloc.
//!
//! The `PolynomialArena` is optimized for FFT/NTT operations:
//!
//! - **64-byte alignment**: Ensures data fits cache lines for SIMD operations.
//! - **4KB page alignment**: Optionally available for huge vector allocations.
//! - **Massive capacity**: Pre-reserved for 1GB+ polynomial vectors.

use crate::bump::BumpAlloc;
use crate::config::{CACHE_LINE_ALIGN, PAGE_ALIGN};
use std::sync::Arc;

/// Specialized handle for Polynomial and FFT data.
///
/// Optimized for cache-line alignment and massive vectors.
pub struct PolynomialArena {
    inner: Arc<BumpAlloc>,
}

impl PolynomialArena {
    /// Create a new `PolynomialArena` wrapping a `BumpAlloc`.
    #[inline]
    pub fn new(inner: Arc<BumpAlloc>) -> Self {
        Self { inner }
    }

    /// Allocate polynomial data with 64-byte alignment for optimal FFT/NTT performance.
    ///
    /// This alignment is critical for SIMD-accelerated operations:
    /// - **AVX-512**: Requires 64-byte alignment
    /// - **AVX/AVX2**: Benefits from 32-byte alignment (64 is a superset)
    /// - **Cache efficiency**: Modern cache lines are 64 bytes
    #[inline]
    pub fn alloc_fft_friendly(&self, size: usize) -> *mut u8 {
        debug_assert!(size > 0);
        self.inner.alloc(size, CACHE_LINE_ALIGN)
    }

    /// Allocate huge vectors with page alignment (4096 bytes).
    ///
    /// Use this for vectors exceeding a few megabytes. Benefits:
    /// - **TLB efficiency**: Reduces translation lookaside buffer misses
    /// - **Huge pages**: May enable transparent huge page usage on Linux
    /// - **DMA compatibility**: Required for some hardware accelerators
    #[inline]
    pub fn alloc_huge(&self, size: usize) -> *mut u8 {
        debug_assert!(size > 0);
        self.inner.alloc(size, PAGE_ALIGN)
    }

    /// Allocate with custom alignment.
    ///
    /// Use this when you have specific alignment requirements.
    /// Alignment must be a power of two.
    #[inline]
    pub fn alloc(&self, size: usize, align: usize) -> *mut u8 {
        debug_assert!(size > 0);
        debug_assert!(align > 0);
        debug_assert!(align.is_power_of_two());
        self.inner.alloc(size, align)
    }

    /// Allocate a typed slice of elements with appropriate alignment.
    ///
    /// This is a convenience method for allocating arrays of field elements
    /// or other ZK primitive types.
    ///
    /// # Safety
    /// The returned pointer must be properly aligned for type T.
    /// The caller is responsible for initializing the memory.
    #[inline]
    pub unsafe fn alloc_slice<T>(&self, count: usize) -> *mut T {
        debug_assert!(count > 0);
        let size = count * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>().max(CACHE_LINE_ALIGN);
        self.inner.alloc(size, align) as *mut T
    }

    /// Reset the polynomial arena.
    ///
    /// # Safety
    /// All previously allocated polynomial memory becomes invalid.
    #[inline]
    pub unsafe fn reset(&self) {
        self.inner.reset();
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaManager;

    #[test]
    fn test_fft_alignment() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 2 * 1024 * 1024, 1024 * 1024).unwrap();
        let poly = PolynomialArena::new(manager.polynomial());

        for _ in 0..100 {
            let ptr = poly.alloc_fft_friendly(1024);
            assert!(!ptr.is_null());
            assert_eq!(
                (ptr as usize) % CACHE_LINE_ALIGN,
                0,
                "FFT allocation not 64-byte aligned"
            );
        }
    }

    #[test]
    fn test_huge_alignment() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 2 * 1024 * 1024, 1024 * 1024).unwrap();
        let poly = PolynomialArena::new(manager.polynomial());

        for _ in 0..10 {
            let ptr = poly.alloc_huge(64 * 1024);
            assert!(!ptr.is_null());
            assert_eq!(
                (ptr as usize) % PAGE_ALIGN,
                0,
                "Huge allocation not page-aligned"
            );
        }
    }

    #[test]
    fn test_typed_slice_allocation() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 2 * 1024 * 1024, 1024 * 1024).unwrap();
        let poly = PolynomialArena::new(manager.polynomial());

        // Allocate u64 field elements
        let ptr: *mut u64 = unsafe { poly.alloc_slice(1024) };
        assert!(!ptr.is_null());
        assert_eq!(
            (ptr as usize) % std::mem::align_of::<u64>(),
            0,
            "Slice not aligned for u64"
        );

        // Write and read
        unsafe {
            for i in 0..1024 {
                *ptr.add(i) = i as u64;
            }
            for i in 0..1024 {
                assert_eq!(*ptr.add(i), i as u64);
            }
        }
    }

    #[test]
    fn test_custom_alignment() {
        let manager = ArenaManager::with_sizes(1024 * 1024, 2 * 1024 * 1024, 1024 * 1024).unwrap();
        let poly = PolynomialArena::new(manager.polynomial());

        // Test various power-of-two alignments
        for align_pow in 0..12 {
            let align = 1usize << align_pow;
            let ptr = poly.alloc(64, align);
            assert!(!ptr.is_null());
            assert_eq!(
                (ptr as usize) % align,
                0,
                "Custom alignment {} failed",
                align
            );
        }
    }
}
