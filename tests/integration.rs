use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use zk_nalloc::NAlloc;

// ============================================================================
// Determinism Tests
// ============================================================================

#[test]
fn test_determinism() {
    let alloc = NAlloc::new();
    let layout = Layout::from_size_align(1024, 64).unwrap();

    unsafe {
        let ptr1 = alloc.alloc(layout);
        let ptr2 = alloc.alloc(layout);

        let diff = (ptr2 as usize) - (ptr1 as usize);
        assert_eq!(
            diff, 1024,
            "Allocations should be sequential in a bump allocator"
        );
    }
}

#[test]
fn test_deterministic_layout_after_reset() {
    let alloc = NAlloc::new();
    let layout = Layout::from_size_align(512, 8).unwrap();

    unsafe {
        // First round of allocations
        let ptr1_round1 = alloc.alloc(layout);
        let ptr2_round1 = alloc.alloc(layout);

        let offset1 = ptr2_round1 as usize - ptr1_round1 as usize;

        // Reset
        alloc.reset_all();

        // Second round should have same relative layout
        let ptr1_round2 = alloc.alloc(layout);
        let ptr2_round2 = alloc.alloc(layout);

        let offset2 = ptr2_round2 as usize - ptr1_round2 as usize;

        assert_eq!(
            offset1, offset2,
            "Allocation offsets should be deterministic after reset"
        );
    }
}

// ============================================================================
// Security Tests
// ============================================================================

#[test]
fn test_witness_security() {
    let alloc = NAlloc::new();
    let witness = alloc.witness();

    let size = 100;
    let ptr = witness.alloc(size, 8);

    unsafe {
        // Verify it was zeroed
        for i in 0..size {
            assert_eq!(*ptr.add(i), 0);
        }

        // Write some "secret" data
        for i in 0..size {
            *ptr.add(i) = 0xFF;
        }

        // Secure reset
        witness.secure_wipe();

        // Verify it was wiped
        for i in 0..size {
            assert_eq!(*ptr.add(i), 0);
        }
    }
}

#[test]
fn test_witness_large_secure_wipe() {
    let alloc = NAlloc::new();
    let witness = alloc.witness();

    // Allocate 1MB of witness data
    let size = 1024 * 1024;
    let ptr = witness.alloc(size, 64);
    assert!(!ptr.is_null());

    unsafe {
        // Fill with sensitive data
        std::ptr::write_bytes(ptr, 0xDE, size);

        // Verify data was written
        for i in (0..size).step_by(4096) {
            assert_eq!(*ptr.add(i), 0xDE);
        }

        // Secure wipe
        witness.secure_wipe();

        // Verify complete erasure
        for i in (0..size).step_by(4096) {
            assert_eq!(*ptr.add(i), 0, "Byte {} not wiped", i);
        }
    }
}

#[test]
fn test_recycled_witness_is_zeroed() {
    let alloc = NAlloc::new();
    let witness = alloc.witness();

    // First allocation
    let ptr1 = witness.alloc(1024, 8);
    unsafe {
        std::ptr::write_bytes(ptr1, 0xAB, 1024);
    }

    // Secure wipe
    unsafe { witness.secure_wipe() };

    // New allocation from recycled pool
    let ptr2 = witness.alloc(1024, 8);

    // Must be zeroed
    unsafe {
        for i in 0..1024 {
            assert_eq!(
                *ptr2.add(i),
                0,
                "Recycled memory not zeroed at offset {}",
                i
            );
        }
    }
}

// ============================================================================
// Alignment Tests
// ============================================================================

#[test]
fn test_alignment() {
    let alloc = NAlloc::new();
    let poly = alloc.polynomial();

    let ptr = poly.alloc_fft_friendly(1024);
    assert_eq!(
        (ptr as usize) % 64,
        0,
        "FFT-friendly allocation must be 64-byte aligned"
    );

    let ptr_huge = poly.alloc_huge(4096);
    assert_eq!(
        (ptr_huge as usize) % 4096,
        0,
        "Huge allocation must be 4096-byte aligned"
    );
}

#[test]
fn test_all_power_of_two_alignments() {
    let alloc = NAlloc::new();
    let poly = alloc.polynomial();

    // Test alignments from 1 byte to 4KB
    for align_pow in 0..13 {
        let align = 1usize << align_pow;
        let ptr = poly.alloc(64, align);
        assert!(!ptr.is_null(), "Allocation failed for align={}", align);
        assert_eq!(
            (ptr as usize) % align,
            0,
            "Alignment check failed for align={}",
            align
        );
    }
}

#[test]
fn test_mixed_alignment_preserves_subsequent() {
    let alloc = NAlloc::new();
    let poly = alloc.polynomial();

    // Allocate with small alignment
    let _ptr1 = poly.alloc(17, 1); // Odd size, 1-byte align

    // Next allocation with large alignment should still be aligned
    let ptr2 = poly.alloc_fft_friendly(1024);
    assert_eq!(
        (ptr2 as usize) % 64,
        0,
        "FFT alignment broken after small allocation"
    );

    let ptr3 = poly.alloc_huge(4096);
    assert_eq!(
        (ptr3 as usize) % 4096,
        0,
        "Page alignment broken after small allocation"
    );
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[test]
fn test_concurrent_allocation() {
    let alloc = Arc::new(NAlloc::new());
    let allocation_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let alloc = Arc::clone(&alloc);
            let allocation_count = Arc::clone(&allocation_count);
            let error_count = Arc::clone(&error_count);

            thread::spawn(move || {
                for _ in 0..1000 {
                    let layout = Layout::from_size_align(64, 8).unwrap();
                    unsafe {
                        let ptr = alloc.alloc(layout);
                        if ptr.is_null() {
                            error_count.fetch_add(1, Ordering::Relaxed);
                        } else {
                            // Write to verify it's accessible
                            ptr.write(42);
                            allocation_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }

    assert_eq!(allocation_count.load(Ordering::Relaxed), 8 * 1000);
    assert_eq!(error_count.load(Ordering::Relaxed), 0);
}

#[test]
fn test_concurrent_witness_access() {
    let alloc = Arc::new(NAlloc::new());
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let alloc = Arc::clone(&alloc);
            thread::spawn(move || {
                let witness = alloc.witness();
                for _ in 0..100 {
                    let ptr = witness.alloc(256, 8);
                    assert!(!ptr.is_null());

                    // Verify zero-initialization
                    unsafe {
                        for i in 0..256 {
                            assert_eq!(*ptr.add(i), 0);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }
}

#[test]
fn test_concurrent_polynomial_fft_alignment() {
    let alloc = Arc::new(NAlloc::new());
    let alignment_errors = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let alloc = Arc::clone(&alloc);
            let alignment_errors = Arc::clone(&alignment_errors);

            thread::spawn(move || {
                let poly = alloc.polynomial();
                for _ in 0..500 {
                    let ptr = poly.alloc_fft_friendly(1024);
                    if !ptr.is_null() && (ptr as usize) % 64 != 0 {
                        alignment_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }

    assert_eq!(
        alignment_errors.load(Ordering::Relaxed),
        0,
        "Concurrent FFT allocations had alignment errors"
    );
}

// ============================================================================
// Arena Exhaustion Tests
// ============================================================================

// NOTE: These tests only work when the `fallback` feature is disabled.
// With fallback enabled, exhausted arenas fall back to system allocator,
// so they never return null.

#[test]
#[cfg(not(feature = "fallback"))]
fn test_scratch_arena_exhaustion() {
    // Create a custom allocator with small arenas for testing
    let manager = zk_nalloc::ArenaManager::with_sizes(
        1024 * 1024, // 1 MB witness
        1024 * 1024, // 1 MB polynomial
        1024 * 1024, // 1 MB scratch
    )
    .unwrap();

    let scratch = manager.scratch();

    let mut alloc_count = 0;
    loop {
        let ptr = scratch.alloc(64 * 1024, 8); // 64 KB chunks
        if ptr.is_null() {
            break;
        }
        alloc_count += 1;
        if alloc_count > 100 {
            panic!("Should have exhausted 1MB scratch with 64KB allocations");
        }
    }

    // Should get about 16 allocations (1MB / 64KB)
    assert!(
        alloc_count >= 15,
        "Should allocate at least 15 chunks, got {}",
        alloc_count
    );
    assert!(
        alloc_count <= 17,
        "Should not allocate more than 17 chunks, got {}",
        alloc_count
    );
}

#[test]
#[cfg(not(feature = "fallback"))]
fn test_exhaustion_returns_null() {
    let manager = zk_nalloc::ArenaManager::with_sizes(
        1024, // Tiny arenas for quick exhaustion
        1024, 1024,
    )
    .unwrap();

    let scratch = manager.scratch();

    // First allocation should succeed
    let ptr1 = scratch.alloc(512, 8);
    assert!(!ptr1.is_null());

    // Second allocation should succeed
    let ptr2 = scratch.alloc(512, 8);
    assert!(!ptr2.is_null());

    // Third allocation should fail (exhausted)
    let ptr3 = scratch.alloc(512, 8);
    assert!(ptr3.is_null(), "Should return null when exhausted");
}

#[test]
#[cfg(not(feature = "fallback"))]
fn test_exhaustion_recovery_after_reset() {
    let manager = zk_nalloc::ArenaManager::with_sizes(
        1024 * 1024,
        1024 * 1024,
        1024, // Small scratch for testing
    )
    .unwrap();

    let scratch = manager.scratch();

    // Exhaust the arena
    while !scratch.alloc(256, 8).is_null() {}

    // Should be exhausted
    assert!(scratch.alloc(256, 8).is_null());

    // Reset
    unsafe { scratch.reset() };

    // Should be able to allocate again
    let ptr = scratch.alloc(256, 8);
    assert!(!ptr.is_null(), "Should recover after reset");
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[test]
fn test_arena_stats() {
    let alloc = NAlloc::new();

    // Initialize arenas
    let _ = alloc.witness().alloc(1024, 8);
    let _ = alloc.polynomial().alloc_fft_friendly(2048);
    unsafe {
        alloc.alloc(Layout::from_size_align(512, 8).unwrap());
    }

    let stats = alloc.stats().expect("Stats should be available");

    assert!(stats.witness_used >= 1024, "Witness usage not tracked");
    assert!(
        stats.polynomial_used >= 2048,
        "Polynomial usage not tracked"
    );
    assert!(stats.scratch_used >= 512, "Scratch usage not tracked");
    assert!(stats.total_used() >= 1024 + 2048 + 512);
    assert!(stats.total_capacity() > 0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_single_byte_allocation() {
    let alloc = NAlloc::new();
    let layout = Layout::from_size_align(1, 1).unwrap();

    unsafe {
        let ptr = alloc.alloc(layout);
        assert!(!ptr.is_null());
        ptr.write(0x42);
        assert_eq!(ptr.read(), 0x42);
    }
}

#[test]
fn test_max_alignment_allocation() {
    let alloc = NAlloc::new();
    let poly = alloc.polynomial();

    // 4KB alignment (page size)
    let ptr = poly.alloc(4096, 4096);
    assert!(!ptr.is_null());
    assert_eq!((ptr as usize) % 4096, 0);
}

#[test]
fn test_realloc_shrink() {
    let alloc = NAlloc::new();
    let layout = Layout::from_size_align(1024, 8).unwrap();

    unsafe {
        let ptr = alloc.alloc(layout);
        assert!(!ptr.is_null());

        // Write data
        std::ptr::write_bytes(ptr, 0xAB, 1024);

        // Shrink (should return same pointer for bump allocator)
        let new_ptr = alloc.realloc(ptr, layout, 512);
        assert_eq!(ptr, new_ptr, "Shrink should return same pointer");
    }
}

#[test]
fn test_realloc_grow_preserves_data() {
    let alloc = NAlloc::new();
    let layout = Layout::from_size_align(64, 8).unwrap();

    unsafe {
        let ptr = alloc.alloc(layout);

        // Write pattern
        for i in 0..64 {
            ptr.add(i).write(i as u8);
        }

        // Grow
        let new_ptr = alloc.realloc(ptr, layout, 256);
        assert!(!new_ptr.is_null());

        // Verify data preserved
        for i in 0..64 {
            assert_eq!(
                new_ptr.add(i).read(),
                i as u8,
                "Data not preserved at offset {}",
                i
            );
        }
    }
}
