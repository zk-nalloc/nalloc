use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use zk_nalloc::NAlloc;
use std::alloc::{GlobalAlloc, Layout, System};
use std::time::Duration;

// We use the Global Allocator API for fairness
static N_ALLOC: NAlloc = NAlloc::new();

fn bench_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Allocation");
    group.measurement_time(Duration::from_secs(5));

    // Scenario 1: Small Witness-like allocations (Field Elements, 32 bytes)
    // This tests the overhead of the allocator for many small objects.
    let layout_small = Layout::from_size_align(32, 8).unwrap();

    group.throughput(Throughput::Elements(1));
    group.bench_function("Small Alloc (32 bytes) - System", |b| {
        b.iter(|| unsafe { System.alloc(layout_small) })
    });
    group.bench_function("Small Alloc (32 bytes) - nalloc", |b| {
        b.iter(|| unsafe { N_ALLOC.alloc(layout_small) })
    });

    // Scenario 2: Large Polynomial Vector (1 MB)
    // This tests the throughput for large contiguous memory blocks.
    let layout_large = Layout::from_size_align(1024 * 1024, 64).unwrap(); // 64-byte align for AVX

    group.throughput(Throughput::Bytes(1024 * 1024));
    group.bench_function("Large Alloc (1MB) - System", |b| {
        b.iter(|| unsafe { System.alloc(layout_large) })
    });
    group.bench_function("Large Alloc (1MB) - nalloc", |b| {
        b.iter(|| unsafe { N_ALLOC.alloc(layout_large) })
    });

    group.finish();
}

fn bench_alloc_dealloc_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pattern");

    // Scenario 3: "ZK Prover Cycle"
    // Allocate a bunch of things, then reset (free everything).
    // Note: System allocator has to free one by one. nalloc resets arena.

    const NUM_ELEMENTS: usize = 10_000;
    let layout = Layout::from_size_align(32, 8).unwrap();

    group.bench_function("Alloc 10k & Free One-by-One - System", |b| {
        b.iter(|| {
            // Collect pointers to free them later
            let mut ptrs = Vec::with_capacity(NUM_ELEMENTS);
            for _ in 0..NUM_ELEMENTS {
                let ptr = unsafe { System.alloc(layout) };
                ptrs.push(ptr);
            }
            // Free one by one
            for ptr in ptrs {
                unsafe { System.dealloc(ptr, layout) };
            }
        })
    });

    // For nalloc, we can just alloc and then reset the arena.
    // However, NAlloc::alloc is stateless (global), so to test reset we need direct access.
    // We'll use a fresh arena manager for this test to be fair and isolated.
    group.bench_function("Alloc 10k & Reset - nalloc", |b| {
        b.iter(|| {
            // In a real loop we might just reset, but here we want to measure the cycle.
            // NAlloc::reset_all() is the equivalent.
            for _ in 0..NUM_ELEMENTS {
                unsafe {
                    black_box(N_ALLOC.alloc(layout));
                }
            }
            unsafe {
                N_ALLOC.reset_all();
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_allocations, bench_alloc_dealloc_pattern);
criterion_main!(benches);
