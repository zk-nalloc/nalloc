# nalloc: ZK-Optimized Memory Allocator

A high-performance, deterministic, and security-hardened memory allocator specifically engineered for Zero-Knowledge Proof (ZKP) systems and cryptographic provers.

[![Crates.io](https://img.shields.io/crates/v/zk-nalloc.svg)](https://crates.io/crates/zk-nalloc)
[![Documentation](https://docs.rs/zk-nalloc/badge.svg)](https://docs.rs/zk-nalloc)
[![License](https://img.shields.io/crates/l/zk-nalloc.svg)](LICENSE)

## Why nalloc?

General-purpose allocators (malloc, jemalloc) are designed for long-lived, heterogeneous workloads. ZK provers, however, exhibit extreme memory patterns: massive short-lived vectors, sensitive witness data, and performance-critical FFT/NTT operations. 

`nalloc` addresses these unique requirements:

- **Performance**: O(1) allocation via Atomic Bump Allocation.
- **Cache-Friendliness**: Guaranteed 64-byte alignment (AVX-512/SIMD optimal) for polynomials.
- **Security**: Hardened volatile wiping of witness data to prevent leakage.
- **Framework-Agnostic**: Works with any ZK system - no external dependencies.
- **Production-Ready**: Panic-free error handling with graceful fallback.

---

## v0.2.0 Features

### 🛡️ Industrial-Grade Improvements

| Feature | Description |
|---------|-------------|
| **Drop Safety** | Prevents use-after-free via Arc reference counting |
| **Panic-Free** | Gracefully falls back to system allocator on errors |
| **Fallback Allocator** | Continues working when arena exhausted |
| **Huge Pages** | 2MB/1GB huge page support (Linux) |
| **Guard Pages** | Buffer overflow protection at arena boundaries |
| **Memory Locking** | Prevent witness data from being swapped |

### Cargo Features

```toml
[dependencies]
zk-nalloc = { version = "0.2.0", features = ["fallback"] }
```

| Feature | Description | Default |
|---------|-------------|---------|
| `fallback` | Fall back to system allocator when arena exhausted | ✅ |
| `huge-pages` | Linux 2MB/1GB huge page support | Optional |
| `guard-pages` | Guard pages at arena boundaries (Linux) | Optional |
| `mlock` | Lock witness memory to prevent swapping | Optional |

---

## Supported ZK Systems

`nalloc` is a pure memory primitive designed to work with **any** ZK framework:

| Framework | Use Case |
|-----------|----------|
| **Halo2** | Plonkish circuits, recursive proofs |
| **Plonky2** | Fast recursive STARKs |
| **Risc0** | zkVM execution |
| **SP1** | Succinct zkVM |
| **Miden** | STARK-based VM |
| **Cairo** | StarkNet proofs |
| **Circom/SnarkJS** | Groth16, PLONK circuits |
| **Arkworks** | General-purpose ZK toolkit |

---

## Architecture: Specialized Arenas

`nalloc` partitions memory into three specialized pools to eliminate fragmentation and enforce security boundaries:

| Arena | Purpose | Optimization | Security |
|-------|---------|--------------|-----------|
| **Witness** | Secret inputs / Witnesses | Zero-on-recycled-alloc | **Secure Wipe** (Volatile) |
| **Polynomial** | FFT / NTT Vectors | 64-byte & Page Alignment | Isolated from scratch |
| **Scratch** | Temp computation space | High-speed bump allocation | O(1) Batch Reset |

---

## Core Features

### 1. Hardened Witness Security
Witness data is handled with extreme caution. The `secure_wipe()` method uses platform-specific primitives that the compiler cannot optimize away:
- **Linux**: `explicit_bzero`
- **macOS**: `memset_s`
- **Windows**: `RtlSecureZeroMemory`
- **Fallback**: Atomic volatile write loops with memory fences.

### 2. Framework-Agnostic Design
`nalloc` has **zero external ZK dependencies**. It provides pure memory primitives that any proving system can utilize. Lock-free `AtomicPtr` initialization prevents recursive allocation deadlocks during prover startup.

### 3. Graceful Error Handling
```rust
use zk_nalloc::NAlloc;

// Fallible initialization
match NAlloc::try_new() {
    Ok(alloc) => { /* use alloc */ }
    Err(e) => eprintln!("Allocation failed: {}", e),
}

// Or use the auto-fallback version
let alloc = NAlloc::new(); // Falls back to system allocator if needed
if alloc.is_fallback_mode() {
    eprintln!("Running in fallback mode");
}
```

### 4. Monitoring & Stats
Easily track your circuit's memory footprint:
```rust
let stats = alloc.stats().expect("Stats available");
println!("Witness used: {} bytes", stats.witness_used);
println!("Polynomial used: {} bytes", stats.polynomial_used);
#[cfg(feature = "fallback")]
println!("Fallback bytes: {} bytes", stats.total_fallback_bytes());
```

---

## Usage

### As a Global Allocator
Add to your `Cargo.toml`:
```toml
[dependencies]
zk-nalloc = "0.2.0"
```

In your `main.rs` or `lib.rs`:
```rust
use zk_nalloc::NAlloc;

#[global_allocator]
static ALLOC: NAlloc = NAlloc::new();

fn main() {
    // All allocations are now routed to specialized arenas
    let data = vec![0u8; 1024];
}
```

### Manual Arena Control
For maximum performance, access arenas directly:
```rust
use zk_nalloc::NAlloc;

fn prove() {
    let nalloc = NAlloc::new();
    
    // 1. Allocate witness data
    let witness = nalloc.witness();
    let secret = witness.alloc(1024, 64);
    
    // 2. Compute proof with your preferred ZK framework...
    
    // 3. Securely erase traces
    unsafe { witness.secure_wipe(); }
}
```

### With Guard Pages (Security)
```rust
use zk_nalloc::ArenaManager;

// Allocate with guard pages for buffer overflow detection
#[cfg(feature = "guard-pages")]
let manager = ArenaManager::with_guard_pages(
    128 * 1024 * 1024,  // witness
    1024 * 1024 * 1024, // polynomial
    256 * 1024 * 1024,  // scratch
)?;
```

### With Memory Locking (Anti-Swap)
```rust
#[cfg(feature = "mlock")]
{
    let manager = ArenaManager::new()?;
    manager.lock_witness()?; // Prevent swapping of sensitive data
}
```

---

## Platform Support & Verification

`nalloc` provides cross-platform abstractions for low-level memory management:
- **macOS**: `mach_vm_allocate` / `mach_vm_deallocate`
- **Linux**: `mmap` / `munmap` (via `rustix`) with huge page support
- **Windows**: `VirtualAlloc` / `VirtualFree` with guard pages

---

## Performance Benchmark

| Task | System Alloc | nalloc | Speedup |
|------|--------------|---------|---------|
| 10k Small Allocs | ~150 μs | ~50 μs | **3x** |
| Large FFT Vector | ~10 μs | ~8 μs | **1.2x** |
| Batch Dealloc | O(N) | **O(1)** | **∞** |

---

## License

Licensed under the GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE) for details.

---

## Contributing

Designed with ❤️ for the ZK community. Contributions for Huge Page support or new platform backends are welcome.
