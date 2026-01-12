//! Platform-specific memory allocation interface.
//!
//! This module provides an abstraction over the operating system's
//! virtual memory allocation APIs:
//! - **Linux**: `mmap` via `rustix` with huge pages and guard pages support
//! - **macOS**: `mach_vm_allocate` via `mach2`
//! - **Windows**: `VirtualAlloc` via direct FFI
//! - **Other Unix**: `mmap` via `libc`

use std::fmt;

/// Error type for system memory allocation failures.
#[derive(Debug, Clone, Copy)]
pub struct AllocFailed {
    /// The size that was requested.
    pub requested_size: usize,
    /// Platform-specific error code, if available.
    pub error_code: Option<i32>,
    /// Error kind for better diagnostics.
    pub kind: AllocErrorKind,
}

/// Classification of allocation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocErrorKind {
    /// Out of memory
    OutOfMemory,
    /// Invalid size or alignment
    InvalidArgument,
    /// Permission denied
    PermissionDenied,
    /// Huge pages not available
    HugePagesUnavailable,
    /// Memory lock failed
    MlockFailed,
    /// Unknown error
    Unknown,
}

impl std::error::Error for AllocFailed {}

impl fmt::Display for AllocFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.error_code {
            Some(code) => write!(
                f,
                "Memory allocation failed ({:?}): requested {} bytes, error code {}",
                self.kind, self.requested_size, code
            ),
            None => write!(
                f,
                "Memory allocation failed ({:?}): requested {} bytes",
                self.kind, self.requested_size
            ),
        }
    }
}

impl AllocFailed {
    /// Create a new allocation failure error.
    pub fn new(size: usize) -> Self {
        Self {
            requested_size: size,
            error_code: None,
            kind: AllocErrorKind::Unknown,
        }
    }

    pub fn with_code(size: usize, code: i32) -> Self {
        Self {
            requested_size: size,
            error_code: Some(code),
            kind: AllocErrorKind::Unknown,
        }
    }

    pub fn with_kind(size: usize, kind: AllocErrorKind) -> Self {
        Self {
            requested_size: size,
            error_code: None,
            kind,
        }
    }

    pub fn out_of_memory(size: usize) -> Self {
        Self::with_kind(size, AllocErrorKind::OutOfMemory)
    }

    pub fn huge_pages_unavailable(size: usize) -> Self {
        Self::with_kind(size, AllocErrorKind::HugePagesUnavailable)
    }

    pub fn mlock_failed(size: usize) -> Self {
        Self::with_kind(size, AllocErrorKind::MlockFailed)
    }
}

/// Huge page size options (Linux only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageSize {
    /// 2 MB huge pages (most common)
    Size2MB,
    /// 1 GB huge pages (requires explicit kernel configuration)
    Size1GB,
}

impl HugePageSize {
    pub fn bytes(&self) -> usize {
        match self {
            HugePageSize::Size2MB => 2 * 1024 * 1024,
            HugePageSize::Size1GB => 1024 * 1024 * 1024,
        }
    }
}

/// Result of a guarded allocation.
#[cfg(feature = "guard-pages")]
pub struct GuardedAlloc {
    /// The usable memory pointer (after the front guard page).
    pub ptr: *mut u8,
    /// Total allocation size including guard pages.
    pub total_size: usize,
    /// Usable size (excluding guard pages).
    pub usable_size: usize,
    /// Base pointer (start of front guard page).
    pub base_ptr: *mut u8,
}

/// Platform-specific memory allocation functions.
pub mod sys {
    use super::*;

    // ========================================================================
    // Linux Implementation (using rustix)
    // ========================================================================

    /// Allocate `size` bytes of virtual memory from the OS.
    #[cfg(target_os = "linux")]
    #[inline]
    pub fn alloc(size: usize) -> Result<*mut u8, AllocFailed> {
        use rustix::mm::{mmap_anonymous, MapFlags, ProtFlags};
        use std::ptr;

        debug_assert!(size > 0);

        unsafe {
            match mmap_anonymous(
                ptr::null_mut(),
                size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::PRIVATE | MapFlags::NORESERVE,
            ) {
                Ok(ptr) => Ok(ptr as *mut u8),
                Err(_) => Err(AllocFailed::out_of_memory(size)),
            }
        }
    }

    /// Allocate memory with huge pages (Linux only).
    #[cfg(all(target_os = "linux", feature = "huge-pages"))]
    pub fn alloc_huge(size: usize, huge_size: HugePageSize) -> Result<*mut u8, AllocFailed> {
        use rustix::mm::{mmap_anonymous, MapFlags, ProtFlags};
        use std::ptr;

        debug_assert!(size > 0);

        // Size must be aligned to huge page size
        let page_size = huge_size.bytes();
        let aligned_size = (size + page_size - 1) & !(page_size - 1);

        // MAP_HUGETLB flags
        let huge_flag = match huge_size {
            HugePageSize::Size2MB => 21 << 26, // MAP_HUGE_2MB
            HugePageSize::Size1GB => 30 << 26, // MAP_HUGE_1GB
        };

        unsafe {
            // First try with huge pages
            let flags = MapFlags::PRIVATE | MapFlags::from_bits_retain(0x40000 | huge_flag); // MAP_HUGETLB

            match mmap_anonymous(
                ptr::null_mut(),
                aligned_size,
                ProtFlags::READ | ProtFlags::WRITE,
                flags,
            ) {
                Ok(ptr) => Ok(ptr as *mut u8),
                Err(_) => Err(AllocFailed::huge_pages_unavailable(size)),
            }
        }
    }

    /// Allocate memory with guard pages at both ends.
    #[cfg(all(target_os = "linux", feature = "guard-pages"))]
    pub fn alloc_with_guards(size: usize) -> Result<GuardedAlloc, AllocFailed> {
        use rustix::mm::{mmap_anonymous, mprotect, MapFlags, MprotectFlags, ProtFlags};
        use std::ptr;

        const PAGE_SIZE: usize = 4096;

        // Allocate: guard_page + usable_memory + guard_page
        let total_size = PAGE_SIZE + size + PAGE_SIZE;

        unsafe {
            let base = match mmap_anonymous(
                ptr::null_mut(),
                total_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::PRIVATE,
            ) {
                Ok(ptr) => ptr as *mut u8,
                Err(_) => return Err(AllocFailed::out_of_memory(size)),
            };

            // Protect front guard page (no access)
            if mprotect(base as *mut _, PAGE_SIZE, MprotectFlags::empty()).is_err() {
                let _ = dealloc(base, total_size);
                return Err(AllocFailed::with_kind(
                    size,
                    AllocErrorKind::PermissionDenied,
                ));
            }

            // Protect rear guard page (no access)
            let rear_guard = base.add(PAGE_SIZE + size);
            if mprotect(rear_guard as *mut _, PAGE_SIZE, MprotectFlags::empty()).is_err() {
                let _ = dealloc(base, total_size);
                return Err(AllocFailed::with_kind(
                    size,
                    AllocErrorKind::PermissionDenied,
                ));
            }

            Ok(GuardedAlloc {
                ptr: base.add(PAGE_SIZE),
                total_size,
                usable_size: size,
                base_ptr: base,
            })
        }
    }

    /// Lock memory to prevent swapping (for sensitive data).
    #[cfg(all(target_os = "linux", feature = "mlock"))]
    pub fn mlock(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        unsafe {
            if libc::mlock(ptr as *const _, size) == 0 {
                Ok(())
            } else {
                Err(AllocFailed::mlock_failed(size))
            }
        }
    }

    /// Unlock previously locked memory.
    #[cfg(all(target_os = "linux", feature = "mlock"))]
    pub fn munlock(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        unsafe {
            if libc::munlock(ptr as *const _, size) == 0 {
                Ok(())
            } else {
                Err(AllocFailed::mlock_failed(size))
            }
        }
    }

    /// Deallocate memory previously allocated with `alloc`.
    #[cfg(target_os = "linux")]
    #[inline]
    pub fn dealloc(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        use rustix::mm::munmap;

        if ptr.is_null() {
            return Ok(());
        }

        unsafe {
            match munmap(ptr as *mut _, size) {
                Ok(()) => Ok(()),
                Err(_) => Err(AllocFailed::new(size)),
            }
        }
    }

    // ========================================================================
    // macOS Implementation (using mach2)
    // ========================================================================

    #[cfg(target_vendor = "apple")]
    #[inline]
    pub fn alloc(size: usize) -> Result<*mut u8, AllocFailed> {
        use mach2::kern_return::KERN_SUCCESS;
        use mach2::traps::mach_task_self;
        use mach2::vm::mach_vm_allocate;
        use mach2::vm_statistics::VM_FLAGS_ANYWHERE;
        use mach2::vm_types::{mach_vm_address_t, mach_vm_size_t};

        debug_assert!(size > 0);

        let task = unsafe { mach_task_self() };
        let mut address: mach_vm_address_t = 0;
        let vm_size: mach_vm_size_t = size as mach_vm_size_t;

        let retval = unsafe { mach_vm_allocate(task, &mut address, vm_size, VM_FLAGS_ANYWHERE) };

        if retval == KERN_SUCCESS {
            Ok(address as *mut u8)
        } else {
            Err(AllocFailed::with_code(size, retval))
        }
    }

    /// Lock memory on macOS.
    #[cfg(all(target_vendor = "apple", feature = "mlock"))]
    pub fn mlock(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        unsafe {
            if libc::mlock(ptr as *const _, size) == 0 {
                Ok(())
            } else {
                Err(AllocFailed::mlock_failed(size))
            }
        }
    }

    #[cfg(all(target_vendor = "apple", feature = "mlock"))]
    pub fn munlock(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        unsafe {
            if libc::munlock(ptr as *const _, size) == 0 {
                Ok(())
            } else {
                Err(AllocFailed::mlock_failed(size))
            }
        }
    }

    /// Allocate with guard pages on macOS.
    #[cfg(all(target_vendor = "apple", feature = "guard-pages"))]
    pub fn alloc_with_guards(size: usize) -> Result<GuardedAlloc, AllocFailed> {
        use mach2::kern_return::KERN_SUCCESS;
        use mach2::traps::mach_task_self;
        use mach2::vm::{mach_vm_allocate, mach_vm_protect};
        use mach2::vm_prot::{VM_PROT_NONE, VM_PROT_READ, VM_PROT_WRITE};
        use mach2::vm_statistics::VM_FLAGS_ANYWHERE;
        use mach2::vm_types::{mach_vm_address_t, mach_vm_size_t};

        const PAGE_SIZE: usize = 4096;
        let total_size = PAGE_SIZE + size + PAGE_SIZE;

        let task = unsafe { mach_task_self() };
        let mut address: mach_vm_address_t = 0;

        let retval = unsafe {
            mach_vm_allocate(
                task,
                &mut address,
                total_size as mach_vm_size_t,
                VM_FLAGS_ANYWHERE,
            )
        };

        if retval != KERN_SUCCESS {
            return Err(AllocFailed::out_of_memory(size));
        }

        let base = address as *mut u8;

        unsafe {
            // Protect front guard
            let ret = mach_vm_protect(task, address, PAGE_SIZE as mach_vm_size_t, 0, VM_PROT_NONE);
            if ret != KERN_SUCCESS {
                let _ = dealloc(base, total_size);
                return Err(AllocFailed::with_kind(
                    size,
                    AllocErrorKind::PermissionDenied,
                ));
            }

            // Protect rear guard
            let rear_addr = address + (PAGE_SIZE + size) as u64;
            let ret = mach_vm_protect(
                task,
                rear_addr,
                PAGE_SIZE as mach_vm_size_t,
                0,
                VM_PROT_NONE,
            );
            if ret != KERN_SUCCESS {
                let _ = dealloc(base, total_size);
                return Err(AllocFailed::with_kind(
                    size,
                    AllocErrorKind::PermissionDenied,
                ));
            }
        }

        Ok(GuardedAlloc {
            ptr: unsafe { base.add(PAGE_SIZE) },
            total_size,
            usable_size: size,
            base_ptr: base,
        })
    }

    /// Deallocate memory previously allocated with `alloc`.
    #[cfg(target_vendor = "apple")]
    #[inline]
    pub fn dealloc(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        use mach2::kern_return::KERN_SUCCESS;
        use mach2::traps::mach_task_self;
        use mach2::vm::mach_vm_deallocate;
        use mach2::vm_types::mach_vm_size_t;

        if ptr.is_null() {
            return Ok(());
        }

        let task = unsafe { mach_task_self() };
        let retval = unsafe { mach_vm_deallocate(task, ptr as u64, size as mach_vm_size_t) };

        if retval == KERN_SUCCESS {
            Ok(())
        } else {
            Err(AllocFailed::with_code(size, retval))
        }
    }

    // ========================================================================
    // Windows Implementation
    // ========================================================================

    #[cfg(target_os = "windows")]
    #[inline]
    pub fn alloc(size: usize) -> Result<*mut u8, AllocFailed> {
        use std::ptr;

        const MEM_COMMIT: u32 = 0x00001000;
        const MEM_RESERVE: u32 = 0x00002000;
        const PAGE_READWRITE: u32 = 0x04;

        extern "system" {
            fn VirtualAlloc(
                lpAddress: *mut u8,
                dwSize: usize,
                flAllocationType: u32,
                flProtect: u32,
            ) -> *mut u8;
        }

        debug_assert!(size > 0);

        let result = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };

        if result.is_null() {
            Err(AllocFailed::out_of_memory(size))
        } else {
            Ok(result)
        }
    }

    /// Allocate with guard pages on Windows.
    #[cfg(all(target_os = "windows", feature = "guard-pages"))]
    pub fn alloc_with_guards(size: usize) -> Result<GuardedAlloc, AllocFailed> {
        use std::ptr;

        const MEM_COMMIT: u32 = 0x00001000;
        const MEM_RESERVE: u32 = 0x00002000;
        const PAGE_READWRITE: u32 = 0x04;
        const PAGE_NOACCESS: u32 = 0x01;
        const PAGE_SIZE: usize = 4096;

        extern "system" {
            fn VirtualAlloc(
                lpAddress: *mut u8,
                dwSize: usize,
                flAllocationType: u32,
                flProtect: u32,
            ) -> *mut u8;
            fn VirtualProtect(
                lpAddress: *mut u8,
                dwSize: usize,
                flNewProtect: u32,
                lpflOldProtect: *mut u32,
            ) -> i32;
        }

        let total_size = PAGE_SIZE + size + PAGE_SIZE;

        let base = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                total_size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };

        if base.is_null() {
            return Err(AllocFailed::out_of_memory(size));
        }

        unsafe {
            let mut old_protect: u32 = 0;

            // Protect front guard
            if VirtualProtect(base, PAGE_SIZE, PAGE_NOACCESS, &mut old_protect) == 0 {
                let _ = dealloc(base, total_size);
                return Err(AllocFailed::with_kind(
                    size,
                    AllocErrorKind::PermissionDenied,
                ));
            }

            // Protect rear guard
            let rear_guard = base.add(PAGE_SIZE + size);
            if VirtualProtect(rear_guard, PAGE_SIZE, PAGE_NOACCESS, &mut old_protect) == 0 {
                let _ = dealloc(base, total_size);
                return Err(AllocFailed::with_kind(
                    size,
                    AllocErrorKind::PermissionDenied,
                ));
            }
        }

        Ok(GuardedAlloc {
            ptr: unsafe { base.add(PAGE_SIZE) },
            total_size,
            usable_size: size,
            base_ptr: base,
        })
    }

    /// Lock memory on Windows.
    #[cfg(all(target_os = "windows", feature = "mlock"))]
    pub fn mlock(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        extern "system" {
            fn VirtualLock(lpAddress: *mut u8, dwSize: usize) -> i32;
        }

        unsafe {
            if VirtualLock(ptr, size) != 0 {
                Ok(())
            } else {
                Err(AllocFailed::mlock_failed(size))
            }
        }
    }

    #[cfg(all(target_os = "windows", feature = "mlock"))]
    pub fn munlock(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        extern "system" {
            fn VirtualUnlock(lpAddress: *mut u8, dwSize: usize) -> i32;
        }

        unsafe {
            if VirtualUnlock(ptr, size) != 0 {
                Ok(())
            } else {
                Err(AllocFailed::mlock_failed(size))
            }
        }
    }

    /// Deallocate memory previously allocated with `alloc`.
    #[cfg(target_os = "windows")]
    #[inline]
    pub fn dealloc(ptr: *mut u8, _size: usize) -> Result<(), AllocFailed> {
        const MEM_RELEASE: u32 = 0x00008000;

        extern "system" {
            fn VirtualFree(lpAddress: *mut u8, dwSize: usize, dwFreeType: u32) -> i32;
        }

        if ptr.is_null() {
            return Ok(());
        }

        // For MEM_RELEASE, dwSize must be 0
        let result = unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };

        if result != 0 {
            Ok(())
        } else {
            Err(AllocFailed::new(0))
        }
    }

    // ========================================================================
    // Unix Fallback (using libc mmap)
    // ========================================================================

    /// Fallback for other Unix-like systems.
    #[cfg(all(
        not(target_os = "linux"),
        not(target_vendor = "apple"),
        not(target_os = "windows"),
        unix
    ))]
    #[inline]
    pub fn alloc(size: usize) -> Result<*mut u8, AllocFailed> {
        use libc::{mmap, MAP_ANON, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE};
        use std::ptr;

        debug_assert!(size > 0);

        let result = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };

        if result == MAP_FAILED {
            Err(AllocFailed::out_of_memory(size))
        } else {
            Ok(result as *mut u8)
        }
    }

    /// Deallocate memory previously allocated with `alloc`.
    #[cfg(all(
        not(target_os = "linux"),
        not(target_vendor = "apple"),
        not(target_os = "windows"),
        unix
    ))]
    #[inline]
    pub fn dealloc(ptr: *mut u8, size: usize) -> Result<(), AllocFailed> {
        use libc::munmap;

        if ptr.is_null() {
            return Ok(());
        }

        let result = unsafe { munmap(ptr as *mut _, size) };

        if result == 0 {
            Ok(())
        } else {
            Err(AllocFailed::new(size))
        }
    }

    // ========================================================================
    // Guard page deallocation helpers
    // ========================================================================

    /// Deallocate a guarded allocation.
    #[cfg(feature = "guard-pages")]
    pub fn dealloc_guarded(guarded: &GuardedAlloc) -> Result<(), AllocFailed> {
        dealloc(guarded.base_ptr, guarded.total_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_dealloc_roundtrip() {
        let size = 4096;
        let ptr = sys::alloc(size).expect("allocation should succeed");

        assert!(!ptr.is_null());

        // Write to verify it's accessible
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, size);
        }

        sys::dealloc(ptr, size).expect("deallocation should succeed");
    }

    #[test]
    fn test_large_allocation() {
        let size = 64 * 1024 * 1024; // 64 MB
        let ptr = sys::alloc(size).expect("large allocation should succeed");

        assert!(!ptr.is_null());

        // Touch first and last pages
        unsafe {
            *ptr = 0x42;
            *ptr.add(size - 1) = 0x42;
        }

        sys::dealloc(ptr, size).expect("deallocation should succeed");
    }

    #[test]
    fn test_alloc_failed_display() {
        let err = AllocFailed::new(1024);
        let msg = format!("{}", err);
        assert!(msg.contains("1024"));
    }

    #[test]
    fn test_alloc_error_kinds() {
        let err = AllocFailed::out_of_memory(1024);
        assert_eq!(err.kind, AllocErrorKind::OutOfMemory);

        let err = AllocFailed::huge_pages_unavailable(1024);
        assert_eq!(err.kind, AllocErrorKind::HugePagesUnavailable);

        let err = AllocFailed::mlock_failed(1024);
        assert_eq!(err.kind, AllocErrorKind::MlockFailed);
    }

    #[test]
    #[cfg(all(target_os = "linux", feature = "guard-pages"))]
    fn test_guard_pages() {
        let size = 4096;
        let guarded = sys::alloc_with_guards(size).expect("guarded allocation should succeed");

        assert!(!guarded.ptr.is_null());
        assert_eq!(guarded.usable_size, size);
        assert!(guarded.total_size > size);

        // Write to usable area
        unsafe {
            std::ptr::write_bytes(guarded.ptr, 0xAB, size);
        }

        sys::dealloc_guarded(&guarded).expect("deallocation should succeed");
    }
}
