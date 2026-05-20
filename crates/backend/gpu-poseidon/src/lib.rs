// GPU (Metal) backend for Poseidon1-16 leaf hashing.
//
// On macOS this dispatches the first-digest layer of the WHIR Merkle tree to the
// integrated Apple GPU via a Metal compute kernel. On other platforms the GPU
// path is a stub and callers should fall back to the CPU implementation.

use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_os = "macos")]
mod metal_impl;

#[cfg(target_os = "macos")]
pub use metal_impl::*;

#[cfg(not(target_os = "macos"))]
pub fn metal_available() -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn build_merkle_tree_full(
    _matrix_u32: &[u32],
    _height: usize,
    _matrix_width: usize,
    _effective_base_width: usize,
    _initial_state: &[u32; 16],
) -> Vec<Vec<[u32; 8]>> {
    unreachable!("Metal Merkle backend is unavailable on this platform")
}

#[cfg(not(target_os = "macos"))]
pub fn build_merkle_tree_full_no_initial(
    _matrix_u32: &[u32],
    _height: usize,
    _matrix_width: usize,
    _full_base_width: usize,
) -> Vec<Vec<[u32; 8]>> {
    unreachable!("Metal Merkle backend is unavailable on this platform")
}

/// Runtime toggle for routing Poseidon work to the GPU. Off by default so
/// existing benchmarks and tests stay on the CPU path; flip it on with
/// [`set_gpu_enabled(true)`] (typically from a CLI flag) before any prover work.
static GPU_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_gpu_enabled(enabled: bool) {
    GPU_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn gpu_enabled() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed)
}
