include!(concat!(env!("OUT_DIR"), "/info.rs"));

#[cfg(feature = "profiling")]
use std::{
    collections::BTreeMap,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

const _: () = assert!(usize::BITS == 64, "this project requires a 64-bit target (for now)");

#[cfg(feature = "profiling")]
static CPU_STAGE: OnceLock<Mutex<String>> = OnceLock::new();
#[cfg(feature = "profiling")]
static CPU_STAGE_TIMING_ENABLED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "profiling")]
static CPU_STAGE_TIMINGS: OnceLock<Mutex<BTreeMap<String, Duration>>> = OnceLock::new();

#[cfg(feature = "profiling")]
pub fn enter_cpu_stage(stage: impl Into<String>) -> CpuStageGuard {
    let stage = stage.into();
    let mut current = cpu_stage().lock().unwrap();
    let previous = std::mem::replace(&mut *current, stage.clone());
    let start = CPU_STAGE_TIMING_ENABLED.load(Ordering::Relaxed).then(Instant::now);
    CpuStageGuard { previous, stage, start }
}

#[cfg(not(feature = "profiling"))]
#[inline(always)]
pub fn enter_cpu_stage(_stage: impl Into<String>) -> CpuStageGuard {
    CpuStageGuard
}

pub const fn cpu_stage_profiling_enabled() -> bool {
    cfg!(feature = "profiling")
}

#[cfg(feature = "profiling")]
pub fn current_cpu_stage() -> String {
    cpu_stage().lock().unwrap().clone()
}

#[cfg(not(feature = "profiling"))]
pub fn current_cpu_stage() -> String {
    "profiling-disabled".to_string()
}

#[cfg(feature = "profiling")]
pub fn reset_cpu_stage_timings() {
    cpu_stage_timings().lock().unwrap().clear();
    CPU_STAGE_TIMING_ENABLED.store(true, Ordering::Relaxed);
}

#[cfg(not(feature = "profiling"))]
pub fn reset_cpu_stage_timings() {}

#[cfg(feature = "profiling")]
pub fn take_cpu_stage_timings() -> Vec<(String, Duration)> {
    CPU_STAGE_TIMING_ENABLED.store(false, Ordering::Relaxed);
    let mut timings = cpu_stage_timings().lock().unwrap();
    std::mem::take(&mut *timings).into_iter().collect()
}

#[cfg(not(feature = "profiling"))]
pub fn take_cpu_stage_timings() -> Vec<(String, std::time::Duration)> {
    Vec::new()
}

#[cfg(feature = "profiling")]
fn cpu_stage() -> &'static Mutex<String> {
    CPU_STAGE.get_or_init(|| Mutex::new("idle".to_string()))
}

#[cfg(feature = "profiling")]
fn cpu_stage_timings() -> &'static Mutex<BTreeMap<String, Duration>> {
    CPU_STAGE_TIMINGS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[cfg(feature = "profiling")]
#[derive(Debug)]
pub struct CpuStageGuard {
    previous: String,
    stage: String,
    start: Option<Instant>,
}

#[cfg(not(feature = "profiling"))]
#[derive(Debug)]
pub struct CpuStageGuard;

#[cfg(feature = "profiling")]
impl Drop for CpuStageGuard {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            *cpu_stage_timings()
                .lock()
                .unwrap()
                .entry(std::mem::take(&mut self.stage))
                .or_default() += start.elapsed();
        }
        *cpu_stage().lock().unwrap() = std::mem::take(&mut self.previous);
    }
}

pub fn peak_rss_bytes() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut ru) };
    let max = ru.ru_maxrss as u64;
    // ru_maxrss unit: bytes on macOS, KiB on Linux.
    if cfg!(target_os = "macos") { max } else { max * 1024 }
}

pub fn process_cpu_time() -> std::time::Duration {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut ru) };
    timeval_duration(ru.ru_utime) + timeval_duration(ru.ru_stime)
}

fn timeval_duration(timeval: libc::timeval) -> std::time::Duration {
    let secs = timeval.tv_sec.try_into().unwrap_or(0);
    let micros = timeval.tv_usec.try_into().unwrap_or(0);
    std::time::Duration::from_secs(secs) + std::time::Duration::from_micros(micros)
}

/// Number of jobs [`flush_rayon`] pushes. Must exceed
/// `crossbeam_deque::deque::BLOCK_CAP` (currently 63 —
/// `crossbeam-deque-0.8.6/src/deque.rs:1191`).
const RAYON_FLUSH_JOBS: usize = 256;

/// Drain rayon's internal queues so they release any storage allocated during the
/// previous phase.
///
/// Rayon's global pool owns a `crossbeam_deque::Injector`, internally a linked list
/// of fixed-size blocks (`Block` and `Injector::push` —
/// `crossbeam-deque-0.8.6/src/deque.rs:1219` and `:1371`). A block is freed only
/// once its last slot has been consumed.
///
/// `rayon::join` from a non-worker thread reaches that injector via
/// `join` (`rayon-core-1.13.0/src/join/mod.rs:132`) ->
/// `registry::in_worker` (`registry.rs:946`) ->
/// `Registry::in_worker_cold` (`:517`) ->
/// `Registry::inject` (`:428`) -> `Injector::push`.
///
/// Under an arena allocator that recycles memory between phases (e.g. `zk-alloc`),
/// a block allocated *during* a phase points into a slab the next `begin_phase()`
/// will reuse. The next push then writes a `JobRef` straight through whatever the
/// application has placed on top, silently corrupting it.
///
/// Pushing more than `BLOCK_CAP` jobs while the arena is off forces the Injector                                        
/// to allocate a fresh tail block (which lands in System), and forces workers to                                      
/// steal the last slot of every preceding block (which destroys them).
pub fn flush_rayon() {
    for _ in 0..RAYON_FLUSH_JOBS {
        rayon::join(|| {}, || {});
    }
}
