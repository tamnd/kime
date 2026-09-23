//! Splitting work across threads.
//!
//! This is the plain version the reference kernels use: scoped threads per call and a shared
//! counter that hands out tasks in order. The pinned worker groups from spec/10-cpu.md replace it
//! when the plan executor lands. Nothing a task computes may depend on which thread runs it, which
//! is what keeps the kernels deterministic under any thread count.

use std::marker::PhantomData;
use std::num::NonZero;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The number of threads the machine offers.
#[must_use]
pub fn available() -> usize {
    std::thread::available_parallelism().map_or(1, NonZero::get)
}

/// Runs `f(i)` for every `i` in `0..n` on up to `threads` threads.
pub fn for_each(n: usize, threads: usize, f: impl Fn(usize) + Sync) {
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        (0..n).for_each(f);
        return;
    }
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            f(i);
        }
    };
    std::thread::scope(|s| {
        for _ in 1..threads {
            s.spawn(work);
        }
        work();
    });
}

/// `(0..n).map(f).collect()` on up to `threads` threads, in order.
///
/// # Panics
///
/// If `f` panics.
pub fn map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    for_each(n, threads, |i| *slots[i].lock().unwrap() = Some(f(i)));
    slots.into_iter().map(|s| s.into_inner().unwrap().unwrap()).collect()
}

/// A buffer many threads write into at once, each to elements no other thread touches.
#[derive(Debug)]
pub struct Shared<'a> {
    ptr: *mut f32,
    len: usize,
    _borrow: PhantomData<&'a mut [f32]>,
}

// SAFETY: Shared is a &mut [f32] split across threads. The only access is `set`, whose contract
// makes every element written by at most one thread, so sending and sharing it is sound.
unsafe impl Send for Shared<'_> {}
// SAFETY: as above.
unsafe impl Sync for Shared<'_> {}

impl<'a> Shared<'a> {
    /// Borrows `buf` for the lifetime of the Shared.
    pub fn new(buf: &'a mut [f32]) -> Self {
        Self { ptr: buf.as_mut_ptr(), len: buf.len(), _borrow: PhantomData }
    }

    /// Writes `v` at `i`.
    ///
    /// # Safety
    ///
    /// No other thread may read or write element `i` while this Shared is alive.
    ///
    /// # Panics
    ///
    /// If `i` is out of bounds.
    #[inline(always)]
    pub unsafe fn set(&self, i: usize, v: f32) {
        assert!(i < self.len);
        // SAFETY: in bounds by the assert, and the caller guarantees no other thread touches i.
        unsafe { self.ptr.add(i).write(v) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once() {
        for threads in [1, 2, 7] {
            for n in [0, 1, 5, 100] {
                let got = map(n, threads, |i| i * 2);
                assert_eq!(got, (0..n).map(|i| i * 2).collect::<Vec<_>>());
            }
        }
    }
}
