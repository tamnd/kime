//! A pool of worker threads that lives as long as the backend, from spec/10-cpu.md.
//!
//! [`Pool::run`] hands out tasks `0..n` through a shared counter to the calling thread and every
//! worker, and returns when all of them are done. Between jobs a worker spins for a while, since
//! the next op of a plan is usually microseconds away, and then sleeps on a condition variable so
//! an idle server costs nothing. A job is a borrowed closure, so running one allocates nothing.
//!
//! Which thread runs a task depends on timing, but what a task computes does not, and that is what
//! keeps results independent of the thread count.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Spins before a worker sleeps. About 50 microseconds on a current core.
const SPIN: u32 = 1 << 14;

type Job = dyn Fn(usize, usize) + Sync;

struct Inner {
    /// Bumped once per job. A worker runs a job when it sees a value it has not seen.
    generation: AtomicU64,
    /// The current job. Written by the caller before it bumps `generation`, and read by workers
    /// after they see the bump.
    job: std::cell::UnsafeCell<*const Job>,
    tasks: AtomicUsize,
    next: AtomicUsize,
    /// Workers still inside the current job.
    busy: AtomicUsize,
    sleepers: AtomicUsize,
    panicked: AtomicBool,
    stop: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

// SAFETY: `job` is written only by the thread inside `Pool::run`, which holds the pool's run lock,
// while every worker is outside a job (busy is 0), and read by workers only between seeing a new
// generation and decrementing busy. The Release store of generation orders the write before the
// reads. Everything else is atomics and a mutex.
unsafe impl Sync for Inner {}
// SAFETY: as above, the raw pointer is only dereferenced under that protocol.
unsafe impl Send for Inner {}

impl Inner {
    fn work(&self, worker: usize) {
        // SAFETY: see the Sync impl. The job outlives this call because the caller waits for busy
        // to reach 0, and for its own part returns only after this returns.
        let job = unsafe { &**self.job.get() };
        let n = self.tasks.load(Ordering::Relaxed);
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            job(i, worker);
        }
    }
}

/// Worker threads, created once.
pub struct Pool {
    inner: Arc<Inner>,
    handles: Vec<JoinHandle<()>>,
    run: Mutex<()>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool").field("threads", &self.threads()).finish()
    }
}

impl Pool {
    /// A pool of `threads` threads counting the caller, so `threads - 1` are spawned.
    ///
    /// # Panics
    ///
    /// If the OS will not start a thread.
    #[must_use]
    pub fn new(threads: usize) -> Self {
        let noop: &'static Job = &|_, _| {};
        let inner = Arc::new(Inner {
            generation: AtomicU64::new(0),
            job: std::cell::UnsafeCell::new(noop as *const Job),
            tasks: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            busy: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let handles = (1..threads.max(1))
            .map(|worker| {
                let inner = Arc::clone(&inner);
                std::thread::Builder::new()
                    .name(format!("kime-cpu-{worker}"))
                    .spawn(move || worker_loop(&inner, worker))
                    .expect("spawn a worker thread")
            })
            .collect();
        Self { inner, handles, run: Mutex::new(()) }
    }

    /// Threads that run tasks, the caller included.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.handles.len() + 1
    }

    /// Runs `f(task, worker)` for every task in `0..n`, where `worker` is below
    /// [`Pool::threads`] and no two tasks run on the same worker at once. The caller is worker 0.
    ///
    /// # Panics
    ///
    /// If `f` panics on any thread.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        if n == 0 {
            return;
        }
        if n == 1 || self.handles.is_empty() {
            (0..n).for_each(|i| f(i, 0));
            return;
        }
        let guard = self.run.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let inner = &*self.inner;
        // SAFETY: no worker is inside a job (the last run waited for busy to reach 0) and the run
        // lock keeps other callers out, so nothing reads `job` now. The lifetime is erased, and
        // the wait below keeps `f` borrowed until every worker is done with it.
        unsafe {
            *inner.job.get() =
                std::mem::transmute::<&(dyn Fn(usize, usize) + Sync + '_), &'static Job>(f)
                    as *const Job;
        }
        inner.tasks.store(n, Ordering::Relaxed);
        inner.next.store(0, Ordering::Relaxed);
        inner.busy.store(self.handles.len(), Ordering::Relaxed);
        inner.generation.fetch_add(1, Ordering::SeqCst);
        if inner.sleepers.load(Ordering::SeqCst) > 0 {
            let _l = inner.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.wake.notify_all();
        }
        let mine = catch_unwind(AssertUnwindSafe(|| inner.work(0)));
        while inner.busy.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
        drop(guard);
        if let Err(e) = mine {
            std::panic::resume_unwind(e);
        }
        assert!(!inner.panicked.swap(false, Ordering::Relaxed), "a kime-cpu worker panicked");
    }
}

fn worker_loop(inner: &Inner, worker: usize) {
    let mut seen = 0;
    loop {
        let mut spins = 0u32;
        loop {
            let g = inner.generation.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            if inner.stop.load(Ordering::Relaxed) {
                return;
            }
            if spins < SPIN {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let l = inner.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.sleepers.fetch_add(1, Ordering::SeqCst);
            let l = if inner.generation.load(Ordering::SeqCst) == seen
                && !inner.stop.load(Ordering::SeqCst)
            {
                inner.wake.wait(l).unwrap_or_else(std::sync::PoisonError::into_inner)
            } else {
                l
            };
            inner.sleepers.fetch_sub(1, Ordering::SeqCst);
            drop(l);
            spins = 0;
        }
        if catch_unwind(AssertUnwindSafe(|| inner.work(worker))).is_err() {
            inner.panicked.store(true, Ordering::Relaxed);
        }
        inner.busy.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let _l = self.inner.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            self.inner.stop.store(true, Ordering::SeqCst);
            self.inner.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn every_task_once_on_a_valid_worker() {
        for threads in [1, 2, 5] {
            let pool = Pool::new(threads);
            for n in [0, 1, 3, 1000] {
                let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
                pool.run(n, &|i, w| {
                    assert!(w < threads);
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
            }
        }
    }

    #[test]
    fn wakes_after_sleeping() {
        let pool = Pool::new(3);
        let count = AtomicU32::new(0);
        for _ in 0..3 {
            std::thread::sleep(std::time::Duration::from_millis(30));
            pool.run(64, &|_, _| {
                count.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(count.load(Ordering::Relaxed), 192);
    }

    #[test]
    fn a_panic_reaches_the_caller_and_the_pool_survives() {
        let pool = Pool::new(4);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(100, &|i, _| assert!(i != 57, "task 57"));
        }));
        assert!(r.is_err());
        let count = AtomicU32::new(0);
        pool.run(10, &|_, _| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 10);
    }
}
