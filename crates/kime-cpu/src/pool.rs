//! A pool of worker threads that lives as long as the backend, from spec/10-cpu.md.
//!
//! [`Pool::run`] hands out tasks `0..n` to the calling thread and every worker, and returns when
//! all of them are done. It waits for the tasks, not for the workers: a step of four tasks on a
//! pool of ten returns once the four are done, even when the other workers have not woken up yet.
//! That matters most on a busy machine, where a worker the OS has not scheduled would otherwise
//! hold up every small step. Between jobs a worker spins for a while, since the next op of a plan
//! is usually microseconds away, and then sleeps on a condition variable so an idle server costs
//! nothing. A job is a borrowed closure, so running one allocates nothing.
//!
//! Tasks are claimed from one word that holds the job's generation, its task count and the next
//! task, so a worker that wakes up late cannot claim a task of a job that has already finished.
//!
//! Which thread runs a task depends on timing, but what a task computes does not, and that is what
//! keeps results independent of the thread count.

use std::any::Any;
use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Spins before a worker sleeps. About 50 microseconds on a current core.
const SPIN: u32 = 1 << 14;

/// Bits of the claim word for the next task and for the task count. The generation takes the
/// other 24.
const BITS: u32 = 20;
/// Tasks in one round of a job. Larger jobs run in rounds.
const ROUND: usize = 1 << BITS;
const MASK: u64 = (1 << BITS) - 1;

type Job = dyn Fn(usize, usize) + Sync;

/// The generation, the task count and the next task in a claim word.
fn unpack(s: u64) -> (u64, usize, usize) {
    (s >> (2 * BITS), ((s >> BITS) & MASK) as usize, (s & MASK) as usize)
}

struct Inner {
    /// `generation << 40 | tasks << 20 | next`. A task is claimed by moving `next` up by one with a
    /// compare and swap, which fails if the job changed in between.
    state: AtomicU64,
    /// The job of each generation, in the slot of its parity. Written by the caller before it
    /// publishes the generation in `state`.
    jobs: [UnsafeCell<*const Job>; 2],
    /// Tasks of the current job that have finished.
    done: AtomicUsize,
    sleepers: AtomicUsize,
    /// The first panic of the current job.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    stop: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

// SAFETY: a job slot is written only by the thread inside `Pool::run`, which holds the pool's run
// lock, and only for a generation whose slot was last used two jobs ago, which finished before the
// last one started. A worker reads a slot only after it claimed a task of that slot's generation,
// and the caller of that generation waits for the task to be done before it returns, so the slot
// is not written while it is read. The Release store of `state` orders the write before the reads.
// Everything else is atomics and mutexes.
unsafe impl Sync for Inner {}
// SAFETY: as above, the raw pointers are only dereferenced under that protocol.
unsafe impl Send for Inner {}

impl Inner {
    /// Claims and runs tasks of the current job until none is left.
    fn work(&self, worker: usize) {
        let mut s = self.state.load(Ordering::Acquire);
        loop {
            let (generation, tasks, next) = unpack(s);
            if next >= tasks {
                return;
            }
            if let Err(now) =
                self.state.compare_exchange_weak(s, s + 1, Ordering::Acquire, Ordering::Acquire)
            {
                s = now;
                continue;
            }
            // SAFETY: see the Sync impl. This task belongs to `generation`, whose caller is still
            // waiting for it.
            let job = unsafe { &**self.jobs[(generation & 1) as usize].get() };
            if let Err(e) = catch_unwind(AssertUnwindSafe(|| job(next, worker))) {
                self.panic
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_or_insert(e);
            }
            self.done.fetch_add(1, Ordering::Release);
            s = self.state.load(Ordering::Acquire);
        }
    }

    /// Whether `state` has a task left to claim.
    fn pending(&self, order: Ordering) -> bool {
        let (_, tasks, next) = unpack(self.state.load(order));
        next < tasks
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
            state: AtomicU64::new(0),
            jobs: [UnsafeCell::new(noop as *const Job), UnsafeCell::new(noop as *const Job)],
            done: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            panic: Mutex::new(None),
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
        for base in (0..n).step_by(ROUND) {
            let tasks = ROUND.min(n - base);
            self.round(tasks, &|i, worker| f(base + i, worker));
        }
        drop(guard);
    }

    /// One job of at most [`ROUND`] tasks. The caller holds the run lock.
    fn round(&self, tasks: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        let inner = &*self.inner;
        let (last, _, _) = unpack(inner.state.load(Ordering::Relaxed));
        // A generation wraps after 16 million jobs, and a claim only goes wrong if a worker stalls
        // between reading the word and swapping it for exactly that many jobs.
        let generation = (last + 1) & ((1 << (64 - 2 * BITS)) - 1);
        // SAFETY: see the Sync impl. The lifetime is erased, and the wait below keeps `f`
        // borrowed until every task is done.
        unsafe {
            *inner.jobs[(generation & 1) as usize].get() =
                std::mem::transmute::<&(dyn Fn(usize, usize) + Sync + '_), &'static Job>(f)
                    as *const Job;
        }
        inner.done.store(0, Ordering::Relaxed);
        inner.state.store(generation << (2 * BITS) | (tasks as u64) << BITS, Ordering::SeqCst);
        if inner.sleepers.load(Ordering::SeqCst) > 0 {
            let _l = inner.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.wake.notify_all();
        }
        inner.work(0);
        while inner.done.load(Ordering::Acquire) < tasks {
            std::hint::spin_loop();
        }
        let panic = inner.panic.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(e) = panic {
            std::panic::resume_unwind(e);
        }
    }
}

fn worker_loop(inner: &Inner, worker: usize) {
    loop {
        let mut spins = 0u32;
        while !inner.pending(Ordering::Acquire) {
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
            let l = if !inner.pending(Ordering::SeqCst) && !inner.stop.load(Ordering::SeqCst) {
                inner.wake.wait(l).unwrap_or_else(std::sync::PoisonError::into_inner)
            } else {
                l
            };
            inner.sleepers.fetch_sub(1, Ordering::SeqCst);
            drop(l);
            spins = 0;
        }
        inner.work(worker);
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
    fn no_worker_runs_two_tasks_at_once_over_many_small_jobs() {
        let pool = Pool::new(6);
        let busy: Vec<AtomicBool> = (0..6).map(|_| AtomicBool::new(false)).collect();
        for job in 0..20_000usize {
            let n = 1 + job % 9;
            let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            pool.run(n, &|i, w| {
                assert!(!busy[w].swap(true, Ordering::AcqRel), "worker {w} ran two tasks at once");
                hits[i].fetch_add(1, Ordering::Relaxed);
                busy[w].store(false, Ordering::Release);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "job {job}");
        }
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
