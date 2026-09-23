//! After a bucket's plan is built, running a batch in it allocates nothing, on any thread. This is
//! its own test binary because the counting allocator sees the whole process.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use common::{Rng, fill, question, tiny};
use kime_cpu::executor_from;
use kime_tensor::{BatchBuf, Outputs};

struct Counting;

static ON: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator and only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ON.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: as the caller promised us.
        unsafe { System.alloc(l) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: as the caller promised us.
        unsafe { System.dealloc(p, l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ON.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: as the caller promised us.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn a_warm_plan_does_not_allocate() {
    let (spec, graph, tensors) = tiny();
    let mut exec = executor_from(&spec, &graph, &tensors, 4).unwrap();
    let mut rng = Rng(9);
    let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
    // Warm up on batches of up to 256 tokens so the buffers reach their size.
    let batches: Vec<Vec<_>> =
        (0..20).map(|_| (0..1 + rng.below(6)).map(|_| question(&mut rng, 40)).collect()).collect();
    for qs in &batches {
        fill(&mut buf, qs);
        exec.run(&buf.batch(), &mut out).unwrap();
    }
    let warm: Vec<_> = exec.warm().collect();
    ON.store(true, Ordering::SeqCst);
    for qs in &batches {
        fill(&mut buf, qs);
        exec.run(&buf.batch(), &mut out).unwrap();
    }
    ON.store(false, Ordering::SeqCst);
    assert_eq!(exec.warm().collect::<Vec<_>>(), warm);
    assert_eq!(COUNT.load(Ordering::SeqCst), 0, "allocations on the warm path");
}
