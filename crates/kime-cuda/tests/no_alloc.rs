//! After a bucket's plan is built on the GPU, running a batch in it allocates nothing on the host,
//! on any thread. This is its own test binary, run without libtest's harness, because the counting
//! allocator sees the whole process. Skipped, with a note, without an NVIDIA GPU.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use common::{Rng, fill, question, tiny};
use kime_cuda::{CudaBackend, Precision};
use kime_tensor::{BatchBuf, Buckets, Executor, HostTensor, Outputs};

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

fn main() {
    for precision in [Precision::F32, Precision::F16] {
        let backend = match CudaBackend::new(0, precision) {
            Ok(b) => b,
            Err(e) => {
                println!("skipped, no GPU: {e}");
                return;
            }
        };
        let (spec, graph, tensors) = tiny();
        let host: Vec<HostTensor<'_>> = (0..tensors.entries().len())
            .map(|i| {
                let v = tensors.view(i);
                HostTensor { dtype: v.dtype, shape: v.shape, bytes: v.bytes }
            })
            .collect();
        let plan = graph.plan(&spec);
        let buckets = Buckets::default();
        let mut exec =
            Executor::new(backend, &host, plan, &buckets, "compat", spec.encoder.vocab, 3).unwrap();
        let mut rng = Rng(9);
        let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
        // Warm up so every bucket the batches use has its plan and the buffers reach their size.
        let batches: Vec<Vec<_>> = (0..20)
            .map(|_| (0..1 + rng.below(6)).map(|_| question(&mut rng, 40)).collect())
            .collect();
        for qs in &batches {
            fill(&mut buf, qs);
            exec.run(&buf.batch(), &mut out).unwrap();
        }
        let warm: Vec<_> = exec.warm().collect();
        COUNT.store(0, Ordering::SeqCst);
        ON.store(true, Ordering::SeqCst);
        for qs in &batches {
            fill(&mut buf, qs);
            exec.run(&buf.batch(), &mut out).unwrap();
        }
        ON.store(false, Ordering::SeqCst);
        assert_eq!(exec.warm().collect::<Vec<_>>(), warm);
        assert_eq!(COUNT.load(Ordering::SeqCst), 0, "allocations on the warm path in {precision:?}");
        println!("a warm {precision:?} plan does not allocate: ok, {} buckets", warm.len());
    }
}
