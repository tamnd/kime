//! The `Backend` trait and the executor that picks a bucket and replays its plan.

use std::fmt;

use crate::bucket::{Bucket, Buckets};
use crate::dtype::DType;
use crate::plan::Graph;

/// A weight as it sits in the checkpoint, before a backend converts it.
#[derive(Debug, Clone, Copy)]
pub struct HostTensor<'a> {
    /// Element type.
    pub dtype: DType,
    /// Shape.
    pub shape: &'a [usize],
    /// Little endian, row major.
    pub bytes: &'a [u8],
}

/// A batch, flattened. Sequence `s` is tokens `cu[s]..cu[s + 1]` and markers
/// `mcu[s]..mcu[s + 1]`, and each marker is a position within its own sequence.
#[derive(Debug, Clone, Copy, Default)]
pub struct Batch<'a> {
    /// Token ids of every sequence, end to end.
    pub ids: &'a [u32],
    /// Sequence starts in `ids`, with the total at the end.
    pub cu: &'a [u32],
    /// Marker positions of every sequence, end to end.
    pub markers: &'a [u32],
    /// Sequence starts in `markers`, with the total at the end.
    pub mcu: &'a [u32],
    /// The question type of each sequence.
    pub qtype: &'a [u8],
}

impl Batch<'_> {
    /// Sequences.
    #[must_use]
    pub fn seqs(&self) -> usize {
        self.cu.len().saturating_sub(1)
    }

    /// Checks that the pieces agree with each other, so a backend can index without checks.
    ///
    /// # Errors
    ///
    /// [`Error::Batch`] naming the first problem.
    pub fn check(&self, vocab: usize, types: usize) -> Result<()> {
        let bad = |m: String| Err(Error::Batch(m));
        let s = self.seqs();
        if self.cu.first() != Some(&0) || self.mcu.first() != Some(&0) {
            return bad("cu and mcu must start at 0".into());
        }
        if self.mcu.len() != s + 1 || self.qtype.len() != s {
            return bad(format!(
                "{s} sequences but {} mcu and {} qtype",
                self.mcu.len(),
                self.qtype.len()
            ));
        }
        if self.cu[s] as usize != self.ids.len() || self.mcu[s] as usize != self.markers.len() {
            return bad("cu and mcu must end at the totals".into());
        }
        for i in 0..s {
            let (a, b) = (self.cu[i], self.cu[i + 1]);
            if b < a || self.mcu[i + 1] < self.mcu[i] {
                return bad(format!("sequence {i} ends before it starts"));
            }
            if usize::from(self.qtype[i]) >= types {
                return bad(format!("sequence {i} has type {} of {types}", self.qtype[i]));
            }
            for &m in &self.markers[self.mcu[i] as usize..self.mcu[i + 1] as usize] {
                if m >= b - a {
                    return bad(format!("marker {m} is past sequence {i} of {} tokens", b - a));
                }
            }
        }
        if let Some(&id) = self.ids.iter().find(|&&id| id as usize >= vocab) {
            return bad(format!("token id {id} is past the vocabulary of {vocab}"));
        }
        Ok(())
    }
}

/// What a batch produces. Reused across batches, so after the first few it does not allocate.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outputs {
    /// One logit per marker, in batch order.
    pub logits: Vec<f32>,
    /// The act head, one pair per sequence.
    pub act: Vec<[f32; 2]>,
}

/// What a backend can do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caps {
    /// `cpu`, `cuda`, `metal` or `ane`.
    pub name: &'static str,
    /// Worker threads, for the CPU.
    pub threads: usize,
    /// Whether plans are captured as device graphs, which makes padding to the bucket matter.
    pub graphs: bool,
    /// Whether host and device share memory.
    pub unified_memory: bool,
}

/// Engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The batch is inconsistent.
    Batch(String),
    /// No bucket of the stage holds the batch.
    NoBucket {
        /// The stage.
        stage: String,
        /// Tokens in the batch.
        tokens: usize,
        /// Sequences.
        seqs: usize,
        /// Markers.
        markers: usize,
    },
    /// The graph uses something the backend does not have, or weights of the wrong shape.
    Unsupported(String),
    /// The device or its driver failed.
    Device(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Batch(m) => write!(f, "bad batch: {m}"),
            Error::NoBucket { stage, tokens, seqs, markers } => write!(
                f,
                "no {stage} bucket holds {tokens} tokens, {seqs} sequences and {markers} markers"
            ),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Device(m) => write!(f, "device: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Engine results.
pub type Result<T> = std::result::Result<T, Error>;

/// A device that runs graphs. The engine is generic over it, so the path through it is
/// monomorphized and there is no dynamic dispatch per op.
pub trait Backend: Send + Sync {
    /// Weights in the backend's own layout.
    type Weights: Send + Sync;
    /// A graph lowered for one bucket, with its arena.
    type Plan: Send;

    /// What the backend can do.
    fn caps(&self) -> Caps;

    /// Converts a checkpoint's tensors, indexed as the graph's weights are.
    ///
    /// # Errors
    ///
    /// When a tensor cannot be converted.
    fn upload(&self, tensors: &[HostTensor<'_>], graph: &Graph) -> Result<Self::Weights>;

    /// Lowers a graph for one bucket. This is where memory is allocated.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for an op or a shape the backend cannot run.
    fn lower(&self, w: &Self::Weights, graph: &Graph, bucket: Bucket) -> Result<Self::Plan>;

    /// Runs a batch that fits the plan's bucket and has been checked.
    ///
    /// # Errors
    ///
    /// When the device fails.
    fn run(&self, plan: &mut Self::Plan, batch: &Batch<'_>, out: &mut Outputs) -> Result<()>;

    /// Bytes the weights take on the device.
    fn weight_bytes(&self, _w: &Self::Weights) -> usize {
        0
    }

    /// Bytes a plan takes on the device, its arena and staging buffers.
    fn plan_bytes(&self, _p: &Self::Plan) -> usize {
        0
    }
}

/// A graph, its weights on one backend, and a plan per bucket built on first use.
#[derive(Debug)]
pub struct Executor<B: Backend> {
    backend: B,
    weights: B::Weights,
    graph: Graph,
    stage: String,
    buckets: Vec<Bucket>,
    plans: Vec<(Bucket, B::Plan)>,
    vocab: usize,
    types: usize,
}

impl<B: Backend> Executor<B> {
    /// Uploads the weights. `vocab` and `types` bound the token ids and question types a batch may
    /// hold, and `stage` names the buckets to use from `buckets`.
    ///
    /// # Errors
    ///
    /// From [`Backend::upload`].
    pub fn new(
        backend: B,
        tensors: &[HostTensor<'_>],
        graph: Graph,
        buckets: &Buckets,
        stage: &str,
        vocab: usize,
        types: usize,
    ) -> Result<Self> {
        let weights = backend.upload(tensors, &graph)?;
        Ok(Self {
            backend,
            weights,
            graph,
            stage: stage.to_string(),
            buckets: buckets.stage(stage).to_vec(),
            plans: Vec::new(),
            vocab,
            types,
        })
    }

    /// The backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// The buckets with a plan built.
    pub fn warm(&self) -> impl Iterator<Item = Bucket> + '_ {
        self.plans.iter().map(|p| p.0)
    }

    /// Bytes on the device: the weights, and the plans built so far.
    pub fn memory(&self) -> (usize, usize) {
        let plans = self.plans.iter().map(|p| self.backend.plan_bytes(&p.1)).sum();
        (self.backend.weight_bytes(&self.weights), plans)
    }

    /// The plans built so far, to inspect or profile.
    pub fn plans_mut(&mut self) -> impl Iterator<Item = &mut B::Plan> + '_ {
        self.plans.iter_mut().map(|p| &mut p.1)
    }

    /// Builds the plan for `bucket` now rather than on first use.
    ///
    /// # Errors
    ///
    /// From [`Backend::lower`].
    pub fn prepare(&mut self, bucket: Bucket) -> Result<usize> {
        if let Some(i) = self.plans.iter().position(|p| p.0 == bucket) {
            return Ok(i);
        }
        let plan = self.backend.lower(&self.weights, &self.graph, bucket)?;
        self.plans.push((bucket, plan));
        Ok(self.plans.len() - 1)
    }

    /// Runs a batch in the smallest bucket that holds it. Once that bucket's plan exists and `out`
    /// has grown to the batch size, this allocates nothing.
    ///
    /// # Errors
    ///
    /// [`Error::Batch`] for an inconsistent batch, [`Error::NoBucket`] for one too big, and
    /// anything the backend reports.
    pub fn run(&mut self, batch: &Batch<'_>, out: &mut Outputs) -> Result<Bucket> {
        batch.check(self.vocab, self.types)?;
        let (t, s, m) = (batch.ids.len(), batch.seqs(), batch.markers.len());
        let bucket = self.buckets.iter().copied().find(|b| b.holds(t, s, m)).ok_or_else(|| {
            Error::NoBucket { stage: self.stage.clone(), tokens: t, seqs: s, markers: m }
        })?;
        let i = self.prepare(bucket)?;
        self.backend.run(&mut self.plans[i].1, batch, out)?;
        Ok(bucket)
    }
}

/// Owned storage for a [`Batch`], reused from one batch to the next so that building one does
/// not allocate once it has grown.
#[derive(Debug, Clone, Default)]
pub struct BatchBuf {
    ids: Vec<u32>,
    cu: Vec<u32>,
    markers: Vec<u32>,
    mcu: Vec<u32>,
    qtype: Vec<u8>,
}

impl BatchBuf {
    /// Empties it, keeping the memory.
    pub fn clear(&mut self) {
        for v in [&mut self.ids, &mut self.cu, &mut self.markers, &mut self.mcu] {
            v.clear();
        }
        self.qtype.clear();
    }

    /// Appends a sequence.
    ///
    /// # Panics
    ///
    /// If the batch passes 2^32 tokens or markers.
    pub fn push(&mut self, ids: &[u32], markers: &[u32], qtype: u8) {
        if self.cu.is_empty() {
            self.cu.push(0);
            self.mcu.push(0);
        }
        self.ids.extend_from_slice(ids);
        self.markers.extend_from_slice(markers);
        self.cu.push(u32::try_from(self.ids.len()).expect("under 2^32 tokens"));
        self.mcu.push(u32::try_from(self.markers.len()).expect("under 2^32 markers"));
        self.qtype.push(qtype);
    }

    /// The batch.
    #[must_use]
    pub fn batch(&self) -> Batch<'_> {
        const EMPTY: &[u32] = &[0];
        let (cu, mcu) =
            if self.cu.is_empty() { (EMPTY, EMPTY) } else { (&self.cu[..], &self.mcu[..]) };
        Batch { ids: &self.ids, cu, markers: &self.markers, mcu, qtype: &self.qtype }
    }
}
