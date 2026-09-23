//! The Laya compat graph on one GPU.

use kime_model::Model;
use kime_tensor::{Buckets, Executor, HostTensor, Result};

use crate::{CudaBackend, Precision};

/// The compat graph of `model` on GPU `ordinal` at `precision`, with the default bucket table.
///
/// # Errors
///
/// [`kime_tensor::Error::Device`] when the GPU cannot be opened, or an error from lowering, which
/// for a checkpoint that loaded means a bug.
pub fn executor(
    model: &Model,
    ordinal: usize,
    precision: Precision,
) -> Result<Executor<CudaBackend>> {
    let t = &model.tensors;
    let host: Vec<HostTensor<'_>> = (0..t.entries().len())
        .map(|i| {
            let v = t.view(i);
            HostTensor { dtype: v.dtype, shape: v.shape, bytes: v.bytes }
        })
        .collect();
    let plan = model.graph.plan(&model.spec);
    let vocab = model.spec.encoder.vocab;
    Executor::new(
        CudaBackend::new(ordinal, precision)?,
        &host,
        plan,
        &Buckets::default(),
        "compat",
        vocab,
        3,
    )
}
