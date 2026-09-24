//! The Laya compat graph on the GPU.

use kime_model::Model;
use kime_tensor::{Buckets, Executor, HostTensor, Result};

use crate::{MetalBackend, Precision};

/// The compat graph of `model` on the system's default GPU at `precision`, with the default bucket
/// table.
///
/// # Errors
///
/// [`kime_tensor::Error::Device`] when there is no GPU or the kernels do not compile, or an error
/// from lowering, which for a checkpoint that loaded means a bug.
pub fn executor(model: &Model, precision: Precision) -> Result<Executor<MetalBackend>> {
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
        MetalBackend::new(precision)?,
        &host,
        plan,
        &Buckets::default(),
        "compat",
        vocab,
        3,
    )
}
