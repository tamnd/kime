# CPU backend

`kime-cpu` makes kime useful on machines with no GPU: CI runners, small VMs, laptops without Apple Silicon, ARM servers. Laya's CPU path is the baseline to beat: 9,396 ms per question with default PyTorch threads on a Ryzen 9 6900HX, and 329 ms at best after hand tuning the thread counts. The target is 33 ms or less for W1 on the same chip, which should be easy with the split architecture, and 10x over tuned Laya on every CPU workload in 13.

## Targets

| ISA level | Examples | GEMM path |
|---|---|---|
| x86-64-v3 (AVX2, FMA, F16C) | Zen 2 and 3, Haswell to Alder Lake | FP32 compute on FP16 weights converted in the micro-kernel. INT8 through `vpmaddubsw`. |
| x86-64-v4 (AVX-512) plus VNNI and BF16 | Zen 4 and 5, Ice Lake and newer Xeon | INT8 through `vpdpbusd`, BF16 through `vdpbf16ps` |
| AMX (Sapphire Rapids and newer Xeon) | c7i, m7i, Granite Rapids | INT8 and BF16 tiles through `asm!` (`tdpbssd`, `tdpbf16ps`), after `arch_prctl(ARCH_REQ_XCOMP_PERM, 18)` |
| AArch64 NEON with dotprod and i8mm | Graviton 3 and 4, Ampere, Apple M series | INT8 through `smmla` and `sdot`, FP16 through `fmla` |
| AArch64 SME | Apple M4, newer ARM | SME FP32 outer products. On Apple, Accelerate `cblas_sgemm` is used instead when it is faster at the given shape. |

Dispatch is at run time with `std::arch::is_x86_feature_detected!` and `is_aarch64_feature_detected!`. The kernel choice per plan is fixed at plan build time, so there is no per call branching.

## Precision

- The default is INT8 weights, per output channel symmetric, with dynamic INT8 activations per token, and INT32 accumulation dequantized to FP32. Laya's CPU path is FP32, and a published ORT dynamic INT8 ModernBERT classifier lost 3.4 points of recall, so INT8 on CPU is only enabled for a checkpoint after it passes the same gate as FP8 on GPU (99.5% argmax agreement and ECE change of 0.005 or less). Native checkpoints are trained with INT8 quantization aware fine tuning in the last epoch so they pass. If a checkpoint fails the gate, BF16 (on v4 and AMX) or FP32 is used.
- LayerNorm, softmax, RoPE, GELU and the scorer run in FP32.
- Attention runs in FP32 with FP16 K and V storage, blocked like FlashAttention so the scores never leave L1 and L2.

## GEMM

The CPU GEMMs are our own micro-kernels in `kime-cpu/src/gemm/`, written with `std::arch` intrinsics. Weights are prepacked into panels for the micro-kernel at load (stored in the prepacked cache).

- Blocking follows the usual GotoBLAS scheme (MC, KC, NC) with sizes picked per microarchitecture from the cache sizes reported by `cpuid` or `sysctl`.
- Epilogues fuse bias, residual, GeGLU, RoPE and the LayerNorm of the next op, exactly as on GPU. On CPU this matters even more, because each extra pass over activations costs memory bandwidth.
- At small M (batch 1, short state) the GEMMs are memory bound on the weights: the s tier is 56M non-embedding parameters, which is 56 MB in INT8. We split N across threads so each core streams a separate slice of the weights and the weights are read once per batch. For a 128 token state on 8 Zen 3 cores at about 40 GB/s of effective DRAM bandwidth, that is about 1.5 ms of weight streaming. The rest is compute at about 0.5 TFLOPS INT8, so W1 lands around 3 to 5 ms.
- Where AMX exists, INT8 GEMMs run on the tiles. MKL reaches about 54 TFLOPS BF16 on a Xeon 8480+. We do not need that peak, but even a simple AMX kernel beats AVX-512 by 4x or more at our sizes.

The `gemm` crate (sarah-quinones) is the fallback for FP32 on any ISA we have not written kernels for, and is used for correctness tests. oneDNN is not a dependency: the operational cost of its C++ build is not worth it for our op set.

## Threading

Laya's 30x slowdown with default PyTorch threads came from oversubscription: the intra-op pool, the inter-op pool and the server's threads all fought for the same cores. kime-cpu owns its threads outright.

- At start, the backend reads the topology (`/sys/devices/system/cpu`, or `sysctl hw.perflevel0` on macOS) and makes one worker group per NUMA node. On hybrid chips (Intel P and E cores, Apple P and E clusters) only performance cores are used, unless `--cpu-cores` says otherwise.
- Each group has one pinned thread per physical core (no SMT siblings) and runs one batch at a time across all its cores. Groups run batches in parallel.
- The hot loop uses a spin barrier with a short spin (about 50 us), then futex parking. There is no rayon and no work stealing in the GEMM path, because work stealing makes the split points, and so the reduction order, depend on timing. That would break determinism.
- The server's tokio runtime and the request threads are kept off the worker cores. With `--cpu-cores 2-15`, cores 0 and 1 are left for the network side.

## Plans on CPU

A lowered CPU plan is a `Vec<Step>`, where each step is a function pointer plus a fixed argument struct with arena offsets resolved. The executor is a loop that calls them in order with a barrier between dependent steps. Steps that do not depend on each other (for example the K and V projections of different question layers) are merged into one parallel region.

## Compat on CPU

The Laya compat model is 421M parameters with FP32 compute and a 512 token sequence per question. It will never be fast on CPU. It is supported for parity and completeness, and uses the same kernels. The target for compat W1 on the 6900HX is 60 ms or less (from Laya's tuned 329 ms), which is about 5.5x from the engine alone.
