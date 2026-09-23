// The kernels of the compat graph that are not GEMMs, compiled with NVRTC when the backend starts.
//
// There are no includes, so NVRTC needs no CUDA headers on the machine: half precision values are
// stored as 16 bit integers and converted with the PTX cvt instructions.
//
// Every kernel reads the batch's real row counts from `n` (tokens, sequences, markers) and is
// launched over the bucket's padded counts, so the same launches work inside a captured graph for
// any batch that fits the bucket. Blocks past the real count return at once.

typedef unsigned short half_t;

#define INF __int_as_float(0x7f800000)

__device__ __forceinline__ float h2f(half_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ half_t f2h(float f) {
    half_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ float ld(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float ld(const half_t* p, size_t i) { return h2f(p[i]); }
__device__ __forceinline__ void st(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void st(half_t* p, size_t i, float v) { p[i] = f2h(v); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// The exact GELU, x/2 (1 + erf(x/sqrt 2)).
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

// out[r] = table[ids[r]], one block per token row.
extern "C" __global__ void embed(float* out, const float* table, const unsigned* ids, const unsigned* n, int d) {
    unsigned r = blockIdx.x;
    if (r >= n[0]) return;
    const float* src = table + (size_t)ids[r] * d;
    float* dst = out + (size_t)r * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) dst[c] = src[c];
}

// LayerNorm with FP32 statistics, one warp per row and four rows per block. `kind` picks the row
// count from `n`. `b` may be null.
template <typename TI, typename TO>
__device__ void layer_norm(TO* out, const TI* x, const float* w, const float* b, const unsigned* n, int kind, int d, float eps) {
    unsigned r = blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (r >= n[kind]) return;
    const TI* xr = x + (size_t)r * d;
    float s = 0.0f;
    for (int c = lane; c < d; c += 32) s += ld(xr, c);
    float mean = warp_sum(s) / d;
    float v = 0.0f;
    for (int c = lane; c < d; c += 32) {
        float t = ld(xr, c) - mean;
        v += t * t;
    }
    float rstd = 1.0f / sqrtf(warp_sum(v) / d + eps);
    for (int c = lane; c < d; c += 32) {
        float y = (ld(xr, c) - mean) * rstd * w[c];
        if (b) y += b[c];
        st(out, (size_t)r * d + c, y);
    }
}

extern "C" __global__ void ln_f32_f32(float* o, const float* x, const float* w, const float* b, const unsigned* n, int k, int d, float e) { layer_norm(o, x, w, b, n, k, d, e); }
extern "C" __global__ void ln_f32_f16(half_t* o, const float* x, const float* w, const float* b, const unsigned* n, int k, int d, float e) { layer_norm(o, x, w, b, n, k, d, e); }
extern "C" __global__ void ln_f16_f16(half_t* o, const half_t* x, const float* w, const float* b, const unsigned* n, int k, int d, float e) { layer_norm(o, x, w, b, n, k, d, e); }
extern "C" __global__ void ln_f16_f32(float* o, const half_t* x, const float* w, const float* b, const unsigned* n, int k, int d, float e) { layer_norm(o, x, w, b, n, k, d, e); }

// y = act(y + b) after a GEMM, one block per row. act is 0 for none, 1 for GELU, 2 for ReLU, and
// `b` may be null.
template <typename T>
__device__ void bias_act(T* y, const float* b, const unsigned* n, int kind, int width, int act) {
    unsigned r = blockIdx.x;
    if (r >= n[kind]) return;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        size_t i = (size_t)r * width + c;
        float v = ld(y, i);
        if (b) v += b[c];
        if (act == 1) v = gelu(v);
        else if (act == 2) v = fmaxf(v, 0.0f);
        st(y, i, v);
    }
}

extern "C" __global__ void bias_act_f32(float* y, const float* b, const unsigned* n, int k, int w, int a) { bias_act(y, b, n, k, w, a); }
extern "C" __global__ void bias_act_f16(half_t* y, const float* b, const unsigned* n, int k, int w, int a) { bias_act(y, b, n, k, w, a); }

// Rotary embedding in place on the q and k parts of each token row, heads of 64, tables [pos, 32].
template <typename T>
__device__ void rope(T* qkv, const float* cosv, const float* sinv, const unsigned* pos, const unsigned* n, int heads) {
    unsigned r = blockIdx.x;
    if (r >= n[0]) return;
    int d = heads * 64;
    unsigned p = pos[r];
    for (int t = threadIdx.x; t < 2 * heads * 32; t += blockDim.x) {
        int part = t / (heads * 32);
        int rem = t % (heads * 32);
        int h = rem / 32, i = rem % 32;
        size_t x = (size_t)r * 3 * d + part * d + h * 64;
        float a = ld(qkv, x + i), b = ld(qkv, x + i + 32);
        float c = cosv[p * 32 + i], s = sinv[p * 32 + i];
        st(qkv, x + i, a * c + -b * s);
        st(qkv, x + i + 32, b * c + a * s);
    }
}

extern "C" __global__ void rope_f16(half_t* x, const float* c, const float* s, const unsigned* p, const unsigned* n, int h) { rope(x, c, s, p, n, h); }
extern "C" __global__ void rope_f32(float* x, const float* c, const float* s, const unsigned* p, const unsigned* n, int h) { rope(x, c, s, p, n, h); }

// q . k over one head, q already scaled and in shared memory.
__device__ __forceinline__ float dot64(const float* q, const half_t* kp) {
    const uint4* k = (const uint4*)kp;
    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < 8; c++) {
        uint4 u = k[c];
        const float* qq = q + c * 8;
        dot += qq[0] * h2f(u.x & 0xffff) + qq[1] * h2f(u.x >> 16);
        dot += qq[2] * h2f(u.y & 0xffff) + qq[3] * h2f(u.y >> 16);
        dot += qq[4] * h2f(u.z & 0xffff) + qq[5] * h2f(u.z >> 16);
        dot += qq[6] * h2f(u.w & 0xffff) + qq[7] * h2f(u.w >> 16);
    }
    return dot;
}

__device__ __forceinline__ float dot64(const float* q, const float* kp) {
    const float4* k = (const float4*)kp;
    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < 16; c++) {
        float4 u = k[c];
        const float* qq = q + c * 4;
        dot += qq[0] * u.x + qq[1] * u.y + qq[2] * u.z + qq[3] * u.w;
    }
    return dot;
}

// Channels 2 lane and 2 lane + 1 of one head.
__device__ __forceinline__ float2 pair(const half_t* v, int lane) {
    unsigned u = ((const unsigned*)v)[lane];
    return make_float2(h2f(u & 0xffff), h2f(u >> 16));
}

__device__ __forceinline__ float2 pair(const float* v, int lane) { return ((const float2*)v)[lane]; }

// Attention over [q | k | v] rows of heads of 64, within each sequence and, when window >= 0, only
// to keys at most `window` positions away. One warp per (query row, head), four rows per block.
// Keys go 32 at a time, one per lane, with an online softmax, and each lane then owns two of the
// 64 output channels.
template <typename T, typename TO>
__device__ void attention(TO* out, const T* qkv, const unsigned* seq, const unsigned* cu, const unsigned* n, int heads, int window) {
    __shared__ float qs[4][64];
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    unsigned i = blockIdx.x * 4 + wid;
    int h = blockIdx.y;
    if (i >= n[0]) return;
    int d = heads * 64;
    size_t stride = 3 * (size_t)d;
    unsigned s = seq[i];
    int lo = cu[s], hi = cu[s + 1];
    int a = lo, b = hi;
    if (window >= 0) {
        a = max(lo, (int)i - window);
        b = min(hi, (int)i + window + 1);
    }
    size_t q = i * stride + h * 64;
    qs[wid][lane] = ld(qkv, q + lane) * 0.125f;
    qs[wid][lane + 32] = ld(qkv, q + lane + 32) * 0.125f;
    __syncwarp();
    float m = -INF, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    for (int j0 = a; j0 < b; j0 += 32) {
        int j = j0 + lane;
        float sc = -INF;
        if (j < b) sc = dot64(qs[wid], qkv + (size_t)j * stride + d + h * 64);
        float mn = fmaxf(m, warp_max(sc));
        float corr = expf(m - mn);
        float p = j < b ? expf(sc - mn) : 0.0f;
        l = l * corr + warp_sum(p);
        acc0 *= corr;
        acc1 *= corr;
        int cnt = min(32, b - j0);
        for (int t = 0; t < cnt; t++) {
            float pt = __shfl_sync(0xffffffff, p, t);
            float2 v = pair(qkv + (size_t)(j0 + t) * stride + 2 * d + h * 64, lane);
            acc0 += pt * v.x;
            acc1 += pt * v.y;
        }
        m = mn;
    }
    float inv = 1.0f / l;
    size_t o = (size_t)i * d + h * 64 + 2 * lane;
    st(out, o, acc0 * inv);
    st(out, o + 1, acc1 * inv);
}

extern "C" __global__ void attention_f32_f32(float* o, const float* x, const unsigned* s, const unsigned* cu, const unsigned* n, int h, int w) { attention(o, x, s, cu, n, h, w); }
extern "C" __global__ void attention_f32_f16(half_t* o, const float* x, const unsigned* s, const unsigned* cu, const unsigned* n, int h, int w) { attention(o, x, s, cu, n, h, w); }
extern "C" __global__ void attention_f16_f32(float* o, const half_t* x, const unsigned* s, const unsigned* cu, const unsigned* n, int h, int w) { attention(o, x, s, cu, n, h, w); }
extern "C" __global__ void attention_f16_f16(half_t* o, const half_t* x, const unsigned* s, const unsigned* cu, const unsigned* n, int h, int w) { attention(o, x, s, cu, n, h, w); }

// out = gelu(x[:, ..inter]) * x[:, inter..], one block per token row.
template <typename T, typename TO>
__device__ void geglu(TO* out, const T* x, const unsigned* n, int inter) {
    unsigned r = blockIdx.x;
    if (r >= n[0]) return;
    size_t xr = (size_t)r * 2 * inter;
    for (int c = threadIdx.x; c < inter; c += blockDim.x)
        st(out, (size_t)r * inter + c, gelu(ld(x, xr + c)) * ld(x, xr + inter + c));
}

extern "C" __global__ void geglu_f32_f32(float* o, const float* x, const unsigned* n, int i) { geglu(o, x, n, i); }
extern "C" __global__ void geglu_f32_f16(half_t* o, const float* x, const unsigned* n, int i) { geglu(o, x, n, i); }
extern "C" __global__ void geglu_f16_f32(float* o, const half_t* x, const unsigned* n, int i) { geglu(o, x, n, i); }
extern "C" __global__ void geglu_f16_f16(half_t* o, const half_t* x, const unsigned* n, int i) { geglu(o, x, n, i); }

// h[r] += table[qtype[seq[r]]], one block per token row.
extern "C" __global__ void add_type(float* h, const float* table, const unsigned* seq, const unsigned* qtype, const unsigned* n, int d) {
    unsigned r = blockIdx.x;
    if (r >= n[0]) return;
    const float* t = table + (size_t)qtype[seq[r]] * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) h[(size_t)r * d + c] += t[c];
}

// out[m] = h[rows[m]], one block per marker.
template <typename T>
__device__ void gather(T* out, const T* h, const unsigned* rows, const unsigned* n, int d) {
    unsigned m = blockIdx.x;
    if (m >= n[2]) return;
    const T* src = h + (size_t)rows[m] * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) out[(size_t)m * d + c] = src[c];
}

extern "C" __global__ void gather_f32(float* o, const float* h, const unsigned* r, const unsigned* n, int d) { gather(o, h, r, n, d); }
extern "C" __global__ void gather_f16(half_t* o, const half_t* h, const unsigned* r, const unsigned* n, int d) { gather(o, h, r, n, d); }

// Per sequence, its first row followed by [top1, top1 - top2, entropy / ln k, k / 255] over the
// softmax of its markers' logits, with k at least 2. One block per sequence, the statistics on one
// thread in the order the CPU takes them.
extern "C" __global__ void act_features(float* out, const float* h, const float* logits, const unsigned* cu, const unsigned* mcu, const unsigned* n, int d) {
    unsigned s = blockIdx.x;
    if (s >= n[1]) return;
    float* o = out + (size_t)s * (d + 4);
    bool empty = cu[s + 1] == cu[s];
    const float* src = h + (size_t)cu[s] * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) o[c] = empty ? 0.0f : src[c];
    if (threadIdx.x != 0) return;
    const float* l = logits + mcu[s];
    int k = mcu[s + 1] - mcu[s];
    float kf = (float)max(k, 2);
    if (k == 0) {
        o[d] = 0.0f;
        o[d + 1] = 0.0f;
        o[d + 2] = 0.0f;
        o[d + 3] = kf / 255.0f;
        return;
    }
    float mx = -INF;
    for (int i = 0; i < k; i++) mx = fmaxf(mx, l[i]);
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += expf(l[i] - mx);
    float top1 = -INF, top2 = 0.0f, ent = 0.0f;
    for (int i = 0; i < k; i++) {
        float q = expf(l[i] - mx) / sum;
        ent += q * logf(fmaxf(q, 1e-9f));
        if (i == 0 || q > top1) {
            if (i != 0) top2 = top1;
            top1 = q;
        } else if (q > top2) {
            top2 = q;
        }
    }
    o[d] = top1;
    o[d + 1] = top1 - top2;
    o[d + 2] = -ent / logf(kf);
    o[d + 3] = kf / 255.0f;
}

// FP32 to FP16, for weights at load.
extern "C" __global__ void to_f16(half_t* out, const float* x, size_t len) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < len; i += (size_t)gridDim.x * blockDim.x)
        out[i] = f2h(x[i]);
}
