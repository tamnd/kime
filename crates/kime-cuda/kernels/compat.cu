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

// Sum over a block of LN_THREADS threads. `red` is shared scratch of one float per warp, and every
// thread gets the total.
#define LN_THREADS 128
#define LN_PER (1024 / LN_THREADS)

__device__ __forceinline__ float block_sum(float v, float* red) {
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (int k = 0; k < LN_THREADS / 32; k++) t += red[k];
    return t;
}

// LayerNorm with FP32 statistics, one block of LN_THREADS per row with the row held in registers,
// so d is at most 1024. `kind` picks the row count from `n`. `b` may be null.
template <typename TI, typename TO>
__device__ void layer_norm(TO* out, const TI* x, const float* w, const float* b, const unsigned* n, int kind, int d, float eps) {
    __shared__ float red[LN_THREADS / 32];
    unsigned r = blockIdx.x;
    if (r >= n[kind]) return;
    const TI* xr = x + (size_t)r * d;
    float v[LN_PER];
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < LN_PER; k++) {
        int c = threadIdx.x + k * LN_THREADS;
        v[k] = c < d ? ld(xr, c) : 0.0f;
        s += v[k];
    }
    float mean = block_sum(s, red) / d;
    float q = 0.0f;
#pragma unroll
    for (int k = 0; k < LN_PER; k++) {
        int c = threadIdx.x + k * LN_THREADS;
        float t = c < d ? v[k] - mean : 0.0f;
        q += t * t;
    }
    float rstd = 1.0f / sqrtf(block_sum(q, red) / d + eps);
#pragma unroll
    for (int k = 0; k < LN_PER; k++) {
        int c = threadIdx.x + k * LN_THREADS;
        if (c < d) {
            float y = (v[k] - mean) * rstd * w[c];
            if (b) y += b[c];
            st(out, (size_t)r * d + c, y);
        }
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

// Attention over [q | k | v] rows of heads of 64, within each sequence and, when window >= 0, only
// to keys at most `window` positions away. One block per ATT_Q query rows and one head. The keys
// any of the rows can see are walked ATT_K at a time, each tile of k and v staged in shared memory
// once for the whole block. Each of the ATT_W warps keeps an online softmax for ATT_R rows: a lane
// scores one key of the tile, then owns two of the 64 output channels. The sums are split in four
// so that no chain of dependent adds is longer than 16.
#ifndef ATT_W
#define ATT_W 16
#endif
#ifndef ATT_R
#define ATT_R 1
#endif
#define ATT_Q (ATT_W * ATT_R)
#define ATT_K 32
// Elements of a k or v tile each thread moves.
#define ATT_E (ATT_K * 64 / (32 * ATT_W))

template <typename T, typename TO>
__device__ void attention(TO* out, const T* qkv, const unsigned* seq, const unsigned* cu, const unsigned* n, int heads, int window) {
    __shared__ float qs[ATT_Q][64];
    __shared__ float ks[ATT_K][65];
    __shared__ __align__(16) float vs[ATT_K][64];
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int nt = n[0];
    int i0 = blockIdx.x * ATT_Q;
    if (i0 >= nt) return;
    int h = blockIdx.y;
    int d = heads * 64;
    size_t stride = 3 * (size_t)d;
    // Rows are sorted by sequence, so both ends of a row's key range only grow with the row, and
    // the block's range runs from its first row's start to its last row's end.
    int lo[ATT_R], hi[ATT_R];
    for (int k = 0; k < ATT_R; k++) {
        int i = i0 + wid * ATT_R + k;
        if (i < nt) {
            unsigned sq = seq[i];
            lo[k] = cu[sq];
            hi[k] = cu[sq + 1];
            if (window >= 0) {
                lo[k] = max(lo[k], i - window);
                hi[k] = min(hi[k], i + window + 1);
            }
        } else {
            lo[k] = hi[k] = 0;
        }
    }
    int last = min(i0 + ATT_Q, nt) - 1;
    int a = cu[seq[i0]], b = cu[seq[last] + 1];
    if (window >= 0) {
        a = max(a, i0 - window);
        b = min(b, last + window + 1);
    }
    for (int e = threadIdx.x; e < ATT_Q * 64; e += blockDim.x) {
        int r = e / 64, c = e % 64;
        qs[r][c] = i0 + r < nt ? ld(qkv, (size_t)(i0 + r) * stride + h * 64 + c) * 0.125f : 0.0f;
    }
    float m[ATT_R], l[ATT_R], acc0[ATT_R], acc1[ATT_R];
    for (int k = 0; k < ATT_R; k++) {
        m[k] = -INF;
        l[k] = acc0[k] = acc1[k] = 0.0f;
    }
    __syncthreads();
    float qr[ATT_R][64];
#pragma unroll
    for (int k = 0; k < ATT_R; k++)
#pragma unroll
        for (int c = 0; c < 64; c++) qr[k][c] = qs[wid * ATT_R + k][c];
    // The next tile is read into registers while the current one is scored.
    float kr[ATT_E], vr[ATT_E];
#pragma unroll
    for (int u = 0; u < ATT_E; u++) {
        int e = threadIdx.x + u * 32 * ATT_W, j = a + e / 64;
        size_t at = (size_t)j * stride + h * 64 + e % 64;
        kr[u] = j < b ? ld(qkv, at + d) : 0.0f;
        vr[u] = j < b ? ld(qkv, at + 2 * d) : 0.0f;
    }
    for (int j0 = a; j0 < b; j0 += ATT_K) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < ATT_E; u++) {
            int e = threadIdx.x + u * 32 * ATT_W;
            ks[e / 64][e % 64] = kr[u];
            vs[e / 64][e % 64] = vr[u];
        }
        __syncthreads();
        int jn = j0 + ATT_K;
        if (jn < b) {
#pragma unroll
            for (int u = 0; u < ATT_E; u++) {
                int e = threadIdx.x + u * 32 * ATT_W, j = jn + e / 64;
                size_t at = (size_t)j * stride + h * 64 + e % 64;
                kr[u] = j < b ? ld(qkv, at + d) : 0.0f;
                vr[u] = j < b ? ld(qkv, at + 2 * d) : 0.0f;
            }
        }
        int j = j0 + lane;
#pragma unroll
        for (int k = 0; k < ATT_R; k++) {
            if (hi[k] <= j0 || lo[k] >= j0 + ATT_K) continue;
            float sc = -INF;
            if (j >= lo[k] && j < hi[k]) {
                float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int c = 0; c < 64; c++) s4[c % 4] += qr[k][c] * ks[lane][c];
                sc = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            }
            float mn = fmaxf(m[k], warp_max(sc));
            float corr = expf(m[k] - mn);
            float p = sc == -INF ? 0.0f : expf(sc - mn);
            l[k] = l[k] * corr + warp_sum(p);
            float s0[2] = {0.0f, 0.0f}, s1[2] = {0.0f, 0.0f};
#pragma unroll
            for (int t = 0; t < ATT_K; t++) {
                float pt = __shfl_sync(0xffffffff, p, t);
                float2 v = *(const float2*)&vs[t][2 * lane];
                s0[t % 2] += pt * v.x;
                s1[t % 2] += pt * v.y;
            }
            acc0[k] = acc0[k] * corr + (s0[0] + s0[1]);
            acc1[k] = acc1[k] * corr + (s1[0] + s1[1]);
            m[k] = mn;
        }
    }
    for (int k = 0; k < ATT_R; k++) {
        int i = i0 + wid * ATT_R + k;
        if (i >= nt) continue;
        float inv = l[k] > 0.0f ? 1.0f / l[k] : 0.0f;
        size_t o = (size_t)i * d + h * 64 + 2 * lane;
        st(out, o, acc0[k] * inv);
        st(out, o + 1, acc1[k] * inv);
    }
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
