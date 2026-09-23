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

// LayerNorm with FP32 statistics, one warp per row with the row held in registers, so d is at
// most 1024, and LN_ROWS rows per block. `kind` picks the row count from `n`. `b` may be null.
#define LN_ROWS 4
#define LN_PER (1024 / 32)

template <typename TI, typename TO>
__device__ void layer_norm(TO* out, const TI* x, const float* w, const float* b, const unsigned* n, int kind, int d, float eps) {
    int lane = threadIdx.x % 32;
    unsigned r = blockIdx.x * LN_ROWS + threadIdx.x / 32;
    if (r >= n[kind]) return;
    const TI* xr = x + (size_t)r * d;
    float v[LN_PER];
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < LN_PER; k++) {
        int c = lane + k * 32;
        v[k] = c < d ? ld(xr, c) : 0.0f;
        s += v[k];
    }
    float mean = warp_sum(s) / d;
    float q = 0.0f;
#pragma unroll
    for (int k = 0; k < LN_PER; k++) {
        int c = lane + k * 32;
        float t = c < d ? v[k] - mean : 0.0f;
        q += t * t;
    }
    float rstd = 1.0f / sqrtf(warp_sum(q) / d + eps);
#pragma unroll
    for (int k = 0; k < LN_PER; k++) {
        int c = lane + k * 32;
        if (c < d) v[k] = (v[k] - mean) * rstd * w[c] + (b ? b[c] : 0.0f);
    }
#pragma unroll
    for (int k = 0; k < LN_PER; k++) {
        int c = lane + k * 32;
        if (c < d) st(out, (size_t)r * d + c, v[k]);
    }
}

extern "C" __global__ void ln_f32_f32(float* o, const float* x, const float* w, const float* b, const unsigned* n, int k, int d, float e) { layer_norm(o, x, w, b, n, k, d, e); }
extern "C" __global__ void ln_f32_f16(half_t* o, const float* x, const float* w, const float* b, const unsigned* n, int k, int d, float e) { layer_norm(o, x, w, b, n, k, d, e); }
extern "C" __global__ void ln_f16_f16(half_t* o, const half_t* x, const float* w, const float* b, const unsigned* n, int k, int d, float e) { layer_norm(o, x, w, b, n, k, d, e); }
extern "C" __global__ void ln_f16_f32(float* o, const half_t* x, const float* w, const float* b, const unsigned* n, int k, int d, float e) { layer_norm(o, x, w, b, n, k, d, e); }

// y = act(y + b) after a GEMM, one block per row. act is 0 for none, 1 for GELU, 2 for ReLU, and
// `b` may be null.
// The elementwise kernels below read EW_U elements per thread before they write any: the compiler
// cannot tell the buffers apart, so a write between two reads would make the second wait on the
// first.
#define EW_U 4

template <typename T>
__device__ void bias_act(T* y, const float* b, const unsigned* n, int kind, int width, int act) {
    unsigned r = blockIdx.x;
    if (r >= n[kind]) return;
    size_t row = (size_t)r * width;
    for (int c0 = threadIdx.x; c0 < width; c0 += EW_U * blockDim.x) {
        float v[EW_U];
#pragma unroll
        for (int u = 0; u < EW_U; u++) {
            int c = c0 + u * blockDim.x;
            v[u] = c < width ? ld(y, row + c) + (b ? b[c] : 0.0f) : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < EW_U; u++) {
            int c = c0 + u * blockDim.x;
            float t = act == 1 ? gelu(v[u]) : act == 2 ? fmaxf(v[u], 0.0f) : v[u];
            if (c < width) st(y, row + c, t);
        }
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
    for (int t0 = threadIdx.x; t0 < 2 * heads * 32; t0 += EW_U * blockDim.x) {
        float a[EW_U], b[EW_U], c[EW_U], sn[EW_U];
#pragma unroll
        for (int u = 0; u < EW_U; u++) {
            int t = t0 + u * blockDim.x, part = t / (heads * 32), rem = t % (heads * 32), i = rem % 32;
            size_t x = (size_t)r * 3 * d + part * d + rem / 32 * 64 + i;
            bool in = t < 2 * heads * 32;
            a[u] = in ? ld(qkv, x) : 0.0f;
            b[u] = in ? ld(qkv, x + 32) : 0.0f;
            c[u] = cosv[p * 32 + i];
            sn[u] = sinv[p * 32 + i];
        }
#pragma unroll
        for (int u = 0; u < EW_U; u++) {
            int t = t0 + u * blockDim.x, part = t / (heads * 32), rem = t % (heads * 32), i = rem % 32;
            if (t >= 2 * heads * 32) continue;
            size_t x = (size_t)r * 3 * d + part * d + rem / 32 * 64 + i;
            st(qkv, x, a[u] * c[u] + -b[u] * sn[u]);
            st(qkv, x + 32, b[u] * c[u] + a[u] * sn[u]);
        }
    }
}

extern "C" __global__ void rope_f16(half_t* x, const float* c, const float* s, const unsigned* p, const unsigned* n, int h) { rope(x, c, s, p, n, h); }
extern "C" __global__ void rope_f32(float* x, const float* c, const float* s, const unsigned* p, const unsigned* n, int h) { rope(x, c, s, p, n, h); }

// Attention over [q | k | v] rows of heads of 64, within each sequence and, when window >= 0, only
// to keys at most `window` positions away. One block per ATT_Q query rows and one head, and each
// of the ATT_W warps keeps an online softmax for ATT_R of the rows. The keys any of the rows can
// see are walked 32 at a time, each tile of k and v staged in shared memory once for the block,
// with the next tile read into registers while the current one is scored. A lane scores one key
// of the tile for all of its warp's rows, reading each k value once and the q values as
// broadcasts, then owns two of the 64 output channels. Each lane keeps its own part of a row's
// softmax sum, and the warp adds them up at the end.
#define ATT_W 8
#define ATT_R 2
#define ATT_Q (ATT_W * ATT_R)
// Elements of a k or v tile and of the q rows each thread moves, and the padded k row, which keeps
// the lanes' 16 byte reads of 32 different rows free of bank conflicts.
#define ATT_E (32 * 64 / (32 * ATT_W))
#define ATT_QE (ATT_Q * 64 / (32 * ATT_W))
#define ATT_KS 68

template <typename T>
__device__ __forceinline__ void att_load(float (&kr)[ATT_E], float (&vr)[ATT_E], const T* qkv, int j0, int b, size_t stride, int col) {
#pragma unroll
    for (int u = 0; u < ATT_E; u++) {
        int e = threadIdx.x + u * 32 * ATT_W, j = j0 + e / 64;
        size_t at = (size_t)j * stride + col + e % 64;
        kr[u] = j < b ? ld(qkv, at) : 0.0f;
        vr[u] = j < b ? ld(qkv, at + stride / 3) : 0.0f;
    }
}

template <typename T, typename TO>
__device__ void attention(TO* out, const T* qkv, const unsigned* seq, const unsigned* cu, const unsigned* n, int heads, int window) {
    __shared__ __align__(16) float qs[ATT_Q][64];
    __shared__ __align__(16) float ks[32][ATT_KS];
    __shared__ __align__(16) float vs[32][64];
    __shared__ __align__(16) float ps[ATT_W][ATT_R][32];
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
#pragma unroll
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
    // Every load is issued before any store to shared memory: the compiler cannot tell that qkv
    // is not shared memory, so a store between two loads would make the second wait on the first.
    float qr[ATT_QE];
#pragma unroll
    for (int u = 0; u < ATT_QE; u++) {
        int e = threadIdx.x + u * 32 * ATT_W, r = e / 64;
        qr[u] = i0 + r < nt ? ld(qkv, (size_t)(i0 + r) * stride + h * 64 + e % 64) * 0.125f : 0.0f;
    }
    float kr[ATT_E], vr[ATT_E];
    att_load(kr, vr, qkv, a, b, stride, d + h * 64);
#pragma unroll
    for (int u = 0; u < ATT_QE; u++) {
        int e = threadIdx.x + u * 32 * ATT_W;
        qs[e / 64][e % 64] = qr[u];
    }
    float m[ATT_R], l[ATT_R], acc0[ATT_R], acc1[ATT_R];
#pragma unroll
    for (int k = 0; k < ATT_R; k++) {
        m[k] = -INF;
        l[k] = acc0[k] = acc1[k] = 0.0f;
    }
    for (int j0 = a; j0 < b; j0 += 32) {
        if (j0 > a) __syncthreads();
#pragma unroll
        for (int u = 0; u < ATT_E; u++) {
            int e = threadIdx.x + u * 32 * ATT_W;
            ks[e / 64][e % 64] = kr[u];
            vs[e / 64][e % 64] = vr[u];
        }
        __syncthreads();
        if (j0 + 32 < b) att_load(kr, vr, qkv, j0 + 32, b, stride, d + h * 64);
        float s[ATT_R][4];
#pragma unroll
        for (int k = 0; k < ATT_R; k++) s[k][0] = s[k][1] = s[k][2] = s[k][3] = 0.0f;
#pragma unroll
        for (int c = 0; c < 64; c += 4) {
            float4 kv = *(const float4*)&ks[lane][c];
#pragma unroll
            for (int k = 0; k < ATT_R; k++) {
                float4 qv = *(const float4*)&qs[wid * ATT_R + k][c];
                s[k][0] += qv.x * kv.x;
                s[k][1] += qv.y * kv.y;
                s[k][2] += qv.z * kv.z;
                s[k][3] += qv.w * kv.w;
            }
        }
        // The rows' maxima are reduced side by side, so their shuffles overlap. A row with none of
        // its keys in the tile keeps its state, since its maximum is still -inf.
        int j = j0 + lane;
        float sc[ATT_R], mx[ATT_R], corr[ATT_R];
#pragma unroll
        for (int k = 0; k < ATT_R; k++) {
            bool in = j >= lo[k] && j < hi[k];
            sc[k] = in ? (s[k][0] + s[k][1]) + (s[k][2] + s[k][3]) : -INF;
            mx[k] = sc[k];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int k = 0; k < ATT_R; k++) mx[k] = fmaxf(mx[k], __shfl_xor_sync(0xffffffff, mx[k], o));
#pragma unroll
        for (int k = 0; k < ATT_R; k++) {
            float mn = fmaxf(m[k], mx[k]);
            corr[k] = mn != -INF ? __expf(m[k] - mn) : 1.0f;
            float p = sc[k] != -INF ? __expf(sc[k] - mn) : 0.0f;
            l[k] = l[k] * corr[k] + p;
            m[k] = mn;
            ps[wid][k][lane] = p;
        }
        __syncwarp();
        float s0[ATT_R][2], s1[ATT_R][2];
#pragma unroll
        for (int k = 0; k < ATT_R; k++) s0[k][0] = s0[k][1] = s1[k][0] = s1[k][1] = 0.0f;
#pragma unroll
        for (int t = 0; t < 32; t += 4) {
            float2 v[4];
#pragma unroll
            for (int x = 0; x < 4; x++) v[x] = *(const float2*)&vs[t + x][2 * lane];
#pragma unroll
            for (int k = 0; k < ATT_R; k++) {
                float4 pt = *(const float4*)&ps[wid][k][t];
                s0[k][0] += pt.x * v[0].x + pt.z * v[2].x;
                s0[k][1] += pt.y * v[1].x + pt.w * v[3].x;
                s1[k][0] += pt.x * v[0].y + pt.z * v[2].y;
                s1[k][1] += pt.y * v[1].y + pt.w * v[3].y;
            }
        }
#pragma unroll
        for (int k = 0; k < ATT_R; k++) {
            acc0[k] = acc0[k] * corr[k] + (s0[k][0] + s0[k][1]);
            acc1[k] = acc1[k] * corr[k] + (s1[k][0] + s1[k][1]);
        }
        __syncwarp();
    }
#pragma unroll
    for (int k = 0; k < ATT_R; k++) {
        int i = i0 + wid * ATT_R + k;
        float sum = warp_sum(l[k]);
        if (i >= nt) continue;
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        size_t o = (size_t)i * d + h * 64 + 2 * lane;
        st(out, o, acc0[k] * inv);
        st(out, o + 1, acc1[k] * inv);
    }
}

extern "C" __global__ void __launch_bounds__(32 * ATT_W) attention_f32_f32(float* o, const float* x, const unsigned* s, const unsigned* cu, const unsigned* n, int h, int w) { attention(o, x, s, cu, n, h, w); }
extern "C" __global__ void __launch_bounds__(32 * ATT_W) attention_f32_f16(half_t* o, const float* x, const unsigned* s, const unsigned* cu, const unsigned* n, int h, int w) { attention(o, x, s, cu, n, h, w); }
extern "C" __global__ void __launch_bounds__(32 * ATT_W) attention_f16_f32(float* o, const half_t* x, const unsigned* s, const unsigned* cu, const unsigned* n, int h, int w) { attention(o, x, s, cu, n, h, w); }
extern "C" __global__ void __launch_bounds__(32 * ATT_W) attention_f16_f16(half_t* o, const half_t* x, const unsigned* s, const unsigned* cu, const unsigned* n, int h, int w) { attention(o, x, s, cu, n, h, w); }

// out = gelu(x[:, ..inter]) * x[:, inter..], one block per token row.
template <typename T, typename TO>
__device__ void geglu(TO* out, const T* x, const unsigned* n, int inter) {
    unsigned r = blockIdx.x;
    if (r >= n[0]) return;
    size_t xr = (size_t)r * 2 * inter;
    for (int c0 = threadIdx.x; c0 < inter; c0 += EW_U * blockDim.x) {
        float g[EW_U], u2[EW_U];
#pragma unroll
        for (int u = 0; u < EW_U; u++) {
            int c = c0 + u * blockDim.x;
            g[u] = c < inter ? ld(x, xr + c) : 0.0f;
            u2[u] = c < inter ? ld(x, xr + inter + c) : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < EW_U; u++) {
            int c = c0 + u * blockDim.x;
            if (c < inter) st(out, (size_t)r * inter + c, gelu(g[u]) * u2[u]);
        }
    }
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
