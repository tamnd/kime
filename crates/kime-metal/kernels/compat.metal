// The kernels of the compat graph, ported from kime-cuda's compat.cu and compiled from source when
// the backend starts.
//
// Every kernel reads the batch's real row counts from `n` (tokens, sequences, markers) and is
// dispatched over the bucket's padded counts, so the same encoding works for any batch that fits
// the bucket. Threadgroups past the real count return at once.

#include <metal_stdlib>
using namespace metal;

#define instantiate(name, func, ...) \
    template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

inline float ld(device const float* p, ulong i) { return p[i]; }
inline float ld(device const half* p, ulong i) { return float(p[i]); }
inline void st(device float* p, ulong i, float v) { p[i] = v; }
inline void st(device half* p, ulong i, float v) { p[i] = half(v); }

// The exact GELU, x/2 (1 + erf(x/sqrt 2)), with erfc from the same rational fit the CPU backend
// uses, good to about 1e-7 relative over the whole range. Metal has no erf.
inline float gelu(float x) {
    float a = min(fabs(x), 13.0f);
    float t = 1.0f / (1.0f + 0.5f * M_SQRT1_2_F * a);
    float r = 0.17087277f;
    r = r * t - 0.82215223f;
    r = r * t + 1.4885159f;
    r = r * t - 1.135204f;
    r = r * t + 0.27886807f;
    r = r * t - 0.18628806f;
    r = r * t + 0.09678418f;
    r = r * t + 0.37409196f;
    r = r * t + 1.0000237f;
    r = r * t - 1.2655122f;
    // a*a split so that the square is exact: ah has 12 bits of mantissa.
    float ah = as_type<float>(as_type<uint>(a) & 0xfffff000u);
    float q = t * precise::exp(-0.5f * ah * ah + (0.5f * (ah - a) * (ah + a) + r));
    float p = x >= 0.0f ? 1.0f - 0.5f * q : 0.5f * q;
    return x < -13.0f ? -0.0f : x * p;
}

// out[r] = table[ids[r]], one threadgroup per token row.
kernel void embed(device float* out [[buffer(0)]], device const float* table [[buffer(1)]],
                  device const uint* ids [[buffer(2)]], device const uint* n [[buffer(3)]],
                  constant int& d [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                  uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    if (r >= n[0]) return;
    device const float* src = table + (ulong)ids[r] * d;
    device float* dst = out + (ulong)r * d;
    for (int c = t; c < d; c += nt) dst[c] = src[c];
}

// LayerNorm with FP32 statistics, one simdgroup per row with the row held in registers, so d is at
// most 1024, and LN_ROWS rows per threadgroup. `kind` picks the row count from `n`.
#define LN_ROWS 4
#define LN_PER (1024 / 32)

template <typename TI, typename TO>
kernel void layer_norm(device TO* out [[buffer(0)]], device const TI* x [[buffer(1)]],
                       device const float* w [[buffer(2)]], device const float* b [[buffer(3)]],
                       device const uint* n [[buffer(4)]], constant int& kind [[buffer(5)]],
                       constant int& d [[buffer(6)]], constant float& eps [[buffer(7)]],
                       constant int& has_b [[buffer(8)]], uint g [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = g * LN_ROWS + sg;
    if (r >= n[kind]) return;
    device const TI* xr = x + (ulong)r * d;
    float v[LN_PER];
    float s = 0.0f;
    for (int k = 0; k < LN_PER; k++) {
        int c = lane + k * 32;
        v[k] = c < d ? ld(xr, c) : 0.0f;
        s += v[k];
    }
    float mean = simd_sum(s) / d;
    float q = 0.0f;
    for (int k = 0; k < LN_PER; k++) {
        int c = lane + k * 32;
        float t = c < d ? v[k] - mean : 0.0f;
        q += t * t;
    }
    float rstd = 1.0f / precise::sqrt(simd_sum(q) / d + eps);
    for (int k = 0; k < LN_PER; k++) {
        int c = lane + k * 32;
        if (c < d) st(out, (ulong)r * d + c, (v[k] - mean) * rstd * w[c] + (has_b ? b[c] : 0.0f));
    }
}

instantiate("ln_f32_f32", layer_norm, float, float)
instantiate("ln_f32_f16", layer_norm, float, half)
instantiate("ln_f16_f32", layer_norm, half, float)
instantiate("ln_f16_f16", layer_norm, half, half)

// c[m, n] = a[m, k] w[n, k]^T on simdgroup matrices with FP32 accumulation, then the bias, the
// activation and the residual add fused into the store. flags: bit 0 bias, bits 1-2 activation
// (1 GELU, 2 ReLU), bit 3 accumulate into c. A threadgroup of four simdgroups computes a GM x GN
// tile, each simdgroup a 16 x 32 quarter of it as 2 x 4 fragments; more fragments than that run out
// of registers. Both inputs are staged GK deep in threadgroup memory as TT, FP32 or FP16, read from
// device memory four at a time. K must be a multiple of 4.
#define GM 32
#define GN 64
#define GK 16
#define GKP (GK + 4)
#define GT 128

template <typename TA, typename TW, typename TO, typename TT>
kernel void gemm(device TO* c [[buffer(0)]], device const TA* a [[buffer(1)]],
                 device const TW* w [[buffer(2)]], device const float* bias [[buffer(3)]],
                 device const uint* n [[buffer(4)]], constant int& kind [[buffer(5)]],
                 constant int& K [[buffer(6)]], constant int& N [[buffer(7)]],
                 constant int& flags [[buffer(8)]], uint2 g [[threadgroup_position_in_grid]],
                 uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    int M = n[kind];
    int m0 = g.y * GM, n0 = g.x * GN;
    if (m0 >= M) return;
    // The tiles, and after the last one the FP32 output tile.
    threadgroup float sh[GM * GN];
    threadgroup TT* as = (threadgroup TT*)sh;
    threadgroup TT* ws = as + GM * GKP;
    simdgroup_float8x8 acc[2][4];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    int sm = (sg / 2) * 16, sn = (sg % 2) * 32;
    for (int k0 = 0; k0 < K; k0 += GK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = t; e < GM * GK / 4; e += GT) {
            int r = e / (GK / 4), k = e % (GK / 4) * 4, gr = m0 + r, gk = k0 + k;
            vec<TT, 4> v = 0;
            if (gr < M && gk < K) v = vec<TT, 4>(*(device const vec<TA, 4>*)(a + (ulong)gr * K + gk));
            *(threadgroup vec<TT, 4>*)&as[r * GKP + k] = v;
        }
        for (int e = t; e < GN * GK / 4; e += GT) {
            int r = e / (GK / 4), k = e % (GK / 4) * 4, gn = n0 + r, gk = k0 + k;
            vec<TT, 4> v = 0;
            if (gn < N && gk < K) v = vec<TT, 4>(*(device const vec<TW, 4>*)(w + (ulong)gn * K + gk));
            *(threadgroup vec<TT, 4>*)&ws[r * GKP + k] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GK; kk += 8) {
            simdgroup_matrix<TT, 8, 8> am[2], bm[4];
            for (int i = 0; i < 2; i++) simdgroup_load(am[i], &as[(sm + 8 * i) * GKP + kk], GKP);
            for (int j = 0; j < 4; j++) simdgroup_load(bm[j], &ws[(sn + 8 * j) * GKP + kk], GKP, ulong2(0, 0), true);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) simdgroup_store(acc[i][j], &sh[(sm + 8 * i) * GN + sn + 8 * j], GN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int act = (flags >> 1) & 3;
    for (int e = t; e < GM * GN; e += GT) {
        int r = e / GN, col = e % GN, gr = m0 + r, gn = n0 + col;
        if (gr >= M || gn >= N) continue;
        float v = sh[e] + (flags & 1 ? bias[gn] : 0.0f);
        v = act == 1 ? gelu(v) : act == 2 ? max(v, 0.0f) : v;
        ulong at = (ulong)gr * N + gn;
        if (flags & 8) v += ld(c, at);
        st(c, at, v);
    }
}

instantiate("gemm_f32_f32_f32", gemm, float, float, float, float)
instantiate("gemm_f32_f16_f16", gemm, float, half, half, half)
instantiate("gemm_f16_f16_f32", gemm, half, half, float, half)
instantiate("gemm_f16_f16_f16", gemm, half, half, half, half)

// Rotary embedding in place on the q and k parts of each token row, heads of 64, tables [pos, 32].
template <typename T>
kernel void rope(device T* qkv [[buffer(0)]], device const float* cosv [[buffer(1)]],
                 device const float* sinv [[buffer(2)]], device const uint* pos [[buffer(3)]],
                 device const uint* n [[buffer(4)]], constant int& heads [[buffer(5)]],
                 uint r [[threadgroup_position_in_grid]], uint t0 [[thread_position_in_threadgroup]],
                 uint nt [[threads_per_threadgroup]]) {
    if (r >= n[0]) return;
    int d = heads * 64;
    uint p = pos[r];
    for (int t = t0; t < 2 * heads * 32; t += nt) {
        int part = t / (heads * 32), rem = t % (heads * 32), i = rem % 32;
        ulong x = (ulong)r * 3 * d + part * d + rem / 32 * 64 + i;
        float a = ld(qkv, x), b = ld(qkv, x + 32), c = cosv[p * 32 + i], s = sinv[p * 32 + i];
        st(qkv, x, a * c - b * s);
        st(qkv, x + 32, b * c + a * s);
    }
}

instantiate("rope_f32", rope, float)
instantiate("rope_f16", rope, half)

// Attention over [q | k | v] rows of heads of 64, with q and k rotated as they are loaded when
// `roped` is set, within each sequence and, when window >= 0, only to keys at most `window`
// positions away. One threadgroup per ATT_Q query rows and one head, and each of the ATT_W
// simdgroups keeps an online softmax for ATT_R of the rows. The keys any of the rows can see are
// walked 32 at a time, each tile of k and v staged in threadgroup memory once. A lane scores one
// key of the tile for all of its simdgroup's rows, then owns two of the 64 output channels.
#define ATT_W 8
#define ATT_R 2
#define ATT_Q (ATT_W * ATT_R)
#define ATT_E (32 * 64 / (32 * ATT_W))
#define ATT_QE (ATT_Q * 64 / (32 * ATT_W))
#define ATT_KS 68

inline void rotate(thread float& a, thread float& b, device const float* cosv, device const float* sinv, uint p, int c) {
    float cs = cosv[p * 32 + c], sn = sinv[p * 32 + c], x = a;
    a = x * cs - b * sn;
    b = b * cs + x * sn;
}

template <typename T, typename TO>
kernel void attention(device TO* out [[buffer(0)]], device const T* qkv [[buffer(1)]],
                      device const float* cosv [[buffer(2)]], device const float* sinv [[buffer(3)]],
                      device const uint* pos [[buffer(4)]], device const uint* seq [[buffer(5)]],
                      device const uint* cu [[buffer(6)]], device const uint* n [[buffer(7)]],
                      constant int& heads [[buffer(8)]], constant int& window [[buffer(9)]],
                      constant int& roped [[buffer(10)]], uint2 g [[threadgroup_position_in_grid]],
                      uint wid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[ATT_Q][64];
    threadgroup float ks[32][ATT_KS];
    threadgroup float vs[32][64];
    threadgroup float ps[ATT_W][ATT_R][32];
    int nt = n[0];
    int i0 = g.x * ATT_Q;
    if (i0 >= nt) return;
    int h = g.y;
    int d = heads * 64;
    ulong stride = 3 * (ulong)d;
    int lo[ATT_R], hi[ATT_R];
    for (int k = 0; k < ATT_R; k++) {
        int i = i0 + wid * ATT_R + k;
        if (i < nt) {
            uint sq = seq[i];
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
    for (int u = 0; u < ATT_QE; u += 2) {
        int i = i0 + wid + ATT_W * (u / 2);
        ulong at = (ulong)i * stride + h * 64 + lane;
        bool in = i < nt;
        float q0 = in ? ld(qkv, at) : 0.0f, q1 = in ? ld(qkv, at + 32) : 0.0f;
        if (roped) rotate(q0, q1, cosv, sinv, in ? pos[i] : 0, lane);
        qs[wid + ATT_W * (u / 2)][lane] = q0 * 0.125f;
        qs[wid + ATT_W * (u / 2)][lane + 32] = q1 * 0.125f;
    }
    float m[ATT_R], l[ATT_R], acc0[ATT_R], acc1[ATT_R];
    for (int k = 0; k < ATT_R; k++) {
        m[k] = -INFINITY;
        l[k] = acc0[k] = acc1[k] = 0.0f;
    }
    int col = d + h * 64;
    for (int j0 = a; j0 < b; j0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int u = 0; u < ATT_E; u += 2) {
            int r = wid + ATT_W * (u / 2), j = j0 + r;
            ulong at = (ulong)j * stride + col + lane;
            bool in = j < b;
            float k0 = in ? ld(qkv, at) : 0.0f, k1 = in ? ld(qkv, at + 32) : 0.0f;
            if (roped) rotate(k0, k1, cosv, sinv, in ? pos[j] : 0, lane);
            ks[r][lane] = k0;
            ks[r][lane + 32] = k1;
            vs[r][lane] = in ? ld(qkv, at + (ulong)d) : 0.0f;
            vs[r][lane + 32] = in ? ld(qkv, at + (ulong)d + 32) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[ATT_R][4];
        for (int k = 0; k < ATT_R; k++) s[k][0] = s[k][1] = s[k][2] = s[k][3] = 0.0f;
        for (int c = 0; c < 64; c += 4) {
            float4 kv = *(threadgroup const float4*)&ks[lane][c];
            for (int k = 0; k < ATT_R; k++) {
                float4 qv = *(threadgroup const float4*)&qs[wid * ATT_R + k][c];
                s[k][0] += qv.x * kv.x;
                s[k][1] += qv.y * kv.y;
                s[k][2] += qv.z * kv.z;
                s[k][3] += qv.w * kv.w;
            }
        }
        int j = j0 + lane;
        float sc[ATT_R], corr[ATT_R];
        for (int k = 0; k < ATT_R; k++) {
            bool in = j >= lo[k] && j < hi[k];
            sc[k] = in ? (s[k][0] + s[k][1]) + (s[k][2] + s[k][3]) : -INFINITY;
            float mn = max(m[k], simd_max(sc[k]));
            corr[k] = mn != -INFINITY ? fast::exp(m[k] - mn) : 1.0f;
            float p = sc[k] != -INFINITY ? fast::exp(sc[k] - mn) : 0.0f;
            l[k] = l[k] * corr[k] + p;
            m[k] = mn;
            ps[wid][k][lane] = p;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float s0[ATT_R][2], s1[ATT_R][2];
        for (int k = 0; k < ATT_R; k++) s0[k][0] = s0[k][1] = s1[k][0] = s1[k][1] = 0.0f;
        for (int t = 0; t < 32; t += 4) {
            float2 v[4];
            for (int x = 0; x < 4; x++) v[x] = *(threadgroup const float2*)&vs[t + x][2 * lane];
            for (int k = 0; k < ATT_R; k++) {
                float4 pt = *(threadgroup const float4*)&ps[wid][k][t];
                s0[k][0] += pt.x * v[0].x + pt.z * v[2].x;
                s0[k][1] += pt.y * v[1].x + pt.w * v[3].x;
                s1[k][0] += pt.x * v[0].y + pt.z * v[2].y;
                s1[k][1] += pt.y * v[1].y + pt.w * v[3].y;
            }
        }
        for (int k = 0; k < ATT_R; k++) {
            acc0[k] = acc0[k] * corr[k] + (s0[k][0] + s0[k][1]);
            acc1[k] = acc1[k] * corr[k] + (s1[k][0] + s1[k][1]);
        }
    }
    for (int k = 0; k < ATT_R; k++) {
        int i = i0 + wid * ATT_R + k;
        float sum = simd_sum(l[k]);
        if (i >= nt) continue;
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        ulong o = (ulong)i * d + h * 64 + 2 * lane;
        st(out, o, acc0[k] * inv);
        st(out, o + 1, acc1[k] * inv);
    }
}

instantiate("attention_f32_f32", attention, float, float)
instantiate("attention_f32_f16", attention, float, half)
instantiate("attention_f16_f32", attention, half, float)
instantiate("attention_f16_f16", attention, half, half)

// out = gelu(x[:, ..inter]) * x[:, inter..], one threadgroup per token row.
template <typename T, typename TO>
kernel void geglu(device TO* out [[buffer(0)]], device const T* x [[buffer(1)]],
                  device const uint* n [[buffer(2)]], constant int& inter [[buffer(3)]],
                  uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                  uint nt [[threads_per_threadgroup]]) {
    if (r >= n[0]) return;
    ulong xr = (ulong)r * 2 * inter;
    for (int c = t; c < inter; c += nt) st(out, (ulong)r * inter + c, gelu(ld(x, xr + c)) * ld(x, xr + inter + c));
}

instantiate("geglu_f32_f32", geglu, float, float)
instantiate("geglu_f32_f16", geglu, float, half)
instantiate("geglu_f16_f32", geglu, half, float)
instantiate("geglu_f16_f16", geglu, half, half)

// h[r] += table[qtype[seq[r]]], one threadgroup per token row.
kernel void add_type(device float* h [[buffer(0)]], device const float* table [[buffer(1)]],
                     device const uint* seq [[buffer(2)]], device const uint* qtype [[buffer(3)]],
                     device const uint* n [[buffer(4)]], constant int& d [[buffer(5)]],
                     uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
    if (r >= n[0]) return;
    device const float* tb = table + (ulong)qtype[seq[r]] * d;
    for (int c = t; c < d; c += nt) h[(ulong)r * d + c] += tb[c];
}

// out[m] = h[rows[m]], one threadgroup per marker.
template <typename T>
kernel void gather(device T* out [[buffer(0)]], device const T* h [[buffer(1)]],
                   device const uint* rows [[buffer(2)]], device const uint* n [[buffer(3)]],
                   constant int& d [[buffer(4)]], uint m [[threadgroup_position_in_grid]],
                   uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    if (m >= n[2]) return;
    device const T* src = h + (ulong)rows[m] * d;
    for (int c = t; c < d; c += nt) out[(ulong)m * d + c] = src[c];
}

instantiate("gather_f32", gather, float)
instantiate("gather_f16", gather, half)

// Per sequence, its first row followed by [top1, top1 - top2, entropy / ln k, k / 255] over the
// softmax of its markers' logits, with k at least 2. One threadgroup per sequence, the statistics
// on one thread in the order the CPU takes them.
kernel void act_features(device float* out [[buffer(0)]], device const float* h [[buffer(1)]],
                         device const float* logits [[buffer(2)]], device const uint* cu [[buffer(3)]],
                         device const uint* mcu [[buffer(4)]], device const uint* n [[buffer(5)]],
                         constant int& d [[buffer(6)]], uint s [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    if (s >= n[1]) return;
    device float* o = out + (ulong)s * (d + 4);
    bool empty = cu[s + 1] == cu[s];
    device const float* src = h + (ulong)cu[s] * d;
    for (int c = t; c < d; c += nt) o[c] = empty ? 0.0f : src[c];
    if (t != 0) return;
    device const float* l = logits + mcu[s];
    int k = mcu[s + 1] - mcu[s];
    float kf = (float)max(k, 2);
    if (k == 0) {
        o[d] = 0.0f;
        o[d + 1] = 0.0f;
        o[d + 2] = 0.0f;
        o[d + 3] = kf / 255.0f;
        return;
    }
    float mx = -INFINITY;
    for (int i = 0; i < k; i++) mx = max(mx, l[i]);
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += precise::exp(l[i] - mx);
    float top1 = -INFINITY, top2 = 0.0f, ent = 0.0f;
    for (int i = 0; i < k; i++) {
        float q = precise::exp(l[i] - mx) / sum;
        ent += q * precise::log(max(q, 1e-9f));
        if (i == 0 || q > top1) {
            if (i != 0) top2 = top1;
            top1 = q;
        } else if (q > top2) {
            top2 = q;
        }
    }
    o[d] = top1;
    o[d + 1] = top1 - top2;
    o[d + 2] = -ent / precise::log(kf);
    o[d + 3] = kf / 255.0f;
}
