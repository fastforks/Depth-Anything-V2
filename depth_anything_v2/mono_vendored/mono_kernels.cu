// Custom CUDA kernels for mono_fast (compiled once: prebuilt fatbin shipped in mono_fast/prebuilt, loaded by kernels.py).
// bilerp_nhwc copied from exp 003 stereoanywhere sa_kernels.cu.
#include <cuda_fp16.h>
#include <stdint.h>

// bilinear resize NHWC fp16 (contiguous), any batch: src index = scale * dst (align_corners) or
// max(scale * (dst + 0.5) - 0.5, 0); scales computed by the caller as PyTorch does (fp32). Same arithmetic as
// upsample_bilinear2d_nhwc_out_frame.
extern "C" __global__ void bilerp_nhwc(const __half* __restrict__ src, int N, int Hs, int Ws, __half* __restrict__ dst,
                                       int Hd, int Wd, int C, float sh, float sw, int align) {
    const int CV = C / 8;
    const long long total = (long long)N * Hd * Wd * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wd);
        long long r = pix / Wd;
        int y = (int)(r % Hd), n = (int)(r / Hd);
        float fy = align ? sh * y : fmaxf(sh * (y + 0.5f) - 0.5f, 0.f);
        float fx = align ? sw * x : fmaxf(sw * (x + 0.5f) - 0.5f, 0.f);
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = y0 + (y0 < Hs - 1), x1 = x0 + (x0 < Ws - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        const __half* s = src + (long long)n * Hs * Ws * C + c0;
        uint4 a = *reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x0) * C);
        uint4 b = *reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x1) * C);
        uint4 c = *reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x0) * C);
        uint4 d = *reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x1) * C);
        const __half *a8 = reinterpret_cast<const __half*>(&a), *b8 = reinterpret_cast<const __half*>(&b);
        const __half *c8 = reinterpret_cast<const __half*>(&c), *d8 = reinterpret_cast<const __half*>(&d);
        uint4 o;
        __half* o8 = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o8[k] = __float2half(hy * (hx * __half2float(a8[k]) + lx * __half2float(b8[k])) +
                                 ly * (hx * __half2float(c8[k]) + lx * __half2float(d8[k])));
        *reinterpret_cast<uint4*>(dst + pix * C + c0) = o;
    }
}


// replicate padding (pad p on each side) of an NHWC fp16 tensor, optional ReLU, into a contiguous [N][H+2p][W+2p][C] buffer
// (C % 8 == 0). Same values as F.pad(mode="replicate") (of relu(x) when relu=1; relu and replicate commute).
extern "C" __global__ void pad_rep_nhwc(const __half* __restrict__ src, int N, int H, int W, int C, int p, int relu,
                                        __half* __restrict__ dst) {
    const int CV = C / 8;
    const int Hp = H + 2 * p, Wp = W + 2 * p;
    const long long total = (long long)N * Hp * Wp * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wp);
        long long r = pix / Wp;
        int y = (int)(r % Hp), n = (int)(r / Hp);
        int sy = min(max(y - p, 0), H - 1), sx = min(max(x - p, 0), W - 1);
        uint4 v = *reinterpret_cast<const uint4*>(src + (((long long)n * H + sy) * W + sx) * C + c0);
        if (relu) {
            __half2* h = reinterpret_cast<__half2*>(&v);
            const __half2 z = __float2half2_rn(0.f);
#pragma unroll
            for (int k = 0; k < 4; ++k) h[k] = __hmax2(h[k], z);
        }
        *reinterpret_cast<uint4*>(dst + pix * C + c0) = v;
    }
}


// reflect padding (pad p on each side, p < H, W) of an NHWC fp16 tensor into a contiguous [N][H+2p][W+2p][C] buffer
// (C % 8 == 0). Same values as F.pad(mode="reflect").
extern "C" __global__ void pad_refl_nhwc(const __half* __restrict__ src, int N, int H, int W, int C, int p,
                                         __half* __restrict__ dst) {
    const int CV = C / 8;
    const int Hp = H + 2 * p, Wp = W + 2 * p;
    const long long total = (long long)N * Hp * Wp * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wp);
        long long r = pix / Wp;
        int y = (int)(r % Hp), n = (int)(r / Hp);
        int sy = y - p, sx = x - p;
        sy = sy < 0 ? -sy : (sy >= H ? 2 * (H - 1) - sy : sy);
        sx = sx < 0 ? -sx : (sx >= W ? 2 * (W - 1) - sx : sx);
        *reinterpret_cast<uint4*>(dst + pix * C + c0) =
            *reinterpret_cast<const uint4*>(src + (((long long)n * H + sy) * W + sx) * C + c0);
    }
}


// (trtgap) separable T-tap resize of a uint8 [C, H, W] image + /255 + per-channel normalize -> fp16 [C, h, w] in one pass
// (taps / weights = mono_fast/resize.py matrices in sparse form; fp32 math). out = ((sum_i sum_j wy[y,i] wx[x,j] img) / 255 - m) / s
extern "C" __global__ void resize_norm_u8(const unsigned char* __restrict__ img, int C, int H, int W, int sc, int sy, int sx,
                                          const int* __restrict__ ix, const float* __restrict__ wx, int Tx,
                                          const int* __restrict__ iy, const float* __restrict__ wy, int Ty, int h, int w,
                                          float m0, float m1, float m2, float s0, float s1, float s2, __half* __restrict__ out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)C * h * w) return;
    int x = i % w; int y = (i / w) % h; int c = i / ((long long)w * h);
    const unsigned char* p = img + (long long)c * sc;   // strided (CHW or HWC-permuted views)
    float acc = 0.f;
    for (int a = 0; a < Ty; ++a) {
        float wyv = wy[y * Ty + a];
        if (wyv == 0.f) continue;
        const unsigned char* row = p + (long long)iy[y * Ty + a] * sy;
        float r = 0.f;
        for (int b = 0; b < Tx; ++b) r += wx[x * Tx + b] * (float)row[(long long)ix[x * Tx + b] * sx];
        acc += wyv * r;
    }
    float m = c == 0 ? m0 : (c == 1 ? m1 : m2), s = c == 0 ? s0 : (c == 1 ? s1 : s2);
    out[i] = __float2half((acc / 255.f - m) / s);
}

// (trtgap) DPT output head tail: out[p] = relu(relu(sum_c x[p, c] * w[c] + b)) in fp32 from an fp16 NHWC [P, C] tensor (C % 8 == 0)
extern "C" __global__ void head_dot_relu(const __half* __restrict__ x, long long P, int C, const float* __restrict__ w, float b,
                                         float* __restrict__ out) {
    long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= P) return;
    const uint4* r = reinterpret_cast<const uint4*>(x + p * C);
    float acc = b;
    for (int c8 = 0; c8 < C / 8; ++c8) {
        uint4 q = __ldg(r + c8);
        const __half* h = reinterpret_cast<const __half*>(&q);
#pragma unroll
        for (int t = 0; t < 8; ++t) acc += __half2float(h[t]) * w[c8 * 8 + t];
    }
    out[p] = fmaxf(acc, 0.f);
}
