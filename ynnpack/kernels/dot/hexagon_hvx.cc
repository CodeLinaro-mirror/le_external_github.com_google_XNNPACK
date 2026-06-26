// Copyright 2026 Google LLC
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/kernels/dot/dot.h"

#ifdef YNN_ARCH_HEXAGON_HVX

#include "ynnpack/base/simd/hexagon_hvx.h"

namespace ynn {
namespace {

using simd::f32x32;

constexpr size_t kVec = 32;

template <int MR, int NV>
YNN_ALWAYS_INLINE void hvx_dot_fp32_block(
    size_t M, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const float* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const float* B,
    size_t C_in_stride_m, const float* C_in, size_t C_out_stride_m,
    float* C_out, size_t n_store) {
  f32x32 acc[MR][NV];
  for (int i = 0; i < MR; ++i) {
    for (int v = 0; v < NV; ++v) {
      acc[i][v] = f32x32(0.0f);
    }
  }

  const float* A_k3 = A;
  const float* B_k3 = B;
  size_t k3 = K3;
  do {
    const float* A_k2 = A_k3;
    const float* B_k2 = B_k3;
    size_t k2 = K2;
    do {
      const float* A_k1 = A_k2;
      const float* B_k1 = B_k2;
      size_t k1 = K1;
      do {
        f32x32 b[NV];
        for (int v = 0; v < NV; ++v) {
          b[v] = simd::load(B_k1 + v * kVec, f32x32::N);
        }
        for (int i = 0; i < MR; ++i) {
          const size_t ri = static_cast<size_t>(i) < M ? i : M - 1;
          const f32x32 a(*offset_bytes(A_k1, ri * A_stride_m));
          for (int v = 0; v < NV; ++v) {
            acc[i][v] = acc[i][v] + a * b[v];
          }
        }
        A_k1 = offset_bytes(A_k1, sizeof(float));
        B_k1 = offset_bytes(B_k1, B_stride_k1);
      } while (--k1);
      A_k2 = offset_bytes(A_k2, A_stride_k2);
      B_k2 = offset_bytes(B_k2, B_stride_k2);
    } while (--k2);
    A_k3 = offset_bytes(A_k3, A_stride_k3);
    B_k3 = offset_bytes(B_k3, B_stride_k3);
  } while (--k3);

  for (size_t i = 0; i < M; ++i) {
    float* c_out_i = offset_bytes(C_out, i * C_out_stride_m);
    const float* c_in_i =
        C_in ? offset_bytes(C_in, i * C_in_stride_m) : nullptr;
    for (int v = 0; v < NV; ++v) {
      const size_t n = (v == NV - 1) ? n_store : kVec;
      f32x32 c = acc[i][v];
      if (c_in_i) {
        const f32x32 in = n == kVec
                              ? simd::load(c_in_i + v * kVec, f32x32::N)
                              : simd::load(c_in_i + v * kVec, n, f32x32(0.0f));
        c = c + in;
      }
      if (n == kVec) {
        simd::store(c_out_i + v * kVec, c);
      } else {
        simd::store(c_out_i + v * kVec, c, n);
      }
    }
  }
}

template <int MR, int NR>
void hvx_dot_fp32(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                  size_t A_stride_m, size_t A_stride_k3, size_t A_stride_k2,
                  const void* A_in, size_t B_stride_k3, size_t B_stride_k2,
                  size_t B_stride_k1, const void* B_in, size_t C_in_stride_m,
                  const void* C_in_in, size_t C_out_stride_m, void* C_out_in) {
  assert(M > 0);
  assert(M <= static_cast<size_t>(MR));
  assert(N > 0);
  assert(K1 > 0 && K2 > 0 && K3 > 0);

  const float* A = static_cast<const float*>(A_in);
  const float* B = static_cast<const float*>(B_in);
  const float* C_in = static_cast<const float*>(C_in_in);
  float* C_out = static_cast<float*>(C_out_in);

  while (N >= NR * kVec) {
    hvx_dot_fp32_block<MR, NR>(M, K3, K2, K1, A_stride_m, A_stride_k3,
                               A_stride_k2, A, B_stride_k3, B_stride_k2,
                               B_stride_k1, B, C_in_stride_m, C_in,
                               C_out_stride_m, C_out, kVec);
    N -= NR * kVec;
    B += NR * kVec;
    if (C_in) C_in += NR * kVec;
    C_out += NR * kVec;
  }
  if (NR > 1) {
    while (N >= kVec) {
      hvx_dot_fp32_block<MR, 1>(M, K3, K2, K1, A_stride_m, A_stride_k3,
                                A_stride_k2, A, B_stride_k3, B_stride_k2,
                                B_stride_k1, B, C_in_stride_m, C_in,
                                C_out_stride_m, C_out, kVec);
      N -= kVec;
      B += kVec;
      if (C_in) C_in += kVec;
      C_out += kVec;
    }
  }
  if (N > 0) {
    hvx_dot_fp32_block<MR, 1>(M, K3, K2, K1, A_stride_m, A_stride_k3,
                              A_stride_k2, A, B_stride_k3, B_stride_k2,
                              B_stride_k1, B, C_in_stride_m, C_in,
                              C_out_stride_m, C_out, N);
  }
}

}  // namespace

#define YNN_HVX_DOT_FP32(NAME, MR, NR)                                        \
  void NAME(size_t m, size_t n, size_t k3, size_t k2, size_t k1,              \
            size_t a_stride_m, size_t a_stride_k3, size_t a_stride_k2,        \
            const void* a, size_t b_stride_k3, size_t b_stride_k2,            \
            size_t b_stride_k1, const void* b, size_t c_in_stride_m,          \
            const void* c_in, size_t c_out_stride_m, void* c_out) {           \
    hvx_dot_fp32<MR, NR>(m, n, k3, k2, k1, a_stride_m, a_stride_k3,           \
                         a_stride_k2, a, b_stride_k3, b_stride_k2,            \
                         b_stride_k1, b, c_in_stride_m, c_in, c_out_stride_m, \
                         c_out);                                              \
  }

YNN_HVX_DOT_FP32(dot_fp32_1x32x1_1x32x1_hvx, 1, 1)
YNN_HVX_DOT_FP32(dot_fp32_1x64x1_1x32x1_hvx, 1, 2)
YNN_HVX_DOT_FP32(dot_fp32_1x128x1_1x32x1_hvx, 1, 4)
YNN_HVX_DOT_FP32(dot_fp32_2x64x1_1x32x1_hvx, 2, 2)
YNN_HVX_DOT_FP32(dot_fp32_4x32x1_1x32x1_hvx, 4, 1)
YNN_HVX_DOT_FP32(dot_fp32_4x64x1_1x32x1_hvx, 4, 2)
YNN_HVX_DOT_FP32(dot_fp32_4x128x1_1x32x1_hvx, 4, 4)
YNN_HVX_DOT_FP32(dot_fp32_6x64x1_1x32x1_hvx, 6, 2)
YNN_HVX_DOT_FP32(dot_fp32_8x32x1_1x32x1_hvx, 8, 1)
YNN_HVX_DOT_FP32(dot_fp32_8x64x1_1x32x1_hvx, 8, 2)

#undef YNN_HVX_DOT_FP32

}  // namespace ynn

#endif  // YNN_ARCH_HEXAGON_HVX
