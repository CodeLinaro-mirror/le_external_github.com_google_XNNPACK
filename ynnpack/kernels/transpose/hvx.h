// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_HVX_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_HVX_H_

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"

namespace ynn {

template <typename ElemSizeBits>
static std::array<HVX_Vector, 2> interleave(ElemSizeBits elem_size_bits,
                                            std::array<HVX_Vector, 2> x) {
  HVX_VectorPair x01 = Q6_W_vshuff_VVR(x[1], x[0], -(elem_size_bits / 8));
  return {Q6_V_lo_W(x01), Q6_V_hi_W(x01)};
}

template <size_t M>
static std::array<HVX_Vector, M> load(
    std::array<HVX_Vector, M>, const void* a, size_t stride, size_t m,
    std::integral_constant<size_t, 16> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  std::array<HVX_Vector, M> x;
  memcpy(&x[0], a, sizeof(HVX_Vector));
  for (size_t i = 1; i < M; ++i) {
    if (i < m) {
      memcpy(&x[i], offset_bytes(a, i * stride), sizeof(HVX_Vector));
    } else {
      x[i] = Q6_V_vzero();
    }
  }
  return x;
}

template <size_t M>
static void store(std::array<HVX_Vector, M> x, void* a, size_t stride, size_t m,
                  std::integral_constant<size_t, 16> /*n_bytes*/) {
  assert(m > 0);
  assert(m <= M);
  memcpy(a, &x[0], sizeof(HVX_Vector));
  for (size_t i = 1; i < M; ++i) {
    if (i < m) {
      memcpy(offset_bytes(a, i * stride), &x[i], sizeof(HVX_Vector));
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_HVX_H_
