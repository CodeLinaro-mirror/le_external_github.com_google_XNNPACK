// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_SUBGRAPH_TEST_FAKE_DYNAMIC_QUANTIZE_H_
#define XNNPACK_TEST_SUBGRAPH_TEST_FAKE_DYNAMIC_QUANTIZE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/quantization.h"

namespace xnnpack {

// Dynamic quantization looks a lot like a float input/output, but the error is
// hard to quantify and test well. Rather than do that, we can just generate
// input data that has (close to) zero error when dynamically quantized, which
// makes it easier to test.
template <typename Data>
void FakeDynamicQuantize(
    Tensor<Data> input, float qmin, float qmax,
    const struct xnn_qd8_quantization_params* quantization_params) {
  for (auto& i : input) {
    const int32_t q =
        std::min(std::max(std::round(i * quantization_params->inv_scale) +
                              quantization_params->zero_point,
                          qmin),
                 qmax);
    i = (q - quantization_params->zero_point) / quantization_params->inv_scale;
  }
}

template <typename Data>
void FakeDynamicQuantize(Tensor<Data> input, xnn_datatype datatype) {
  auto minmax = std::minmax_element(input.begin(), input.end());
  const float rmin = *minmax.first;
  const float rmax = *minmax.second;
  if (datatype == xnn_datatype_qdint8) {
    struct xnn_qd8_quantization_params quantization_params =
        xnn_qd8_asymmetric_quantization_params(rmin, rmax);
    FakeDynamicQuantize(input, INT8_MIN, INT8_MAX, &quantization_params);
  } else if (datatype == xnn_datatype_qduint8) {
    struct xnn_qd8_quantization_params quantization_params =
        xnn_qdu8_asymmetric_quantization_params(rmin, rmax);
    FakeDynamicQuantize(input, 0, UINT8_MAX, &quantization_params);
  } else {
    XNN_UNREACHABLE;
  }
}

template <typename Data>
void FakeDynamicQuantize(const Tensor<quantized<Data>>& input, xnn_datatype) {}

}  // namespace xnnpack

#endif  // XNNPACK_TEST_SUBGRAPH_TEST_FAKE_DYNAMIC_QUANTIZE_H_
