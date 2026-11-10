// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

TEST(DotSumTest, SplitF32ToBf16) {
  // We will test splitting into 1, 2, and 3 values.
  // We create a subgraph with 1 input (fp32) and 3 outputs (bf16).
  subgraph_ptr subgraph = create_subgraph(4, 0);  // 1 input + 3 outputs
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_ids[3] = {1, 2, 3};
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(
        ynn_define_tensor(subgraph.get(), ynn_type_bf16, 1, nullptr, nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_ids[i]),
        ynn_status_success);
  }

  // We call define_convert_f32_to_bf16_sum to split x into y_ids.
  // We pass pre-defined external outputs.
  ASSERT_EQ(define_convert_f32_to_bf16_sum(subgraph.get(), x_id, 3, y_ids, 0),
            ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  size_t shape[] = {4};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 1, shape),
            ynn_status_success);

  // Input data: some floats that require different levels of precision.
  // e.g. pi, e, some small values, etc.
  std::vector<float> x_data = {
      3.14159265f,  // needs many bits
      1.0f,         // exact in bf16
      0.1f,         // not exact
      0.0001f       // small
  };
  std::vector<bfloat16> y0_data(4, bfloat16(0.0f));
  std::vector<bfloat16> y1_data(4, bfloat16(0.0f));
  std::vector<bfloat16> y2_data(4, bfloat16(0.0f));

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x_data.data()),
            ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), y_ids[0], y0_data.data()),
      ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), y_ids[1], y1_data.data()),
      ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), y_ids[2], y2_data.data()),
      ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  for (size_t i = 0; i < x_data.size(); ++i) {
    float x = x_data[i];
    float v0 = static_cast<float>(y0_data[i]);
    float v1 = static_cast<float>(y1_data[i]);
    float v2 = static_cast<float>(y2_data[i]);

    float rec1 = v0;
    float rec2 = v0 + v1;
    float rec3 = v0 + v1 + v2;

    // Relative errors
    float err1 = std::abs(x - rec1) / std::abs(x);
    float err2 = std::abs(x - rec2) / std::abs(x);
    float err3 = std::abs(x - rec3) / std::abs(x);

    // 1 split (bf16) has 8 bits of precision (relative error ~ 1/256 = 0.0039)
    // 2 splits has ~ 16 bits of precision (relative error ~ 1/65536 = 1.5e-5)
    // 3 splits has ~ 24 bits of precision (relative error ~ 1/16M = 6e-8)

    // For x = 1.0, it should be exact in first split, others 0.
    if (x == 1.0f) {
      EXPECT_EQ(v0, 1.0f);
      EXPECT_EQ(v1, 0.0f);
      EXPECT_EQ(v2, 0.0f);
    } else {
      EXPECT_LT(err1, 0.01f);
      if (err1 > 0.0f) {  // if it wasn't already exact
        EXPECT_LT(err2, err1);
        EXPECT_LT(err2, 0.0001f);
        if (err2 > 0.0f) {
          EXPECT_LT(err3, err2);
          // For float, 3 splits should be very close to exact.
          EXPECT_LT(err3, 1e-6f);
        }
      }
    }
  }
}

TEST(DotSumTest, DefineDotSum) {
  // Test composite dot with various split sizes.
  // We will test 2x2 split (which is what bf16x3 rewrite conceptually does,
  // but here we compute all 4 products).
  // Actually, let's test 2x2.
  // Input A (fp32) -> split into A0, A1 (bf16)
  // Input B (fp32) -> split into B0, B1 (bf16)
  // We will run:
  // 1. Reference float dot: A * B
  // 2. Composite dot: composite_dot({A0, A1}, {B0, B1})
  // Compare results.

  subgraph_ptr subgraph = create_subgraph(3, 0);  // A (in), B (in), C (out)
  ASSERT_NE(subgraph, nullptr);

  uint32_t a_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);

  uint32_t b_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_id),
            ynn_status_success);

  uint32_t c_id = 2;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &c_id),
            ynn_status_success);

  // Define splits in subgraph
  uint32_t a_values[2] = {YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID};
  ASSERT_EQ(
      define_convert_f32_to_bf16_sum(subgraph.get(), a_id, 2, a_values, 0),
      ynn_status_success);

  uint32_t b_values[2] = {YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID};
  ASSERT_EQ(
      define_convert_f32_to_bf16_sum(subgraph.get(), b_id, 2, b_values, 0),
      ynn_status_success);

  // Call dot sum
  // We use num_k_dims = 1
  ASSERT_EQ(define_dot_sum(subgraph.get(), 1, 2, a_values, 2, b_values,
                           YNN_INVALID_VALUE_ID, c_id, 0),
            ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  size_t a_shape[] = {2, 4};  // M=2, K=4
  size_t b_shape[] = {4, 3};  // K=4, N=3

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, b_shape),
            ynn_status_success);

  std::vector<float> a_data = {1.1f,  2.2f,  3.3f,  4.4f,
                               -1.1f, 0.55f, 1.55f, -2.55f};
  std::vector<float> b_data = {0.55f, 1.1f,  2.2f, -0.55f, 0.0f,  1.1f,
                               1.1f,  -1.1f, 0.0f, 2.2f,   1.55f, -1.1f};
  std::vector<float> c_data(6, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id, a_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), b_id, b_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), c_id, c_data.data()),
            ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  // Compute reference
  std::vector<float> expected_c(6, 0.0f);
  for (size_t m = 0; m < 2; ++m) {
    for (size_t n = 0; n < 3; ++n) {
      float sum = 0.0f;
      for (size_t k = 0; k < 4; ++k) {
        sum += a_data[m * 4 + k] * b_data[k * 3 + n];
      }
      expected_c[m * 3 + n] = sum;
    }
  }

  // Compare. Since we used 2 splits (which is 4 products total),
  // the precision should be very high (approx 16 bits of precision, relative
  // error < 1.5e-5).
  for (size_t i = 0; i < 6; ++i) {
    float ref = expected_c[i];
    float val = c_data[i];
    float err = std::abs(ref - val);
    if (std::abs(ref) > 1e-5f) {
      err /= std::abs(ref);
      EXPECT_LT(err, 1e-4f)
          << "at index " << i << ", ref=" << ref << ", val=" << val;
    } else {
      EXPECT_LT(err, 1e-4f)
          << "at index " << i << ", ref=" << ref << ", val=" << val;
    }
  }
}

TEST(DotTest, BlockwiseDot) {
  subgraph_ptr subgraph = create_subgraph(3, 0);
  ASSERT_NE(subgraph, nullptr);

  const size_t M = 2;
  const size_t K = 4;
  const size_t N = 2;
  const size_t block_size = 2;
  const size_t num_blocks = K / block_size;

  uint32_t a_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);

  // B is static: shape [K, N] = [4, 2], type int8
  const int8_t b_data[K * N] = {
      1, 2, 3, 4, 5, 6, 7, 8,
  };
  const size_t b_dims[2] = {K, N};
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, b_dims, b_data,
                              YNN_VALUE_FLAG_COPY_DATA, &b_id),
            ynn_status_success);

  // B scales: shape [N, num_blocks] = [2, 2], type fp32
  const float b_scale_data[N * num_blocks] = {
      0.5f,
      0.25f,
      2.0f,
      1.0f,
  };
  const size_t b_scale_dims[2] = {N, num_blocks};
  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, b_scale_dims,
                        b_scale_data, YNN_VALUE_FLAG_COPY_DATA, &b_scale_id),
      ynn_status_success);

  uint32_t out_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id),
            ynn_status_success);

  int32_t reduce_axis = -1;
  uint32_t min_max_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_reduce(subgraph.get(), ynn_reduce_min_max, 1,
                              &reduce_axis, a_id, YNN_INVALID_VALUE_ID,
                              &min_max_id, YNN_NODE_FLAG_KEEP_DIMS),
            ynn_status_success);

  uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      ynn_define_dynamic_quantization(subgraph.get(), min_max_id, ynn_type_int8,
                                      &a_zp_id, &a_scale_id, 0),
      ynn_status_success);

  uint32_t quantized_a_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_quantize(subgraph.get(), a_id, ynn_type_int8, a_zp_id,
                                a_scale_id, &quantized_a_id, 0),
            ynn_status_success);

  ASSERT_EQ(
      define_blockwise_dot(subgraph.get(), quantized_a_id, a_zp_id, a_scale_id,
                           b_id, YNN_INVALID_VALUE_ID, b_scale_id, block_size,
                           YNN_INVALID_VALUE_ID, ynn_type_fp32, out_id, 0),
      ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  const size_t a_shape[2] = {M, K};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_shape),
            ynn_status_success);

  const std::vector<float> a_data = {
      1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -2.0f,
  };
  std::vector<float> out_data(M * N, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id,
                                        const_cast<float*>(a_data.data())),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), out_id, out_data.data()),
            ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  // Reference computation:
  // For each m, n:
  // sum over blocks b: (sum over k in block: a_quant[m, k] * b[k, n]) *
  // b_scale[n, b] * a_scale[m] Because a is dynamically quantized, let's verify
  // output is non-zero and finite
  for (size_t i = 0; i < M * N; ++i) {
    EXPECT_FALSE(std::isnan(out_data[i]));
    EXPECT_FALSE(std::isinf(out_data[i]));
  }
}

TEST(DotTest, BlockwiseDotWithZeroPoints) {
  for (bool has_a_zp : {false, true}) {
    for (bool has_b_zp : {false, true}) {
      subgraph_ptr subgraph = create_subgraph(2, 0);
      ASSERT_NE(subgraph, nullptr);

      const size_t M = 2;
      const size_t K = 4;
      const size_t N = 2;
      const size_t block_size = 2;
      const size_t num_blocks = K / block_size;

      // A: [M, K] = [2, 4], type int8
      const int8_t a_data[M * K] = {1, 2, 3, 4, -1, 0, 2, -2};
      uint32_t a_id = 0;
      ASSERT_EQ(
          ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, nullptr, nullptr,
                            YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
          ynn_status_success);

      uint32_t out_id = 1;
      ASSERT_EQ(
          ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr, nullptr,
                            YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id),
          ynn_status_success);

      // A zero point: [M, 1] = [2, 1]
      const int8_t a_zp_data[M] = {1, -1};
      const size_t a_zp_dims[2] = {M, 1};
      uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
      if (has_a_zp) {
        ASSERT_EQ(
            ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, a_zp_dims,
                              a_zp_data, YNN_VALUE_FLAG_COPY_DATA, &a_zp_id),
            ynn_status_success);
      }

      // A scale: [M, 1] = [2, 1]
      const float a_scale_data[M] = {2.0f, 0.5f};
      const size_t a_scale_dims[2] = {M, 1};
      uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
      ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2,
                                  a_scale_dims, a_scale_data,
                                  YNN_VALUE_FLAG_COPY_DATA, &a_scale_id),
                ynn_status_success);

      // B: [K, N] = [4, 2], type int8
      const int8_t b_data[K * N] = {
          1, 2, 3, 4, 5, 6, 7, 8,
      };
      const size_t b_dims[2] = {K, N};
      uint32_t b_id = YNN_INVALID_VALUE_ID;
      ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, b_dims,
                                  b_data, YNN_VALUE_FLAG_COPY_DATA, &b_id),
                ynn_status_success);

      // B zero point: [N, num_blocks] = [2, 2]
      const int8_t b_zp_data[N * num_blocks] = {1, 2, -1, 3};
      const size_t b_zp_dims[2] = {N, num_blocks};
      uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
      if (has_b_zp) {
        ASSERT_EQ(
            ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, b_zp_dims,
                              b_zp_data, YNN_VALUE_FLAG_COPY_DATA, &b_zp_id),
            ynn_status_success);
      }

      // B scale: [N, num_blocks] = [2, 2]
      const float b_scale_data[N * num_blocks] = {0.5f, 2.0f, 0.25f, 1.0f};
      const size_t b_scale_dims[2] = {N, num_blocks};
      uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
      ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2,
                                  b_scale_dims, b_scale_data,
                                  YNN_VALUE_FLAG_COPY_DATA, &b_scale_id),
                ynn_status_success);

      ASSERT_EQ(
          define_blockwise_dot(subgraph.get(), a_id, a_zp_id, a_scale_id, b_id,
                               b_zp_id, b_scale_id, block_size,
                               YNN_INVALID_VALUE_ID, ynn_type_fp32, out_id, 0),
          ynn_status_success);

      ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
                ynn_status_success);

      runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
      ASSERT_NE(runtime, nullptr);

      const size_t a_shape[2] = {M, K};
      ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_shape),
                ynn_status_success);
      std::vector<float> out_data(M * N, 0.0f);
      ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id,
                                            const_cast<int8_t*>(a_data)),
                ynn_status_success);
      ASSERT_EQ(
          ynn_set_external_value_data(runtime.get(), out_id, out_data.data()),
          ynn_status_success);

      ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
      ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

      // Compute expected reference output
      for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
          float sum = 0.0f;
          for (size_t b = 0; b < num_blocks; ++b) {
            float block_sum = 0.0f;
            float b_zp = has_b_zp ? b_zp_data[n * num_blocks + b] : 0.0f;
            float a_zp = has_a_zp ? a_zp_data[m] : 0.0f;
            for (size_t k_in_block = 0; k_in_block < block_size; ++k_in_block) {
              size_t k = b * block_size + k_in_block;
              float a_val = static_cast<float>(a_data[m * K + k]) - a_zp;
              float b_val = static_cast<float>(b_data[k * N + n]) - b_zp;
              block_sum += a_val * b_val;
            }
            float b_scale = b_scale_data[n * num_blocks + b];
            sum += block_sum * b_scale;
          }
          float expected = sum * a_scale_data[m];
          EXPECT_NEAR(out_data[m * N + n], expected, 1e-4f)
              << "Mismatch at (" << m << ", " << n
              << ") with has_a_zp=" << has_a_zp << ", has_b_zp=" << has_b_zp;
        }
      }
    }
  }
}

TEST(DotTest, BlockwiseDotWithFloatZeroPoints) {
  for (ynn_type b_type : {ynn_type_int8, ynn_type_int2}) {
    for (bool has_a_zp : {false, true}) {
      subgraph_ptr subgraph = create_subgraph(2, 0);
      ASSERT_NE(subgraph, nullptr);

      const size_t M = 2;
      const size_t K = 4;
      const size_t N = 4;
      const size_t block_size = 2;
      const size_t num_blocks = K / block_size;

      // A: [M, K] = [2, 4], type int8
      const int8_t a_data[M * K] = {1, 2, 3, 4, -1, 0, 2, -2};
      uint32_t a_id = 0;
      ASSERT_EQ(
          ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, nullptr, nullptr,
                            YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
          ynn_status_success);

      uint32_t out_id = 1;
      ASSERT_EQ(
          ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr, nullptr,
                            YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id),
          ynn_status_success);

      // A zero point: [M, 1] = [2, 1]
      const int8_t a_zp_data[M] = {1, -2};
      const size_t a_zp_dims[2] = {M, 1};
      uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
      if (has_a_zp) {
        ASSERT_EQ(
            ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, a_zp_dims,
                              a_zp_data, YNN_VALUE_FLAG_COPY_DATA, &a_zp_id),
            ynn_status_success);
      }

      // A scale: [M, 1] = [2, 1]
      const float a_scale_data[M] = {2.0f, 0.5f};
      const size_t a_scale_dims[2] = {M, 1};
      uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
      ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2,
                                  a_scale_dims, a_scale_data,
                                  YNN_VALUE_FLAG_COPY_DATA, &a_scale_id),
                ynn_status_success);

      // B: [K, N] = [4, 4] in [-2, 1] so values fit both int8 and int2.
      const int8_t b_vals[K * N] = {
          -2, -1, 0,  1,   // k = 0
          1,  -2, -1, 0,   // k = 1
          0,  1,  -2, -1,  // k = 2
          -1, 0,  1,  -2,  // k = 3
      };
      const size_t b_dims[2] = {K, N};
      uint32_t b_id = YNN_INVALID_VALUE_ID;
      if (b_type == ynn_type_int8) {
        ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, b_dims,
                                    b_vals, YNN_VALUE_FLAG_COPY_DATA, &b_id),
                  ynn_status_success);
      } else {
        uint8_t packed_b[K * N / 4] = {};
        for (size_t i = 0; i < K * N; ++i) {
          packed_b[i / 4] |= (static_cast<uint8_t>(b_vals[i]) & 0x03)
                             << ((i % 4) * 2);
        }
        ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int2, 2, b_dims,
                                    packed_b, YNN_VALUE_FLAG_COPY_DATA, &b_id),
                  ynn_status_success);
      }

      // Fractional B zero point: [N, num_blocks] = [4, 2]
      const float b_zp_data[N * num_blocks] = {
          -0.5f, 0.25f, 1.5f, -0.75f, 0.5f, -1.25f, -0.5f, 0.75f,
      };
      const size_t b_zp_dims[2] = {N, num_blocks};
      uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
      ASSERT_EQ(
          ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, b_zp_dims,
                            b_zp_data, YNN_VALUE_FLAG_COPY_DATA, &b_zp_id),
          ynn_status_success);

      // B scale: [N, num_blocks] = [4, 2]
      const float b_scale_data[N * num_blocks] = {
          0.5f, 2.0f, 0.25f, 1.0f, 1.5f, 0.75f, 0.5f, 1.25f,
      };
      const size_t b_scale_dims[2] = {N, num_blocks};
      uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
      ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2,
                                  b_scale_dims, b_scale_data,
                                  YNN_VALUE_FLAG_COPY_DATA, &b_scale_id),
                ynn_status_success);

      ASSERT_EQ(
          define_blockwise_dot(subgraph.get(), a_id, a_zp_id, a_scale_id, b_id,
                               b_zp_id, b_scale_id, block_size,
                               YNN_INVALID_VALUE_ID, ynn_type_fp32, out_id, 0),
          ynn_status_success);

      ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
                ynn_status_success);

      runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
      ASSERT_NE(runtime, nullptr);

      const size_t a_shape[2] = {M, K};
      ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_shape),
                ynn_status_success);
      std::vector<float> out_data(M * N, 0.0f);
      ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id,
                                            const_cast<int8_t*>(a_data)),
                ynn_status_success);
      ASSERT_EQ(
          ynn_set_external_value_data(runtime.get(), out_id, out_data.data()),
          ynn_status_success);

      ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
      ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

      for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
          float sum = 0.0f;
          for (size_t b = 0; b < num_blocks; ++b) {
            float block_sum = 0.0f;
            float b_zp = b_zp_data[n * num_blocks + b];
            float a_zp = has_a_zp ? a_zp_data[m] : 0.0f;
            for (size_t k_in_block = 0; k_in_block < block_size; ++k_in_block) {
              size_t k = b * block_size + k_in_block;
              float a_val = static_cast<float>(a_data[m * K + k]) - a_zp;
              float b_val = static_cast<float>(b_vals[k * N + n]) - b_zp;
              block_sum += a_val * b_val;
            }
            sum += block_sum * b_scale_data[n * num_blocks + b];
          }
          float expected = sum * a_scale_data[m];
          EXPECT_NEAR(out_data[m * N + n], expected, 1e-4f)
              << "Mismatch at (" << m << ", " << n
              << ") with has_a_zp=" << has_a_zp << ", b_type=" << b_type;
        }
      }
    }
  }
}

TEST(DotTest, BlockwiseDotInvalidParameters) {
  subgraph_ptr subgraph = create_subgraph(5, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t a_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);

  uint32_t b_id = 1;
  const size_t b_dims[2] = {4, 2};
  const int8_t b_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, b_dims, b_data,
                              YNN_VALUE_FLAG_COPY_DATA, &b_id),
            ynn_status_success);

  uint32_t b_scale_id = 2;
  const size_t b_scale_dims[2] = {2, 2};
  const float b_scale_data[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, b_scale_dims,
                        b_scale_data, YNN_VALUE_FLAG_COPY_DATA, &b_scale_id),
      ynn_status_success);

  uint32_t out_id = YNN_INVALID_VALUE_ID;

  // block_size == 0
  EXPECT_EQ(define_blockwise_dot(subgraph.get(), a_id, YNN_INVALID_VALUE_ID,
                                 YNN_INVALID_VALUE_ID, b_id,
                                 YNN_INVALID_VALUE_ID, b_scale_id,
                                 /*block_size=*/0, YNN_INVALID_VALUE_ID,
                                 ynn_type_fp32, out_id, 0),
            ynn_status_invalid_parameter);
  EXPECT_EQ(out_id, YNN_INVALID_VALUE_ID);

  uint32_t fp32_a_id = 3;
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_INPUT, &fp32_a_id),
      ynn_status_success);
  uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
  const int32_t b_zp_data[4] = {1, 2, 3, 4};
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int32, 2, b_scale_dims,
                              b_zp_data, YNN_VALUE_FLAG_COPY_DATA, &b_zp_id),
            ynn_status_success);
  for (uint32_t zp : {uint32_t{YNN_INVALID_VALUE_ID}, b_zp_id}) {
    EXPECT_EQ(define_blockwise_dot(
                  subgraph.get(), fp32_a_id, YNN_INVALID_VALUE_ID,
                  YNN_INVALID_VALUE_ID, b_id, zp, b_scale_id, /*block_size=*/2,
                  YNN_INVALID_VALUE_ID, ynn_type_fp32, out_id, 0),
              ynn_status_unsupported_parameter);
    EXPECT_EQ(out_id, YNN_INVALID_VALUE_ID);
  }

  uint32_t rg_id = 4;
  const size_t rg_dims[2] = {0, 1};
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int32, 2, rg_dims,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &rg_id),
            ynn_status_success);
  const size_t ragged_b_dims[3] = {2, 4, 2};
  const int8_t ragged_b_data[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                    1, 2, 3, 4, 5, 6, 7, 8};
  uint32_t ragged_b_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_int8, 3, ragged_b_dims,
                        ragged_b_data, YNN_VALUE_FLAG_COPY_DATA, &ragged_b_id),
      ynn_status_success);
  const size_t ragged_param_dims[3] = {2, 2, 2};
  const float ragged_scale_data[8] = {1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f};
  uint32_t ragged_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 3,
                              ragged_param_dims, ragged_scale_data,
                              YNN_VALUE_FLAG_COPY_DATA, &ragged_scale_id),
            ynn_status_success);
  EXPECT_EQ(
      define_blockwise_dot(
          subgraph.get(), fp32_a_id, YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID,
          ragged_b_id, YNN_INVALID_VALUE_ID, ragged_scale_id, /*block_size=*/2,
          YNN_INVALID_VALUE_ID, ynn_type_fp32, out_id, 0, rg_id),
      ynn_status_unsupported_parameter);
  EXPECT_EQ(out_id, YNN_INVALID_VALUE_ID);
}

struct BlockwiseRaggedCase {
  size_t batch = 0;      // 0: rank-2 [M, K]; >0: rank-3 [batch, M, K].
  size_t g = 3;          // Group count of B [G, K, N].
  size_t b_param_g = 0;  // 0: same as `g`; 1: broadcast [1, N, num_blocks].
  size_t m = 4;
  size_t k = 32;
  size_t n = 16;
  size_t block_size = 16;
  bool has_a_zp = true;
  bool has_a_scale = true;
  bool has_b_zp = true;
  bool has_bias = false;
  bool static_m = false;
  bool dynamic_rg = false;
  bool reshape_twice = false;
  std::vector<int32_t> rg;
};

void RunBlockwiseRaggedCase(const BlockwiseRaggedCase& tc) {
  const size_t rank = tc.batch == 0 ? 2 : 3;
  const size_t batches = tc.batch == 0 ? 1 : tc.batch;
  const size_t param_g = tc.b_param_g == 0 ? tc.g : tc.b_param_g;
  const size_t num_blocks = tc.k / tc.block_size;

  const uint32_t num_external =
      3 + (tc.has_a_zp ? 1 : 0) + (tc.has_a_scale ? 1 : 0);
  subgraph_ptr subgraph = create_subgraph(num_external, 0);
  ASSERT_NE(subgraph, nullptr);

  const size_t a_dims_2d[2] = {tc.static_m ? tc.m : 0, tc.static_m ? tc.k : 0};
  const size_t a_dims_3d[3] = {tc.static_m ? tc.batch : 0,
                               tc.static_m ? tc.m : 0, tc.static_m ? tc.k : 0};
  const size_t col_dims_2d[2] = {tc.static_m ? tc.m : 0, 1};
  const size_t col_dims_3d[3] = {tc.static_m ? tc.batch : 0,
                                 tc.static_m ? tc.m : 0, 1};
  const size_t* a_def_dims =
      tc.static_m ? (rank == 2 ? a_dims_2d : a_dims_3d) : nullptr;
  const size_t* col_def_dims = rank == 2 ? col_dims_2d : col_dims_3d;

  uint32_t next_id = 0;
  uint32_t a_id = next_id++;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, rank, a_def_dims,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);
  uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
  if (tc.has_a_zp) {
    a_zp_id = next_id++;
    ASSERT_EQ(
        ynn_define_tensor(subgraph.get(), ynn_type_int32, rank, col_def_dims,
                          nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_zp_id),
        ynn_status_success);
  }
  uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
  if (tc.has_a_scale) {
    a_scale_id = next_id++;
    ASSERT_EQ(
        ynn_define_tensor(subgraph.get(), ynn_type_fp32, rank, col_def_dims,
                          nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_scale_id),
        ynn_status_success);
  }
  uint32_t rg_id = next_id++;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int32, rank,
                              tc.dynamic_rg ? nullptr : col_def_dims, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &rg_id),
            ynn_status_success);
  uint32_t out_id = next_id++;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, rank, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id),
            ynn_status_success);

  std::vector<int8_t> b_data(tc.g * tc.k * tc.n);
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<int8_t>(static_cast<int>(i % 15) - 7);
  }
  const size_t b_dims[3] = {tc.g, tc.k, tc.n};
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 3, b_dims,
                              b_data.data(), YNN_VALUE_FLAG_COPY_DATA, &b_id),
            ynn_status_success);

  std::vector<int32_t> b_zp_data(param_g * tc.n * num_blocks);
  std::vector<float> b_scale_data(param_g * tc.n * num_blocks);
  for (size_t i = 0; i < b_scale_data.size(); ++i) {
    b_zp_data[i] = static_cast<int32_t>(i % 5) - 2;
    b_scale_data[i] = 0.125f * static_cast<float>((i % 4) + 1);
  }
  const size_t b_param_dims[3] = {param_g, tc.n, num_blocks};
  uint32_t b_zp_id = YNN_INVALID_VALUE_ID;
  if (tc.has_b_zp) {
    ASSERT_EQ(
        ynn_define_tensor(subgraph.get(), ynn_type_int32, 3, b_param_dims,
                          b_zp_data.data(), YNN_VALUE_FLAG_COPY_DATA, &b_zp_id),
        ynn_status_success);
  }
  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 3, b_param_dims,
                              b_scale_data.data(), YNN_VALUE_FLAG_COPY_DATA,
                              &b_scale_id),
            ynn_status_success);

  std::vector<float> bias_data(tc.n);
  for (size_t n = 0; n < tc.n; ++n) {
    bias_data[n] = 0.25f * static_cast<float>(static_cast<int>(n) - 3);
  }
  uint32_t bias_id = YNN_INVALID_VALUE_ID;
  if (tc.has_bias) {
    const size_t bias_dims[1] = {tc.n};
    ASSERT_EQ(
        ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, bias_dims,
                          bias_data.data(), YNN_VALUE_FLAG_COPY_DATA, &bias_id),
        ynn_status_success);
  }

  ASSERT_EQ(define_blockwise_dot(subgraph.get(), a_id, a_zp_id, a_scale_id,
                                 b_id, b_zp_id, b_scale_id, tc.block_size,
                                 bias_id, ynn_type_fp32, out_id, 0, rg_id),
            ynn_status_success);
  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  const int num_runs = (tc.reshape_twice && !tc.static_m) ? 2 : 1;
  for (int run = 0; run < num_runs; ++run) {
    const size_t cur_m = run == 0 ? tc.m : tc.m + 2;
    const size_t total_rows = batches * cur_m;
    const size_t a_shape_2d[2] = {cur_m, tc.k};
    const size_t a_shape_3d[3] = {tc.batch, cur_m, tc.k};
    const size_t col_shape_2d[2] = {cur_m, 1};
    const size_t col_shape_3d[3] = {tc.batch, cur_m, 1};
    const size_t* a_shape = rank == 2 ? a_shape_2d : a_shape_3d;
    const size_t* col_shape = rank == 2 ? col_shape_2d : col_shape_3d;

    ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, rank, a_shape),
              ynn_status_success);
    if (tc.has_a_zp) {
      ASSERT_EQ(
          ynn_set_external_value_shape(runtime.get(), a_zp_id, rank, col_shape),
          ynn_status_success);
    }
    if (tc.has_a_scale) {
      ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_scale_id, rank,
                                             col_shape),
                ynn_status_success);
    }
    ASSERT_EQ(
        ynn_set_external_value_shape(runtime.get(), rg_id, rank, col_shape),
        ynn_status_success);

    std::vector<int8_t> a_data(total_rows * tc.k);
    for (size_t i = 0; i < a_data.size(); ++i) {
      a_data[i] = static_cast<int8_t>(static_cast<int>((i + run) % 11) - 5);
    }
    std::vector<int32_t> a_zp_data(total_rows);
    std::vector<float> a_scale_data(total_rows);
    for (size_t r = 0; r < total_rows; ++r) {
      a_zp_data[r] = static_cast<int32_t>((r + run) % 7) - 3;
      a_scale_data[r] = 0.25f * static_cast<float>((r % 4) + 1);
    }
    std::vector<int32_t> rg_data(total_rows);
    if (run == 0 && !tc.rg.empty()) {
      ASSERT_EQ(tc.rg.size(), total_rows);
      rg_data = tc.rg;
    } else {
      for (size_t batch_idx = 0; batch_idx < batches; ++batch_idx) {
        for (size_t m_idx = 0; m_idx < cur_m; ++m_idx) {
          rg_data[batch_idx * cur_m + m_idx] =
              static_cast<int32_t>((m_idx * tc.g) / cur_m);
        }
      }
    }
    std::vector<float> out_data(total_rows * tc.n, -999.0f);

    ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id, a_data.data()),
              ynn_status_success);
    if (tc.has_a_zp) {
      ASSERT_EQ(
          ynn_set_external_value_data(runtime.get(), a_zp_id, a_zp_data.data()),
          ynn_status_success);
    }
    if (tc.has_a_scale) {
      ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_scale_id,
                                            a_scale_data.data()),
                ynn_status_success);
    }
    ASSERT_EQ(ynn_set_external_value_data(runtime.get(), rg_id, rg_data.data()),
              ynn_status_success);
    ASSERT_EQ(
        ynn_set_external_value_data(runtime.get(), out_id, out_data.data()),
        ynn_status_success);

    ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
    ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

    for (size_t row = 0; row < total_rows; ++row) {
      const size_t g_idx = static_cast<size_t>(rg_data[row]);
      const size_t pg_idx = param_g == 1 ? 0 : g_idx;
      const float a_zp =
          tc.has_a_zp ? static_cast<float>(a_zp_data[row]) : 0.0f;
      for (size_t n = 0; n < tc.n; ++n) {
        float sum = 0.0f;
        for (size_t b = 0; b < num_blocks; ++b) {
          float block_sum = 0.0f;
          const float b_zp =
              tc.has_b_zp ? static_cast<float>(
                                b_zp_data[(pg_idx * tc.n + n) * num_blocks + b])
                          : 0.0f;
          for (size_t kb = 0; kb < tc.block_size; ++kb) {
            const size_t k = b * tc.block_size + kb;
            const float a_val =
                static_cast<float>(a_data[row * tc.k + k]) - a_zp;
            const float b_val =
                static_cast<float>(b_data[(g_idx * tc.k + k) * tc.n + n]) -
                b_zp;
            block_sum += a_val * b_val;
          }
          sum += block_sum * b_scale_data[(pg_idx * tc.n + n) * num_blocks + b];
        }
        if (tc.has_a_scale) sum *= a_scale_data[row];
        if (tc.has_bias) sum += bias_data[n];
        EXPECT_NEAR(out_data[row * tc.n + n], sum, 1e-4f)
            << "run=" << run << " row=" << row << " n=" << n;
      }
    }
  }
}

TEST(DotTest, BlockwiseRaggedDot) {
  // Zero-point, bias, and skipped-group variations.
  for (bool has_a_zp : {false, true}) {
    for (bool has_b_zp : {false, true}) {
      BlockwiseRaggedCase tc;
      tc.has_a_zp = has_a_zp;
      tc.has_b_zp = has_b_zp;
      tc.has_bias = has_a_zp && has_b_zp;
      tc.rg = {0, 0, 2, 2};
      RunBlockwiseRaggedCase(tc);
    }
  }

  // Single group (G = 1) and multi-group (G = 4) with bias and dynamic reshape.
  for (size_t g : {size_t{1}, size_t{4}}) {
    BlockwiseRaggedCase tc;
    tc.g = g;
    tc.m = 8;
    tc.n = 8;
    tc.has_bias = true;
    tc.reshape_twice = true;
    RunBlockwiseRaggedCase(tc);
  }

  // num_blocks == G (K = 48, block_size = 16, G = 3) with M in {1, G} and
  // static/dynamic M.
  for (size_t m : {size_t{1}, size_t{3}}) {
    for (bool static_m : {false, true}) {
      BlockwiseRaggedCase tc;
      tc.g = 3;
      tc.m = m;
      tc.k = 48;
      tc.n = 8;
      tc.has_bias = true;
      tc.static_m = static_m;
      tc.rg =
          (m == 1) ? std::vector<int32_t>{2} : std::vector<int32_t>{0, 1, 2};
      RunBlockwiseRaggedCase(tc);
    }
  }

  // Group-broadcast B params ([1, N, num_blocks] across G = 3).
  for (bool dynamic_rg : {false, true}) {
    BlockwiseRaggedCase tc;
    tc.g = 3;
    tc.b_param_g = 1;
    tc.m = 5;
    tc.n = 8;
    tc.has_bias = true;
    tc.dynamic_rg = dynamic_rg;
    tc.reshape_twice = true;
    tc.rg = {0, 1, 1, 2, 2};
    RunBlockwiseRaggedCase(tc);
  }

  // Batched A/row_group ([B = 3, M = 5, K]) with unbatched B ([G = 4, K, N]).
  {
    BlockwiseRaggedCase tc;
    tc.batch = 3;
    tc.g = 4;
    tc.m = 5;
    tc.has_bias = true;
    tc.rg = {0, 0, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 2, 3, 3};
    RunBlockwiseRaggedCase(tc);
  }

  // Batched M = 1, G = 1 without A quantization params (static & dynamic M).
  for (bool static_m : {false, true}) {
    BlockwiseRaggedCase tc;
    tc.batch = 3;
    tc.g = 1;
    tc.m = 1;
    tc.n = 8;
    tc.has_a_zp = false;
    tc.has_a_scale = false;
    tc.has_b_zp = true;
    tc.static_m = static_m;
    tc.dynamic_rg = !static_m;
    RunBlockwiseRaggedCase(tc);
  }
}

TEST(DotTest, BlockwiseRaggedDotUnusedPoisonedGroupAndOutOfRangeGroupFails) {
  subgraph_ptr subgraph = create_subgraph(4, 0);
  ASSERT_NE(subgraph, nullptr);

  constexpr size_t G = 3;
  constexpr size_t M = 6;
  constexpr size_t K = 32;
  constexpr size_t N = 8;
  constexpr size_t block_size = 16;
  constexpr size_t num_blocks = K / block_size;
  const size_t col_dims[2] = {0, 1};

  uint32_t a_id = 0;
  uint32_t a_scale_id = 1;
  uint32_t rg_id = 2;
  uint32_t out_id = 3;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, col_dims, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_scale_id),
      ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int32, 2, col_dims,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &rg_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id),
            ynn_status_success);

  std::vector<int8_t> b_data(G * K * N, 2);
  const size_t b_dims[3] = {G, K, N};
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 3, b_dims,
                              b_data.data(), YNN_VALUE_FLAG_COPY_DATA, &b_id),
            ynn_status_success);

  std::vector<float> b_scale_data(G * N * num_blocks, 0.5f);
  // Poison unused group 1 in b_scale with NaN and -Inf.
  for (size_t n = 0; n < N; ++n) {
    for (size_t b = 0; b < num_blocks; ++b) {
      b_scale_data[(1 * N + n) * num_blocks + b] =
          (b % 2 == 0) ? std::numeric_limits<float>::quiet_NaN()
                       : -std::numeric_limits<float>::infinity();
    }
  }
  const size_t b_param_dims[3] = {G, N, num_blocks};
  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 3, b_param_dims,
                              b_scale_data.data(), YNN_VALUE_FLAG_COPY_DATA,
                              &b_scale_id),
            ynn_status_success);

  ASSERT_EQ(define_blockwise_dot(subgraph.get(), a_id, YNN_INVALID_VALUE_ID,
                                 a_scale_id, b_id, YNN_INVALID_VALUE_ID,
                                 b_scale_id, block_size, YNN_INVALID_VALUE_ID,
                                 ynn_type_fp32, out_id, 0, rg_id),
            ynn_status_success);
  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), nullptr, 0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  const size_t a_shape[2] = {M, K};
  const size_t col_shape[2] = {M, 1};
  std::vector<int8_t> a_data(M * K, 3);
  std::vector<float> a_scale_data(M, 0.25f);
  std::vector<int32_t> rg_data = {0, 0, 0, 2, 2, 2};
  std::vector<float> out_data(M * N, std::numeric_limits<float>::quiet_NaN());

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_shape),
            ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_shape(runtime.get(), a_scale_id, 2, col_shape),
      ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), rg_id, 2, col_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id, a_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_scale_id,
                                        a_scale_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), rg_id, rg_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), out_id, out_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  const float expected = static_cast<float>(K * 3 * 2) * 0.5f * 0.25f;
  for (float v : out_data) {
    EXPECT_NEAR(v, expected, 1e-4f);
  }

  for (int32_t bad_group : {int32_t{-1}, static_cast<int32_t>(G)}) {
    rg_data = {0, 0, 0, bad_group, 2, 2};
    EXPECT_NE(ynn_invoke_runtime(runtime.get()), ynn_status_success)
        << "bad_group=" << bad_group;
  }
  rg_data = {0, 0, 2, 0, 2, 2};
  EXPECT_NE(ynn_invoke_runtime(runtime.get()), ynn_status_success);
}

}  // namespace
}  // namespace ynn
