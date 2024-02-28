// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include "batch-matrix-multiply-operator-tester.h"
#include <gtest/gtest.h>

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_1_b_2) {
  BatchMatMulOperatorTester()
      .batch_dims_a({1})
      .batch_dims_b({2})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_1_b_3) {
  BatchMatMulOperatorTester()
      .batch_dims_a({1})
      .batch_dims_b({3})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_2_b_6) {
  BatchMatMulOperatorTester()
      .batch_dims_a({2})
      .batch_dims_b({6})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_3_b_6) {
  BatchMatMulOperatorTester()
      .batch_dims_a({3})
      .batch_dims_b({6})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_2_3_b_2_3) {
  BatchMatMulOperatorTester()
      .batch_dims_a({2, 3})
      .batch_dims_b({2, 3})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_1_3_b_2_3) {
  BatchMatMulOperatorTester()
      .batch_dims_a({1, 3})
      .batch_dims_b({2, 3})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_2_3_b_1_3) {
  BatchMatMulOperatorTester()
      .batch_dims_a({2, 3})
      .batch_dims_b({1, 3})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_2_3_b_4_3) {
  BatchMatMulOperatorTester()
      .batch_dims_a({2, 3})
      .batch_dims_b({4, 3})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, broadcast_a_2_6_b_4_3) {
  BatchMatMulOperatorTester()
      .batch_dims_a({2, 6})
      .batch_dims_b({4, 3})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, bad_broadcast_a_4_b_6) {
  BatchMatMulOperatorTester()
      .batch_dims_a({4})
      .batch_dims_b({6})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .expected_status_reshape(xnn_status_invalid_parameter)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, unit_batch) {
  BatchMatMulOperatorTester()
      .batch_dims_a({1})
      .batch_dims_b({1})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, small_batch) {
  BatchMatMulOperatorTester()
      .batch_dims_a({5})
      .batch_dims_b({5})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, unit_batch_bigger_matrices) {
  BatchMatMulOperatorTester()
      .batch_dims_a({1})
      .batch_dims_b({1})
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, small_batch_bigger_matrices) {
  BatchMatMulOperatorTester()
      .batch_dims_a({5})
      .batch_dims_b({5})
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, unit_batch_transpose_b) {
  BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_dims_a({1})
      .batch_dims_b({1})
      .m(17)
      .k(23)
      .n(17)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, small_batch_tranpose_b) {
  BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_dims_a({5})
      .batch_dims_b({5})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, unit_batch_bigger_matrices_transpose_b) {
  BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_dims_a({1})
      .batch_dims_b({1})
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F32, small_batch_bigger_matrices_transpose_b) {
  BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_dims_a({5})
      .batch_dims_b({5})
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF32();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, unit_batch) {
  BatchMatMulOperatorTester()
      .batch_dims_a({1})
      .batch_dims_b({1})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, small_batch) {
  BatchMatMulOperatorTester()
      .batch_dims_a({5})
      .batch_dims_b({5})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, unit_batch_bigger_matrices) {
  BatchMatMulOperatorTester()
      .batch_dims_a({1})
      .batch_dims_b({1})
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, small_batch_bigger_matrices) {
  BatchMatMulOperatorTester()
      .batch_dims_a({5})
      .batch_dims_b({5})
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, unit_batch_transpose_b) {
  BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_dims_a({1})
      .batch_dims_b({1})
      .m(17)
      .k(23)
      .n(17)
      .iterations(3)
      .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, small_batch_tranpose_b) {
  BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_dims_a({5})
      .batch_dims_b({5})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, unit_batch_bigger_matrices_transpose_b) {
  BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_dims_a({1})
      .batch_dims_b({1})
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF16();
}

TEST(BATCH_MATRIX_MULTIPLY_NC_F16, small_batch_bigger_matrices_transpose_b) {
  BatchMatMulOperatorTester()
      .transpose_b(true)
      .batch_dims_a({5})
      .batch_dims_b({5})
      .m(37)
      .k(101)
      .n(71)
      .iterations(3)
      .TestF16();
}
