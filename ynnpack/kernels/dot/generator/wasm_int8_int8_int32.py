# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 wasm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.wasm import wasm


class wasm_int8_int8_int32(wasm):
  def __init__(self, arch, tile_shape):
    super().__init__(arch, "int8_int8_int32", "int32_t", tile_shape)
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]


class wasm_simd128_int8_int8_int32(wasm_int8_int8_int32):

  def __init__(self, arch="simd128", tile_shape=(1, 4, 1)):
    super().__init__(arch, tile_shape)

  def load_a_tile_k_tail(self, i, k, nk):
    if k % nk != 0:
      return ""
    if nk == 8:
      return f"v128_t a_{i}_{k} = wasm_i16x8_load8x8({self.a_ptr(i, k)});\n"
    else:
      assert(nk == 1)
      return (
          f"int32_t a_val_{i}_{k} = (int32_t)*{self.a_ptr(i, k)};\n"
          f"v128_t a_{i}_{k} = wasm_i16x8_splat(a_val_{i}_{k});\n"
      )

  def load_b_tile(self, k, j):
    if self.b_chunk_n < 8 or j + 8 > self.block_shape[1]:
      # We can only load 4 values.
      load32 = f"wasm_v128_load32_zero({self.b_ptr(k, j, 'int32_t')})"
      return f"v128_t b_{k}_{j} = wasm_i16x8_extend_low_i8x16({load32});\n"
    elif j % 8 == 0:
      # The next 8 values are all in bounds of the block.
      return f"v128_t b_{k}_{j} = wasm_i16x8_load8x8({self.b_ptr(k, j)});\n"
    else:
      return ""

  def product(self, i, j, k):
    if self.b_chunk_n < 8 or j + 8 > self.block_shape[1]:
      b_load = f"b_{k}_{(j//4)*4}"
      b_ext = f"wasm_i32x4_extend_low_i16x8({b_load})"
    else:
      b_load = f"b_{k}_{(j//8)*8}"
      b_ext = (
          f"wasm_i32x4_extend_low_i16x8({b_load})"
          if j % 8 < 4
          else f"wasm_i32x4_extend_high_i16x8({b_load})"
      )

    a_lane = f"wasm_i16x8_extract_lane(a_{i}_{(k//8)*8}, {k%8})"
    a_ext = f"wasm_i32x4_splat({a_lane})"
    return (
        f"c_{i}_{j} = wasm_i32x4_add(c_{i}_{j}, wasm_i32x4_mul({b_ext},"
        f" {a_ext}));\n"
    )


generate_dot_kernels(
    wasm_simd128_int8_int8_int32(),
    [
        (1, 16, 8),
        (2, 16, 8),
        (3, 16, 8),
        (4, 16, 8),
        (1, 8, 8),
        (2, 8, 8),
        (3, 8, 8),
        (4, 8, 8),
        (5, 8, 8),
        (6, 8, 8),
        (8, 8, 8),
        (8, 4, 8),
    ],
)
