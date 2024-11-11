// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#ifdef __cplusplus

#include <type_traits>

namespace xnnpack {

// We need a type that distinguishes an intX_t from a quantized intX_t. We can't
// do arithmetic on these, because we don't know the quantization parameters.
// TODO: find a better file for this to live in
template <typename T>
struct quantized {
  T value;
  using type = T;

  operator T() const { return value; }

  quantized() = default;
  quantized(T t) : value(t) {}
  quantized<T>& operator=(T t) {
    value = t;
    return *this;
  }
};

template <typename T>
struct is_quantized : std::false_type {};

template <typename T>
struct is_quantized<quantized<T>> : std::true_type {};

template <typename T>
struct unwrap_quantized {
  using type = T;
};

template <>
struct unwrap_quantized<quantized<int8_t>> {
  using type = int8_t;
};

template <>
struct unwrap_quantized<quantized<uint8_t>> {
  using type = uint8_t;
};

}  // namespace xnnpack

#endif
