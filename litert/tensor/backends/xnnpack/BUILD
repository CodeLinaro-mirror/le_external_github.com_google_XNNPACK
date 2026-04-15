# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "arithmetic_xnnpack",
    srcs = ["arithmetic_xnnpack.cc"],
    hdrs = ["arithmetic_xnnpack.h"],
    deps = [
        ":xnnpack_utils",
        "//:XNNPACK",
        "//litert/tensor:arithmetic_graph",
        "//litert/tensor/internal:buffer",
        "//litert/tensor/internal:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/internal:mixin",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_test(
    name = "arithmetic_xnnpack_test",
    srcs = ["arithmetic_xnnpack_test.cc"],
    deps = [
        ":arithmetic_xnnpack",
        ":xnnpack_conversion",
        "//:XNNPACK",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor/internal:buffer",
        "//litert/tensor/internal:datatypes",
        "//litert/tensor/runners/xnnpack:xnnpack_runner",
        "//litert/tensor/utils:matchers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "xnnpack_conversion",
    srcs = ["xnnpack_conversion.cc"],
    hdrs = ["xnnpack_conversion.h"],
    deps = [
        ":arithmetic_xnnpack",
        ":xnnpack_utils",
        "//:XNNPACK",
        "//litert/tensor",
        "//litert/tensor/internal:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/internal:graph_traversal",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "xnnpack_utils",
    hdrs = ["xnnpack_utils.h"],
    deps = [
        "//:XNNPACK",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_test(
    name = "xnnpack_conversion_test",
    srcs = ["xnnpack_conversion_test.cc"],
    deps = [
        ":arithmetic_xnnpack",
        ":xnnpack_conversion",
        "//:XNNPACK",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor/internal:datatypes",
        "//litert/tensor/utils:matchers",
        "@abseil-cpp//absl/status",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)
