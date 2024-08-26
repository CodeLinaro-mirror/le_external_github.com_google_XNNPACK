// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, channel_subtile, channel_round, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, unroll)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_subtile, channel_round) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, channel_subtile, channel_round, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif


XNN_UKERNEL(0, xnn_x32_zerob_gemm_ukernel_2c1s1r__scalar_float, /*channel_tile=*/2, /*channel_subtile=*/1, /*channel_round=*/1)
XNN_UKERNEL(0, xnn_x32_zerob_gemm_ukernel_2c1s1r__scalar_int, /*channel_tile=*/2, /*channel_subtile=*/1, /*channel_round=*/1)
XNN_UKERNEL(0, xnn_x32_zerob_gemm_ukernel_2c2s1r__scalar_float, /*channel_tile=*/2, /*channel_subtile=*/2, /*channel_round=*/1)
XNN_UKERNEL(0, xnn_x32_zerob_gemm_ukernel_2c2s1r__scalar_int, /*channel_tile=*/2, /*channel_subtile=*/2, /*channel_round=*/1)
XNN_UKERNEL(0, xnn_x32_zerob_gemm_ukernel_4c1s1r__scalar_float, /*channel_tile=*/4, /*channel_subtile=*/1, /*channel_round=*/1)
XNN_UKERNEL(0, xnn_x32_zerob_gemm_ukernel_4c1s1r__scalar_int, /*channel_tile=*/4, /*channel_subtile=*/1, /*channel_round=*/1)
XNN_UKERNEL(0, xnn_x32_zerob_gemm_ukernel_4c4s1r__scalar_float, /*channel_tile=*/4, /*channel_subtile=*/4, /*channel_round=*/1)
XNN_UKERNEL(0, xnn_x32_zerob_gemm_ukernel_4c4s1r__scalar_int, /*channel_tile=*/4, /*channel_subtile=*/4, /*channel_round=*/1)


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

