// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, k, mr, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, unroll)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, k, mr) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, k, mr, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packx_ukernel_4x__neon_st4_x4, /*k=*/4, /*mr=*/4)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packx_ukernel_4x__neon_st4_x4_prfm, /*k=*/4, /*mr=*/4)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packx_ukernel_4x__neon_st4_x8, /*k=*/8, /*mr=*/4)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packx_ukernel_4x__neon_st4_x8_prfm, /*k=*/8, /*mr=*/4)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packx_ukernel_8x__neon_st4_x4, /*k=*/4, /*mr=*/8)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packx_ukernel_8x__neon_st4_x4_prfm, /*k=*/4, /*mr=*/8)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packx_ukernel_8x__neon_st4_x8, /*k=*/8, /*mr=*/8)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packx_ukernel_8x__neon_st4_x8_prfm, /*k=*/8, /*mr=*/8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(0, xnn_x32_packx_ukernel_4x__sse, /*k=*/4, /*mr=*/4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_x32_packx_ukernel_4x__wasmsimd, /*k=*/4, /*mr=*/4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_x32_packx_ukernel_2x__scalar, /*k=*/1, /*mr=*/2)
XNN_UKERNEL(0, xnn_x32_packx_ukernel_3x__scalar, /*k=*/1, /*mr=*/3)
XNN_UKERNEL(0, xnn_x32_packx_ukernel_4x__scalar, /*k=*/1, /*mr=*/4)


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

