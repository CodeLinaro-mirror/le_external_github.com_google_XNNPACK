// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, unroll)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4, /*NR=*/12, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4_prfm, /*NR=*/12, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8, /*NR=*/12, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8_prfm, /*NR=*/12, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4, /*NR=*/2, /*KR=*/4, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4_prfm, /*NR=*/2, /*KR=*/4, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4, /*NR=*/2, /*KR=*/4, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_u4, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0,   xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_u4, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_u4, /*NR=*/3, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_u4, /*NR=*/3, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_u4, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
XNN_UKERNEL(0, xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1)
#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u2, /*NR=*/1, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u4, /*NR=*/1, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u8, /*NR=*/1, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u2, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u4, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u8, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u2, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u4, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u8, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u2, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u8, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/(xnn_init_hardware_config()->vlenb / sizeof(uint32_t)))
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV()


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

