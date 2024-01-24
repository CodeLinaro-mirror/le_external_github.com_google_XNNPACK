// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/scalar-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>


void xnn_f32_vrsqrt_ukernel__scalar_rsqrt_u12(
    size_t batch, const float* input, float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    const float vx4 = input[4];
    const float vx5 = input[5];
    const float vx6 = input[6];
    const float vx7 = input[7];
    const float vx8 = input[8];
    const float vx9 = input[9];
    const float vxA = input[10];
    const float vxB = input[11];
    input += 12;

    const float vt0 = sqrtf(vx0);
    const float vt1 = sqrtf(vx1);
    const float vt2 = sqrtf(vx2);
    const float vt3 = sqrtf(vx3);
    const float vt4 = sqrtf(vx4);
    const float vt5 = sqrtf(vx5);
    const float vt6 = sqrtf(vx6);
    const float vt7 = sqrtf(vx7);
    const float vt8 = sqrtf(vx8);
    const float vt9 = sqrtf(vx9);
    const float vtA = sqrtf(vxA);
    const float vtB = sqrtf(vxB);
    const float vy0 = 1.0f / vt0;
    const float vy1 = 1.0f / vt1;
    const float vy2 = 1.0f / vt2;
    const float vy3 = 1.0f / vt3;
    const float vy4 = 1.0f / vt4;
    const float vy5 = 1.0f / vt5;
    const float vy6 = 1.0f / vt6;
    const float vy7 = 1.0f / vt7;
    const float vy8 = 1.0f / vt8;
    const float vy9 = 1.0f / vt9;
    const float vyA = 1.0f / vtA;
    const float vyB = 1.0f / vtB;

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output[4] = vy4;
    output[5] = vy5;
    output[6] = vy6;
    output[7] = vy7;
    output[8] = vy8;
    output[9] = vy9;
    output[10] = vyA;
    output[11] = vyB;
    output += 12;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vy = 1.0f / sqrtf(vx);
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
