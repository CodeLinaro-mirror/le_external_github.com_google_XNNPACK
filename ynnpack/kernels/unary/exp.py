"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("exp")
def exp_fp32(a, x, output_multiplier, input_multiplier):
  # Polynomial coefficients
  p = [
      2.7769648004e-03 * output_multiplier,
      4.8084631562e-02 * output_multiplier,
      3.4672188759e-01 * output_multiplier,
      1.0000000000e00 * output_multiplier,
  ]
  q = [
      -2.7651260607e-03,
      4.7981843352e-02,
      -3.4642529488e-01,
      1.0000000000e00,
  ]

  va = load(a) * input_multiplier
  # Clamp `vz_prime = x * log2(e)` to the maximum exponents.
  vz_prime = min(max(va, -127.0), 128.0)

  # Decompose x * log2e into `z` (integer part) and `r` (remainder).
  vz = round_small_fp32(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = exp2_round(vz)
  v2z = copynan(v2z, va)

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)

  # Divide the numerator by the denominator, obtaining 2^r.
  v2r = vp / vq

  # Compute 2^z * 2^r.
  vx = v2z * v2r

  return store(vx, x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    # These are actually fp32, but we convert them to fp64 when loading them.
    Scalar("output_multiplier", Float(64)),
    Scalar("input_multiplier", Float(64)),
)
@operator_name("exp")
def exp_fp64(a, x, output_multiplier, input_multiplier):
  # Polynomial coefficients
  p = [
      f64(3.430671987749682348e-06) * output_multiplier,
      f64(1.754214714900316551e-04) * output_multiplier,
      f64(3.930681642138933278e-03) * output_multiplier,
      f64(4.871246780757146344e-02) * output_multiplier,
      f64(3.331084219217221309e-01) * output_multiplier,
      f64(9.999999999999998890e-01) * output_multiplier,
  ]
  q = [
      f64(-7.126417699553038831e-06),
      f64(2.821496207245147419e-04),
      f64(-5.316864104706260294e-03),
      f64(5.804581129084830649e-02),
      f64(-3.600387586382234328e-01),
      f64(1.000000000000000000e00),
  ]

  va = load(a) * input_multiplier
  # Clamp `vz_prime = x * log2(e)` to the maximum exponents.
  vz_prime = min(max(va, -1023.0), 1024.0)

  # Decompose x * log2e into `z` (integer part) and `r` (remainder).
  vz = round_small_fp64(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = exp2_round(vz)
  v2z = copynan(v2z, va)

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)

  # Divide the numerator by the denominator, obtaining 2^r.
  v2r = vp / vq

  # Compute 2^z * 2^r.
  vx = v2z * v2r

  return store(vx, x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("expm1")
def expm1_fp32(a, x, output_multiplier, input_multiplier):
  # Polynomial coefficients
  p = [
      5.4308967665e-03,
      -1.4816340059e-02,
      6.9314712286e-01,
      0.0000000000e+00
  ]
  q = [
      -3.5687740892e-03,
      5.5278953165e-02,
      -3.6794918776e-01,
      1.0000000000e+00
  ]

  va = load(a) * input_multiplier
  # Clamp `vz_prime = x * log2(e)` to the maximum exponents.
  vz_prime = min(max(va, -127.0), 128.0)

  # Decompose x into `z` (integer part) and `r` (remainder).
  vz = round_small_fp32(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = exp2_round(vz)
  v2z = copynan(v2z, va)

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)

  # Divide the numerator by the denominator, obtaining 2^r - 1.
  v2r = vp / vq

  # Compute 2^z * (v2r + 1), rearranging the arithmetic to avoid catastrophic
  # cancellation.
  vx = v2z * v2r + (v2z - 1.0)

  return store(vx * output_multiplier, x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("output_multiplier", Float(64)),
    Scalar("input_multiplier", Float(64)),
)
@operator_name("expm1")
def expm1_fp64(a, x, output_multiplier, input_multiplier):
  # Polynomial coefficients
  p = [
      f64(1.060941252694225928e-05),
      f64(3.247057313720804134e-05),
      f64(9.254091633662777511e-03),
      f64(2.839398265006370936e-03),
      f64(6.931471805599456193e-01),
      f64(0.000000000000000000e+00)
  ]
  q = [
      f64(-4.742450436450693637e-06),
      f64(2.130038498162553369e-04),
      f64(-4.416190606899681077e-03),
      f64(5.196888413519899774e-02),
      f64(-3.424772044839380358e-01),
      f64(1.000000000000000000e+00)
  ]

  va = load(a) * input_multiplier
  # Clamp `vz_prime = x * log2(e)` to the maximum exponents.
  vz_prime = min(max(va, -1023.0), 1024.0)

  # Decompose x into `z` (integer part) and `r` (remainder).
  vz = round_small_fp64(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = exp2_round(vz)
  v2z = copynan(v2z, va)

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)

  # Divide the numerator by the denominator, obtaining 2^r - 1.
  v2r = vp / vq

  # Compute 2^z * (v2r + 1), rearranging the arithmetic to avoid catastrophic
  # cancellation when v2z = 1.
  vx = v2z * v2r + (v2z - f64(1.0))

  return store(vx * output_multiplier, x)
