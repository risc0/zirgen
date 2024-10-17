// This code is automatically generated

#include "fp.h"
#include "fpext.h"

#include <cstdint>

constexpr size_t INV_RATE = 4;
__constant__ FpExt poly_mix[4];

__device__ FpExt poly_fp(uint32_t idx,
                         uint32_t size,
                         const Fp* ctrl,
                         const Fp* out,
                         const Fp* data,
                         const Fp* mix,
                         const Fp* accum) {
  uint32_t mask = size - 1;
  Fp x0(0);
  Fp x1(1);
  Fp x2({0, 1, 0, 0});
  FpExt x3 = FpExt(0);
  Fp x4 = global[0];
  Fp x5 = x0 - x4;
  FpExt x6 = x3 + poly_mix[0] * x5;
  Fp x7 = data[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x8 = x1 - x7;
  Fp x9 = data[1 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x10 = x8 - x9;
  FpExt x11 = x6 + poly_mix[1] * x10;
  Fp x12 = mix[3];
  Fp x13 = mix[2];
  Fp x14 = x12 * x2;
  Fp x15 = x13 + x14;
  Fp x16 = mix[1];
  Fp x17 = x15 * x2;
  Fp x18 = x16 + x17;
  Fp x19 = mix[0];
  Fp x20 = x18 * x2;
  Fp x21 = x19 + x20;
  Fp x22 = mix[7];
  Fp x23 = mix[6];
  Fp x24 = x22 * x2;
  Fp x25 = x23 + x24;
  Fp x26 = mix[5];
  Fp x27 = x25 * x2;
  Fp x28 = x26 + x27;
  Fp x29 = mix[4];
  Fp x30 = x28 * x2;
  Fp x31 = x29 + x30;
  Fp x32 = accum[3 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x33 = accum[2 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x34 = x32 * x2;
  Fp x35 = x33 + x34;
  Fp x36 = accum[1 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x37 = x35 * x2;
  Fp x38 = x36 + x37;
  Fp x39 = accum[0 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x40 = x38 * x2;
  Fp x41 = x39 + x40;
  Fp x42 = data[2 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x43 = data[3 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x44 = x21 * x43;
  Fp x45 = x44 + x31;
  Fp x46 = accum[3 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x47 = accum[2 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x48 = x46 * x2;
  Fp x49 = x47 + x48;
  Fp x50 = accum[1 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x51 = x49 * x2;
  Fp x52 = x50 + x51;
  Fp x53 = accum[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x54 = x52 * x2;
  Fp x55 = x53 + x54;
  Fp x56 = x55 - x41;
  Fp x57 = x56 * x45;
  Fp x58 = x57 - x42;
  FpExt x59 = x3 + poly_mix[0] * x58;
  FpExt x60 = x11 + x7 * x59 * poly_mix[2];
  FpExt x61 = x60 + x9 * x59 * poly_mix[3];
  return x61;
}

__global__ void eval_check(Fp* check,
                           const Fp* ctrl,
                           const Fp* data,
                           const Fp* accum,
                           const Fp* mix,
                           const Fp* out,
                           const Fp rou,
                           uint32_t po2,
                           uint32_t domain) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < domain) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum);
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret.elems[0];
    check[domain * 1 + cycle] = ret.elems[1];
    check[domain * 2 + cycle] = ret.elems[2];
    check[domain * 3 + cycle] = ret.elems[3];
  }
}
