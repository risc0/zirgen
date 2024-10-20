// This code is automatically generated

#include "fp.h"
#include "fpext.h"

#include <cstdint>

constexpr size_t INV_RATE = 4;
__constant__ FpExt poly_mix[6];

__device__ FpExt poly_fp(uint32_t idx,
                         uint32_t size,
                         const Fp* ctrl,
                         const Fp* out,
                         const Fp* data,
                         const Fp* mix,
                         const Fp* accum) {
  uint32_t mask = size - 1;
  Fp x0(11);
  Fp x1(1);
  Fp x2(2);
  Fp x3(3);
  Fp x4(2013265919);
  Fp x5({0, 1, 0, 0});
  FpExt x6 = FpExt(0);
  Fp x7 = global[0];
  Fp x8 = x0 - x7;
  FpExt x9 = x6 + poly_mix[0] * x8;
  Fp x10 = data[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x11 = x1 - x10;
  Fp x12 = data[1 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x13 = x11 - x12;
  FpExt x14 = x9 + poly_mix[1] * x13;
  Fp x15 = data[2 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x16 = data[3 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x17 = x2 - x15;
  FpExt x18 = x6 + poly_mix[0] * x17;
  Fp x19 = x3 - x16;
  FpExt x20 = x18 + poly_mix[1] * x19;
  FpExt x21 = x14 + x10 * x20 * poly_mix[2];
  Fp x22 = x4 - x15;
  FpExt x23 = x6 + poly_mix[0] * x22;
  FpExt x24 = x23 + poly_mix[1] * x19;
  FpExt x25 = x21 + x12 * x24 * poly_mix[3];
  Fp x26 = mix[3];
  Fp x27 = mix[2];
  Fp x28 = x26 * x5;
  Fp x29 = x28 + x27;
  Fp x30 = mix[1];
  Fp x31 = x29 * x5;
  Fp x32 = x31 + x30;
  Fp x33 = mix[0];
  Fp x34 = x32 * x5;
  Fp x35 = x34 + x33;
  Fp x36 = mix[7];
  Fp x37 = mix[6];
  Fp x38 = x36 * x5;
  Fp x39 = x38 + x37;
  Fp x40 = mix[5];
  Fp x41 = x39 * x5;
  Fp x42 = x41 + x40;
  Fp x43 = mix[4];
  Fp x44 = x42 * x5;
  Fp x45 = x44 + x43;
  Fp x46 = accum[3 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x47 = accum[2 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x48 = x46 * x5;
  Fp x49 = x48 + x47;
  Fp x50 = accum[1 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x51 = x49 * x5;
  Fp x52 = x51 + x50;
  Fp x53 = accum[0 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x54 = x52 * x5;
  Fp x55 = x54 + x53;
  Fp x56 = x35 * x16;
  Fp x57 = x56 + x45;
  Fp x58 = accum[3 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x59 = accum[2 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x60 = x58 * x5;
  Fp x61 = x60 + x59;
  Fp x62 = accum[1 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x63 = x61 * x5;
  Fp x64 = x63 + x62;
  Fp x65 = accum[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x66 = x64 * x5;
  Fp x67 = x66 + x65;
  Fp x68 = x67 - x55;
  Fp x69 = x68 * x57;
  Fp x70 = x69 - x15;
  FpExt x71 = x6 + poly_mix[0] * x70;
  FpExt x72 = x25 + x10 * x71 * poly_mix[4];
  FpExt x73 = x72 + x12 * x71 * poly_mix[5];
  return x73;
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
