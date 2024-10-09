// This code is automatically generated

#include "fp.h"
#include "fpext.h"

#include <cstdint>

constexpr size_t INV_RATE = 4;
__constant__ FpExt poly_mix[13];

__device__ FpExt poly_fp(uint32_t idx,
                         uint32_t size,
                         const Fp* ctrl,
                         const Fp* out,
                         const Fp* data,
                         const Fp* mix,
                         const Fp* accum) {
  uint32_t mask = size - 1;
  Fp x0(1);
  Fp x1(0);
  Fp x2({0, 1, 0, 0});
  FpExt x3 = FpExt(0);
  Fp x4 = data[2 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x5 = data[3 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x6 = data[4 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x7 = data[5 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x8 = data[6 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x9 = x0 - x7;
  Fp x10 = x9 * x7;
  FpExt x11 = x3 + poly_mix[0] * x10;
  Fp x12 = x0 - x8;
  Fp x13 = x12 * x8;
  FpExt x14 = x11 + poly_mix[1] * x13;
  Fp x15 = x7 + x8;
  Fp x16 = x15 - x0;
  FpExt x17 = x14 + poly_mix[2] * x16;
  Fp x18 = x8 - x4;
  FpExt x19 = x17 + poly_mix[3] * x18;
  Fp x20 = x5 + x6;
  Fp x21 = data[7 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x22 = x20 - x21;
  Fp x23 = x22 * x7;
  FpExt x24 = x19 + poly_mix[4] * x23;
  Fp x25 = x5 - x6;
  Fp x26 = x25 - x21;
  Fp x27 = x26 * x8;
  FpExt x28 = x24 + poly_mix[5] * x27;
  Fp x29 = global[0];
  Fp x30 = x21 - x29;
  FpExt x31 = x28 + poly_mix[6] * x30;
  Fp x32 = x29 - x21;
  FpExt x33 = x31 + poly_mix[7] * x32;
  Fp x34 = data[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x35 = x0 - x34;
  FpExt x36 = x33 + poly_mix[8] * x35;
  Fp x37 = data[1 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x38 = x1 - x37;
  FpExt x39 = x36 + poly_mix[9] * x38;
  Fp x40 = data[8 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x41 = x1 - x40;
  Fp x42 = x41 * x37;
  FpExt x43 = x39 + poly_mix[10] * x42;
  Fp x44 = mix[3];
  Fp x45 = mix[2];
  Fp x46 = x44 * x2;
  Fp x47 = x45 + x46;
  Fp x48 = mix[1];
  Fp x49 = x47 * x2;
  Fp x50 = x48 + x49;
  Fp x51 = mix[0];
  Fp x52 = x50 * x2;
  Fp x53 = x51 + x52;
  Fp x54 = mix[7];
  Fp x55 = mix[6];
  Fp x56 = x54 * x2;
  Fp x57 = x55 + x56;
  Fp x58 = mix[5];
  Fp x59 = x57 * x2;
  Fp x60 = x58 + x59;
  Fp x61 = mix[4];
  Fp x62 = x60 * x2;
  Fp x63 = x61 + x62;
  Fp x64 = accum[3 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x65 = accum[2 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x66 = x64 * x2;
  Fp x67 = x65 + x66;
  Fp x68 = accum[1 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x69 = x67 * x2;
  Fp x70 = x68 + x69;
  Fp x71 = accum[0 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x72 = x70 * x2;
  Fp x73 = x71 + x72;
  Fp x74 = data[11 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x75 = data[10 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x76 = x74 * x2;
  Fp x77 = x75 + x76;
  Fp x78 = data[9 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x79 = x77 * x2;
  Fp x80 = x78 + x79;
  Fp x81 = x80 * x2;
  Fp x82 = x40 + x81;
  Fp x83 = data[12 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x84 = x83 * x2;
  Fp x85 = x74 + x84;
  Fp x86 = x85 * x2;
  Fp x87 = x75 + x86;
  Fp x88 = x87 * x2;
  Fp x89 = x78 + x88;
  Fp x90 = x53 * x89;
  Fp x91 = x90 + x63;
  Fp x92 = accum[3 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x93 = accum[2 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x94 = x92 * x2;
  Fp x95 = x93 + x94;
  Fp x96 = accum[1 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x97 = x95 * x2;
  Fp x98 = x96 + x97;
  Fp x99 = accum[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x100 = x98 * x2;
  Fp x101 = x99 + x100;
  Fp x102 = x101 - x73;
  Fp x103 = x102 * x91;
  Fp x104 = x103 - x82;
  Fp x105 = x34 * x104;
  FpExt x106 = x43 + poly_mix[11] * x105;
  Fp x107 = x102 * x37;
  FpExt x108 = x106 + poly_mix[12] * x107;
  return x108;
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
