// This code is automatically generated

#include "fp.h"
#include "fpext.h"

#include <cstdint>

constexpr size_t INV_RATE = 4;
__constant__ FpExt poly_mix[4];

__device__
FpExt poly_fp(uint32_t idx,
            uint32_t size,
            const Fp* code,
            const Fp* out,
            const Fp* data,
            const Fp* mix,
            const Fp* accum) {
  uint32_t mask = size - 1;
  Fp x5(1);
  FpExt x6 = FpExt(0);
  Fp x7 = code[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x8 = data[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x9 = x8 - x5;
  FpExt x10 = x6 + poly_mix[0] * x9;
  FpExt x11 = x6 + x7 * x10 * poly_mix[0];
  Fp x12 = code[1 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x13 = data[0 * size + ((idx - INV_RATE * 2) & mask)];
  Fp x14 = data[0 * size + ((idx - INV_RATE * 1) & mask)];
  Fp x15 = x14 + x13;
  Fp x16 = x8 - x15;
  FpExt x17 = x6 + poly_mix[0] * x16;
  FpExt x18 = x11 + x12 * x17 * poly_mix[1];
  Fp x19 = code[2 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x20 = out[0];
  Fp x21 = x20 - x8;
  FpExt x22 = x6 + poly_mix[0] * x21;
  FpExt x23 = x18 + x19 * x22 * poly_mix[2];
  Fp x24 = x7 + x12;
  Fp x25 = x24 + x19;
  Fp x26 = accum[0 * size + ((idx - INV_RATE * 0) & mask)];
  Fp x27 = x26 - x5;
  FpExt x28 = x6 + poly_mix[0] * x27;
  FpExt x29 = x23 + x25 * x28 * poly_mix[3];
  return x29;
}

extern "C" __global__
void eval_check(Fp* check,
                const Fp* code,
                const Fp* data,
                const Fp* accum,
                const Fp* mix,
                const Fp* out,
                const Fp& rou,
                const uint32_t& po2,
                const uint32_t& domain) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < domain) {
    FpExt tot = poly_fp(cycle, domain, code, out, data, mix, accum);
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret.elems[0];
    check[domain * 1 + cycle] = ret.elems[1];
    check[domain * 2 + cycle] = ret.elems[2];
    check[domain * 3 + cycle] = ret.elems[3];
  }
}
