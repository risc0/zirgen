// This code is automatically generated

#include <metal_stdlib>

#include "fp.h"
#include "fpext.h"

using namespace metal;

constant size_t INV_RATE = 4;

FpExt poly_fp(uint idx,
            uint size,
            const device Fp* code,
            const device Fp* out,
            const device Fp* data,
            const device Fp* mix,
            const device Fp* accum,
            const device FpExt* poly_mix) {
    uint mask = size - 1;
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

kernel void eval_check(device Fp* check,
                       const device Fp* code,
                       const device Fp* data,
                       const device Fp* accum,
                       const device Fp* mix,
                       const device Fp* out,
                       const device FpExt* poly_mix,
                       const device Fp& rou,
                       const device uint32_t& po2,
                       const device uint32_t& domain,
                       uint cycle [[thread_position_in_grid]]) {
    FpExt tot = poly_fp(cycle, domain, code, out, data, mix, accum, poly_mix);
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret.elems[0];
    check[domain * 1 + cycle] = ret.elems[1];
    check[domain * 2 + cycle] = ret.elems[2];
    check[domain * 3 + cycle] = ret.elems[3];
}
