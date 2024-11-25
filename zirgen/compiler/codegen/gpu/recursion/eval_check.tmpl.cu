// This code is automatically generated

#include "supra/fp.h"
#include "supra/fpext.h"

#include <cstdint>

namespace {{cppNamespace}} {

constexpr size_t INV_RATE = 4;
__constant__ FpExt poly_mix[{{num_mix_powers}}];

static __device__ __forceinline__ FpExt poly_fp(uint32_t idx,
                                                uint32_t size,
                                                const Fp* ctrl,
                                                const Fp* out,
                                                const Fp* data,
                                                const Fp* mix,
                                                const Fp* accum) {
  uint32_t mask = size - 1;
{{#block}}
  {{.}}
{{/block}}
}

__global__ void eval_check(Fp* check,
                           const Fp* ctrl,
                           const Fp* data,
                           const Fp* accum,
                           const Fp* mix,
                           const Fp* out,
                           const Fp rou,
                           const uint32_t po2,
                           const uint32_t domain) {
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

} {{cppNamespace}}
