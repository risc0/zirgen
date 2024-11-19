#include "supra/fp.h"

#include <cstdint>

__constant__ FpExt poly_mix[483];

extern __device__ FpExt keccak_poly_fp(uint32_t idx,
                                       uint32_t size,
                                       const Fp* ctrl,
                                       const Fp* out,
                                       const Fp* data,
                                       const Fp* mix,
                                       const Fp* accum);

__global__ void keccak_eval_check(Fp* check,
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
    FpExt tot = keccak_poly_fp(cycle, domain, ctrl, out, data, mix, accum);
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
  }
}
