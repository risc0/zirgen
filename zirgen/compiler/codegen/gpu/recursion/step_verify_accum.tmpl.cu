// This code is automatically generated

#include "fp.h"
#include "fpext.h"

namespace risc0::circuit::recursion {

__global__ void step_verify_accum(const Fp* ctrl,
                                  const Fp* data,
                                  const Fp* mix,
                                  {{pools}},
                                  Fp* accum,
                                  uint32_t steps,
                                  uint32_t count) {
  uint32_t mask = steps - 1;
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) {
    return;
  }
{{#body}}
  {{.}}
{{/body}}
}

} // namespace risc0::circuit::recursion
