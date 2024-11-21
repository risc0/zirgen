// This code is automatically generated

#include "supra/fp.h"

#include <cstdint>

{{#decls}}
extern __device__ FpExt {{fn}}(uint32_t idx, uint32_t size{{args}});
{{/decls}}

constexpr size_t INV_RATE = 4;
extern __constant__ FpExt poly_mix[{{num_mix_powers}}];

{{#funcs}}
__device__ FpExt {{fn}}(uint32_t idx,
                         uint32_t size
                         {{args}}) {
  uint32_t mask = size - 1;
{{#block}}
  {{.}}
{{/block}}
}
{{/funcs}}
