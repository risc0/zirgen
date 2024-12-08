// This code is automatically generated

#include "supra/fp.h"

{{#defs}}
#include "eval_check.cuh"
{{/defs}}

#include <cstdint>

namespace {{cppNamespace}}::cuda {

{{#decls}}

{{#declFuncs}}
extern __device__ FpExt {{fn}}(uint32_t idx, uint32_t size{{args}});
{{/declFuncs}}

constexpr size_t INV_RATE = 4;
constexpr size_t kNumPolyMixPows = {{num_mix_powers}};
extern __constant__ FpExt poly_mix[kNumPolyMixPows];

{{/decls}}

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

}  // namespace {{cppNamespace}}::cuda
