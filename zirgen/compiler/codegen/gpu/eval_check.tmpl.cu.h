// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const char cu_eval_check_tmpl[] = R"(
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
)";
