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

// This code is automatically generated

#include "fp.h"
#include "fpext.h"

#include <cstdint>

constexpr size_t kInvRate = 4;

// clang-format off
namespace risc0::circuit::{{name}} {

{{#decls}}
FpExt {{fn}}(size_t cycle, size_t steps, FpExt* poly_mix{{args}});
{{/decls}}

{{#funcs}}
FpExt {{fn}}(size_t cycle, size_t steps, FpExt* poly_mix{{args}}) {
  size_t mask = steps - 1;
{{#body}}
  {{.}}
{{/body}}
}
{{/funcs}}

} // namespace risc0::circuit::{{name}}
// clang-format on
