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

const char metal_eval_check_tmpl[] = R"(
// This code is automatically generated

#include <metal_stdlib>

#include "fp.h"
#include "fpext.h"

using namespace metal;

constant size_t INV_RATE = 4;

FpExt poly_fp(uint idx,
            uint size,
            const device Fp* ctrl,
            const device Fp* out,
            const device Fp* data,
            const device Fp* mix,
            const device Fp* accum,
            const device FpExt* poly_mix) {
    uint mask = size - 1;
{{#block}}
    {{.}}
{{/block}}
}

kernel void eval_check(device Fp* check,
                       const device Fp* ctrl,
                       const device Fp* data,
                       const device Fp* accum,
                       const device Fp* mix,
                       const device Fp* out,
                       const device FpExt* poly_mix,
                       const device Fp& rou,
                       const device uint32_t& po2,
                       const device uint32_t& domain,
                       uint cycle [[thread_position_in_grid]]) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum, poly_mix);
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret.elems[0];
    check[domain * 1 + cycle] = ret.elems[1];
    check[domain * 2 + cycle] = ret.elems[2];
    check[domain * 3 + cycle] = ret.elems[3];
}
)";
