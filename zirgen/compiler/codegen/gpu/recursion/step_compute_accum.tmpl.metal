// This code is automatically generated

#include <metal_stdlib>

#include "fp.h"
#include "fpext.h"

using namespace metal;

kernel void step_compute_accum(uint cycle [[thread_position_in_grid]],
                               const device uint& steps,
                               const device Fp* ctrl,
                               const device Fp* data,
                               const device Fp* mix,
                               {{pools}}) {
    uint mask = steps - 1;
{{#body}}
    {{.}}
{{/body}}
}
