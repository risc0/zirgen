// Copyright 2024 RISC Zero, Inc.
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

use risc0_zkp::{
    adapter::{MixState, PolyExt, PolyExtStep, PolyExtStepDef},
    field::baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
};

use super::CircuitImpl;

#[allow(missing_docs)]
#[rustfmt::skip]
pub const DEF: PolyExtStepDef = PolyExtStepDef {
    block: &[PolyExtStep::Const(0), // loc(unknown)
PolyExtStep::Const(1), // loc(unknown)
PolyExtStep::Shift, // loc(unknown)
PolyExtStep::True, // loc(unknown)
PolyExtStep::GetGlobal(0, 0), // loc(callsite(unknown at callsite("Reg"("<preamble>":4:21) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":20:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown)))))
PolyExtStep::Sub(0, 3), // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":20:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
PolyExtStep::AndEqz(0, 4), // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":20:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
PolyExtStep::Get(9), // loc(callsite(unknown at callsite("Top"("zirgen/dsl/test/simple-accum.zir":21:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
PolyExtStep::Sub(1, 5), // loc(callsite(unknown at callsite("Top"("zirgen/dsl/test/simple-accum.zir":22:14) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
PolyExtStep::Get(10), // loc(callsite(unknown at callsite("Reg"("<preamble>":4:21) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":22:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown)))))
PolyExtStep::Sub(6, 7), // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":22:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
PolyExtStep::AndEqz(1, 8), // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":22:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
PolyExtStep::GetGlobal(1, 3), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::GetGlobal(1, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(9, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(10, 11), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::GetGlobal(1, 1), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(12, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(13, 14), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::GetGlobal(1, 0), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(15, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(16, 17), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::GetGlobal(1, 7), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::GetGlobal(1, 6), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(19, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(20, 21), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::GetGlobal(1, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(22, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(23, 24), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::GetGlobal(1, 4), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(25, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(26, 27), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(7), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(29, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(30, 31), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(3), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(32, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(33, 34), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(1), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(35, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(36, 37), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(11), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":232:47 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(12), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":141:56 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(18, 40), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":142:36 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(41, 28), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":234:47 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(6), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(4), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(43, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(44, 45), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(46, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(47, 48), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Get(0), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(49, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Add(50, 51), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Sub(52, 38), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":173:42 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Mul(53, 42), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":174:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::Sub(54, 39), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":176:42 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::AndEqz(0, 55), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":178:33 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::AndCond(2, 5, 3), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":439:9 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
PolyExtStep::AndCond(4, 7, 3), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":439:9 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
],
    ret: 5,
};

impl PolyExt<BabyBear> for CircuitImpl {
    fn poly_ext(
        &self,
        mix: &BabyBearExtElem,
        u: &[BabyBearExtElem],
        args: &[&[BabyBearElem]],
    ) -> MixState<BabyBearExtElem> {
        DEF.step::<BabyBear>(mix, u, args)
    }
}
