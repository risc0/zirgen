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
    block: &[PolyExtStep::Const(11), // loc(unknown)
PolyExtStep::Const(1), // loc(unknown)
PolyExtStep::Const(2), // loc(unknown)
PolyExtStep::Const(3), // loc(unknown)
PolyExtStep::Const(2013265919), // loc(unknown)
PolyExtStep::Shift, // loc(unknown)
PolyExtStep::True, // loc(unknown)
PolyExtStep::GetGlobal(0, 0), // loc(callsite(unknown at callsite("Reg"("<preamble>":4:21) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":17:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown)))))
PolyExtStep::Sub(0, 6), // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":17:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::AndEqz(0, 7), // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":17:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::Get(9), // loc(callsite(unknown at callsite("Top"("zirgen/dsl/test/simple-accum.zir":18:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::Sub(1, 8), // loc(callsite(unknown at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:14) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::Get(10), // loc(callsite(unknown at callsite("Reg"("<preamble>":4:21) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown)))))
PolyExtStep::Sub(9, 10), // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::AndEqz(1, 11), // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::Get(11), // loc(callsite(unknown at callsite("NondetMyArgument"("zirgen/dsl/test/simple-accum.zir":5:18) at callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":10:25) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))))
PolyExtStep::Get(12), // loc(callsite(unknown at callsite("NondetMyArgument"("zirgen/dsl/test/simple-accum.zir":6:18) at callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":10:25) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))))
PolyExtStep::Sub(2, 12), // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":11:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::AndEqz(0, 14), // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":11:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::Sub(3, 13), // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":12:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::AndEqz(3, 15), // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":12:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::AndCond(2, 8, 4), // loc(callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:22) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown)))
PolyExtStep::Sub(4, 12), // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":11:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:54) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::AndEqz(0, 16), // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":11:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:54) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::AndEqz(6, 15), // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":12:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:54) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
PolyExtStep::AndCond(5, 10, 7), // loc(callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:22) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown)))
PolyExtStep::GetGlobal(1, 3), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::GetGlobal(1, 2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(17, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(19, 18), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::GetGlobal(1, 1), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(20, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(22, 21), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::GetGlobal(1, 0), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(23, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(25, 24), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::GetGlobal(1, 7), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::GetGlobal(1, 6), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(27, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(29, 28), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::GetGlobal(1, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(30, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(32, 31), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::GetGlobal(1, 4), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(33, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(35, 34), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Get(7), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Get(5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(37, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(39, 38), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Get(3), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(40, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(42, 41), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Get(1), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(43, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(45, 44), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(26, 13), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":141:36 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(47, 36), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":233:47 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Get(6), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Get(4), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(49, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(51, 50), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Get(2), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(52, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(54, 53), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Get(0), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(55, 5), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Add(57, 56), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Sub(58, 46), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":172:42 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Mul(59, 48), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":173:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::Sub(60, 12), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":175:42 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::AndEqz(0, 61), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":177:33 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::AndCond(8, 8, 9), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":438:9 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
PolyExtStep::AndCond(10, 10, 9), // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":438:9 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
],
    ret: 11,
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
