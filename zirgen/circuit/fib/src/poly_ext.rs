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
    block: &[PolyExtStep::Const(1), // loc(unknown)
PolyExtStep::True, // zirgen/circuit/fib/fib.cpp:38
PolyExtStep::Get(1), // zirgen/circuit/fib/fib.cpp:20
PolyExtStep::Get(4), // zirgen/circuit/fib/fib.cpp:21
PolyExtStep::Sub(2, 0), // zirgen/circuit/fib/fib.cpp:21
PolyExtStep::AndEqz(0, 3), // zirgen/circuit/fib/fib.cpp:21
PolyExtStep::AndCond(0, 1, 1), // zirgen/circuit/fib/fib.cpp:20
PolyExtStep::Get(2), // zirgen/circuit/fib/fib.cpp:23
PolyExtStep::Get(6), // zirgen/circuit/fib/fib.cpp:24
PolyExtStep::Get(5), // zirgen/circuit/fib/fib.cpp:24
PolyExtStep::Add(6, 5), // zirgen/circuit/fib/fib.cpp:24
PolyExtStep::Sub(2, 7), // zirgen/circuit/fib/fib.cpp:24
PolyExtStep::AndEqz(0, 8), // zirgen/circuit/fib/fib.cpp:24
PolyExtStep::AndCond(2, 4, 3), // zirgen/circuit/fib/fib.cpp:23
PolyExtStep::Get(3), // zirgen/circuit/fib/fib.cpp:26
PolyExtStep::GetGlobal(0, 0), // zirgen/circuit/fib/fib.cpp:28
PolyExtStep::Sub(10, 2), // zirgen/circuit/fib/fib.cpp:28
PolyExtStep::AndEqz(0, 11), // zirgen/circuit/fib/fib.cpp:28
PolyExtStep::AndCond(4, 9, 5), // zirgen/circuit/fib/fib.cpp:26
PolyExtStep::Add(1, 4), // zirgen/circuit/fib/fib.cpp:34
PolyExtStep::Add(12, 9), // zirgen/circuit/fib/fib.cpp:34
PolyExtStep::Get(0), // zirgen/circuit/fib/fib.cpp:35
PolyExtStep::Sub(14, 0), // zirgen/circuit/fib/fib.cpp:35
PolyExtStep::AndEqz(0, 15), // zirgen/circuit/fib/fib.cpp:35
PolyExtStep::AndCond(6, 13, 7), // zirgen/circuit/fib/fib.cpp:34
],
    ret: 8,
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
