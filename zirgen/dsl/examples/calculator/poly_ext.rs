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
    block: &[PolyExtStep::Const(5), // loc(unknown)
PolyExtStep::Const(1), // loc(unknown)
PolyExtStep::Const(0), // loc(unknown)
PolyExtStep::True, // loc(unknown)
PolyExtStep::Get(2), // loc(callsite(unknown at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :40:19) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Get(3), // loc(callsite(unknown at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :41:21) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Get(4), // loc(callsite(unknown at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :42:21) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Get(5), // loc(callsite(unknown at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :43:25) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Get(6), // loc(callsite(unknown at callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :13:36) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::Get(7), // loc(callsite(unknown at callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :13:36) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::Sub(1, 7), // loc(callsite(unknown at callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :15:28) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::Mul(7, 9), // loc(callsite(unknown at callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :15:21) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::AndEqz(0, 10), // loc(callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :15:37) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Sub(1, 8), // loc(callsite(unknown at callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :15:28) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::Mul(8, 11), // loc(callsite(unknown at callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :15:21) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::AndEqz(1, 12), // loc(callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :15:37) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Add(7, 8), // loc(callsite(unknown at callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :17:4) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::Sub(13, 1), // loc(callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :17:32) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::AndEqz(2, 14), // loc(callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :17:32) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Sub(8, 3), // loc(callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :19:56) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::AndEqz(3, 15), // loc(callsite( OneHot ( zirgen/dsl/examples/calculator/calculator.zir :19:56) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:20) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Add(4, 5), // loc(callsite(unknown at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :46:10) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Get(8), // loc(callsite(unknown at callsite( Reg ( <preamble> :4:21) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :46:9) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::Sub(16, 17), // loc(callsite( Reg ( <preamble> :5:7) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :46:9) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::AndEqz(0, 18), // loc(callsite( Reg ( <preamble> :5:7) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :46:9) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::AndCond(4, 7, 5), // loc(callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:25) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))
PolyExtStep::Sub(4, 5), // loc(callsite(unknown at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :48:10) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::Sub(19, 17), // loc(callsite( Reg ( <preamble> :5:7) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :48:9) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::AndEqz(0, 20), // loc(callsite( Reg ( <preamble> :5:7) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :48:9) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))
PolyExtStep::AndCond(6, 8, 7), // loc(callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :44:25) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))
PolyExtStep::AndEqz(8, 2), // loc(callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :52:13) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))
PolyExtStep::Sub(17, 6), // loc(callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :53:11) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))
PolyExtStep::AndEqz(9, 21), // loc(callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :53:11) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))
PolyExtStep::GetGlobal(0, 0), // loc(callsite(unknown at callsite( Reg ( <preamble> :4:21) at callsite( SetGlobalResult ( zirgen/dsl/examples/calculator/calculator.zir :24:18) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :54:19) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown))))))
PolyExtStep::Sub(0, 22), // loc(callsite( Reg ( <preamble> :5:7) at callsite( SetGlobalResult ( zirgen/dsl/examples/calculator/calculator.zir :24:18) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :54:19) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
PolyExtStep::AndEqz(10, 23), // loc(callsite( Reg ( <preamble> :5:7) at callsite( SetGlobalResult ( zirgen/dsl/examples/calculator/calculator.zir :24:18) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :54:19) at callsite( Top ( zirgen/dsl/examples/calculator/calculator.zir :39:2) at unknown)))))
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
