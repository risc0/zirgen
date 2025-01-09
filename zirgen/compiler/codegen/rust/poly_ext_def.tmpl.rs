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

use risc0_zkp::{
    adapter::{MixState, PolyExt, PolyExtStep, PolyExtStepDef},
    field::baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
};

use super::CircuitImpl;

#[allow(missing_docs)]
#[rustfmt::skip]
pub const DEF: PolyExtStepDef = PolyExtStepDef {
    block: {{block}},
    ret: {{ret}},
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
