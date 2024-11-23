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

use risc0_zkp::{
    adapter::{CircuitCoreDef, TapsProvider},
    field::baby_bear::BabyBear,
    taps::TapSet,
};

pub(crate) mod info;
pub(crate) mod poly_ext;
pub(crate) mod taps;

pub(crate) struct CircuitImpl;

#[allow(unused)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
pub(crate) mod circuit {
    risc0_zirgen_dsl::zirgen_preamble! {}

    include! {"types.rs.inc"}
    include! {"defs.rs.inc"}
    include! {"layout.rs.inc"}
}

impl CircuitCoreDef<BabyBear> for CircuitImpl {}

impl TapsProvider for CircuitImpl {
    fn get_taps(&self) -> &'static TapSet<'static> {
        self::taps::TAPSET
    }
}
