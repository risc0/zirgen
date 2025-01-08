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

use risc0_zkp::taps::{TapData, TapSet};

#[allow(missing_docs)]

pub const TAPSET: &TapSet = &TapSet::<'static> {
    taps: &[
        {{#taps}}
        TapData {
            offset: {{offset}},
            back: {{back}},
            group: {{group}},
            combo: {{combo}},
            skip: {{skip}},
        },
        {{/taps}}
    ],
    combo_taps: &[
        {{#combo_taps}} {{.}}, {{/combo_taps}}
    ],
    combo_begin: &[
        {{#combo_begin}} {{.}}, {{/combo_begin}}
    ],
    group_begin: &[
        {{#group_begin}} {{.}}, {{/group_begin}}
    ],
    combos_count: {{combos_count}},
    reg_count: {{reg_count}},
    tot_combo_backs: {{tot_combo_backs}},
    // TODO: Generate these instead of hardcoding:
    group_names: &["accum", "code", "data"],
};
