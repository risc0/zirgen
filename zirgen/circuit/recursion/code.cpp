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

#include "zirgen/circuit/recursion/code.h"

namespace zirgen::recursion {

MicroInstImpl::MicroInstImpl() : opcode(Label("opcode"), "code") {
  for (size_t i = 0; i < 3; i++) {
    operands.emplace_back(Label("operand", i), "code");
  }
}

MicroInstsImpl::MicroInstsImpl() {
  for (size_t i = 0; i < 3; i++) {
    insts.emplace_back(Label("inst", i));
  }
}

MacroInstImpl::MacroInstImpl()
    : opcode(Label("opcode"),
             Labels({"nop",
                     "wom_init",
                     "wom_fini",
                     "bit_and_elem",
                     "bit_op_shorts",
                     "sha_init",
                     "sha_fini",
                     "sha_load",
                     "sha_mix",
                     "set_global"}),
             "code",
             false) {
  for (size_t i = 0; i < 3; i++) {
    operands.emplace_back(Label("operand", i), "code");
  }
}

Poseidon2MemInstImpl::Poseidon2MemInstImpl()
    : doMont(Label("do_mont"), "code")
    , keepState(Label("keep_state"), "code")
    , keepUpperState(Label("keep_upper_state"), "code")
    , prepFull(Label("prep_full"), "code")
    , group(Label("group"), Labels({"g0", "g1", "g2"}), "code") {
  for (size_t i = 0; i < 8; i++) {
    inputs.emplace_back(Label("inputs", i), "code");
  }
}

Poseidon2FullInstImpl::Poseidon2FullInstImpl()
    : cycle(Label("cycle"), Labels({"c0", "c1", "c2", "c3"}), "code", false) {}

CheckedBytesInstImpl::CheckedBytesInstImpl()
    : evalPoint(Label("eval_point"), "code")
    , keepCoeffs(Label("keep_coeffs"), "code")
    , keepUpperState(Label("keep_upper_state"), "code")
    , prepFull(Label("prep_full"), "code") {}

CodeImpl::CodeImpl()
    : writeAddr(Label("write_addr"), "code")
    , select(Label("select"),
             Labels({"micro_ops",
                     "macro_ops",
                     "poseidon2_load",
                     "poseidon2_full",
                     "poseidon2_partial",
                     "poseidon2_store",
                     "checked_bytes"}),
             "code",
             false)
    , inst(Label("inst"),
           Labels({"micro_ops",
                   "macro_ops",
                   "poseidon2_load",
                   "poseidon2_full",
                   "poseidon2_partial",
                   "poseidon2_store",
                   "checked_bytes"}),
           select) {}

} // namespace zirgen::recursion
