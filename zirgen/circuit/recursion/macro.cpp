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

#include "zirgen/circuit/recursion/macro.h"

namespace zirgen::recursion {

WomInitWrapperImpl::WomInitWrapperImpl(WomHeader header) : init(header) {}

void WomInitWrapperImpl::set(MacroInst inst, Val writeAddr) {
  XLOG("WOM_INIT");
}

WomFiniWrapperImpl::WomFiniWrapperImpl(WomHeader header) : fini(header) {}

void WomFiniWrapperImpl::set(MacroInst inst, Val writeAddr) {
  XLOG("WOM_FINI");
  fini->element->setFini(writeAddr);
}

SetGlobalImpl::SetGlobalImpl(WomHeader header) : body(Label("wom_body"), header, 4, 3) {
  for (size_t i = 0; i < kOutSize; i++) {
    outRegs.emplace_back(Label("out", i), "out");
  }
}

void SetGlobalImpl::set(MacroInst inst, Val writeAddr) {
  // Offset to write in multiples of 1/2 digest, since we don't have
  // enough registers to write a full digest at once.
  XLOG("SET_GLOBAL, writing to digest %u//2, part %u%%2", inst->operands[1], inst->operands[1]);
  select->set(inst->operands[1]);

  for (size_t index = 0; index != kOutDigests * 2; ++index) {
    IF(select->at(index)) {
      for (size_t i = 0; i < kDigestWords / 2; ++i) {
        auto vals = regs[i]->doRead(inst->operands[0] + i);
        for (size_t j = 0; j < 2; ++j) {
          XLOG("SET_GLOBAL(%u + %u, %u) -> %x, %e", index, i, j, vals[j], vals);
          outRegs[index * 8 + i * 2 + j]->set(vals[j]);
        }
      }
    }
  }
}

MacroOpImpl::MacroOpImpl(Code code, WomHeader header)
    : mux(Label("mux"),
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
          code->inst->at<size_t(OpType::MACRO)>()->opcode,
          header) {}

void MacroOpImpl::set(Code inst, Val writeAddr) {
  mux->doMux([&](auto inner) { inner->set(inst->inst->at<size_t(OpType::MACRO)>(), writeAddr); });
}

} // namespace zirgen::recursion
