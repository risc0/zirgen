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

#include "zirgen/Dialect/Zll/Analysis/TapsAnalysis.h"

#include <set>

namespace zirgen::Zll {

using namespace mlir;

struct TapAttrOrder {
  bool operator()(TapAttr a, TapAttr b) const {
    return std::make_tuple(a.getRegGroupId(), a.getOffset(), a.getBack()) <
           std::make_tuple(b.getRegGroupId(), b.getOffset(), b.getBack());
  }
};

TapsAnalysis::TapsAnalysis(Operation* op) : ctx(op->getContext()) {
  tapAttrs = lookupModuleAttr<Zll::TapsAttr>(op).getTaps();

  std::map</*regGroupId=*/unsigned,
           std::map</*offset=*/unsigned,
                    /*backs=*/std::set<unsigned>>>
      backsByReg;

  for (auto tapAttr : tapAttrs) {
    backsByReg[tapAttr.getRegGroupId()][tapAttr.getOffset()].insert(tapAttr.getBack());
  }

  std::set<std::set<unsigned>> backCombos;
  for (auto [regGroup, regs] : backsByReg) {
    (void)regGroup;
    for (auto [reg, backs] : regs) {
      (void)reg;
      backCombos.insert(backs);
    }
  }

  tapSet.tapCount = tapAttrs.size();
  if (tapAttrs.empty()) {
    return;
  }
  for (auto [index, backs] : llvm::enumerate(backCombos)) {
    auto backVec = llvm::to_vector(backs);
    auto comboAttr = comboToAttr(backVec);
    backComboIndex[comboAttr] = index;
    tapSet.combos.push_back(
        Combo{.combo = unsigned(index), .backs = std::vector(backVec.begin(), backVec.end())});
  }

  unsigned tapPos = 0;
  for (auto [regGroupId, regGroup] : backsByReg) {
    // Fill in any empty groups
    while (tapSet.groups.size() < regGroupId)
      tapSet.groups.emplace_back();

    auto& group = tapSet.groups.emplace_back();
    for (auto [reg, backs] : regGroup) {
      std::vector<unsigned> backsVec(backs.begin(), backs.end());
      group.regs.push_back(Reg{
          .offset = reg, .combo = getComboIndex(backsVec), .tapPos = tapPos, .backs = backsVec});
      tapPos += backsVec.size();
    }
  }
  assert(tapPos == tapAttrs.size());

  for (auto [idx, tapAttr] : llvm::enumerate(tapAttrs)) {
    tapIndex[tapAttr] = idx;
  }
}

mlir::ArrayAttr TapsAnalysis::comboToAttr(llvm::ArrayRef<unsigned> backs) {
  Builder builder(ctx);
  auto backAttrs = llvm::to_vector(llvm::map_range(
      backs, [&](uint32_t back) -> mlir::Attribute { return builder.getUI32IntegerAttr(back); }));

  return builder.getArrayAttr(backAttrs);
}

} // namespace zirgen::Zll
