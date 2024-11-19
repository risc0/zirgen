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

#pragma once

#include "mlir/IR/Operation.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include <vector>

namespace zirgen::Zll {

// Analyzes a set of taps to gather the data we need for ZKP.  This is
// in a format agreeable to the verifier to get compiled to zkr.
//
// TODO: It would be nice to reuse this logic for e.g generating the
// tap data for risc0_zkp::risc0::taps::TapSet. instead of having a
// separate implementation.
class TapsAnalysis {
public:
  struct Reg {
    unsigned offset;
    unsigned combo;
    unsigned tapPos;
    std::vector<unsigned> backs;
  };

  struct Group {
    std::vector<Reg> regs;
  };

  struct Combo {
    unsigned combo;
    std::vector<unsigned> backs;
  };

  struct TapSet {
    std::vector<TapsAnalysis::Group> groups;
    std::vector<TapsAnalysis::Combo> combos;
    unsigned tapCount;
  };

  TapsAnalysis(mlir::Operation* op);

  const TapSet& getTapSet() { return tapSet; }
  llvm::ArrayRef<TapAttr> getTapAttrs() { return tapAttrs; }

  unsigned getTapIndex(unsigned regGroupId, unsigned offset, unsigned back) {
    return getTapIndex(TapAttr::get(ctx, regGroupId, offset, back));
  }
  unsigned getTapIndex(TapAttr tapAttr) { return tapIndex.lookup(tapAttr); }

private:
  mlir::ArrayAttr comboToAttr(llvm::ArrayRef<unsigned> backs);
  unsigned getComboIndex(llvm::ArrayRef<unsigned> backs) {
    return backComboIndex.lookup(comboToAttr(backs));
  }

  mlir::MLIRContext* ctx;

  TapSet tapSet;
  llvm::DenseMap<TapAttr, /*index=*/unsigned> tapIndex;
  llvm::ArrayRef<TapAttr> tapAttrs;
  llvm::SmallVector<mlir::ArrayAttr> tapComboAttrs;
  llvm::DenseMap<mlir::ArrayAttr, /*comboIndex=*/unsigned> backComboIndex;
};

using TapSet = TapsAnalysis::TapSet;

} // namespace zirgen::Zll
