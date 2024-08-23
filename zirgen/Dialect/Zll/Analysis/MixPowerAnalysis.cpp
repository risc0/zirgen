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

#include "zirgen/Dialect/Zll/Analysis/MixPowerAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace zirgen {

MixPowAnalysis::MixPowAnalysis(Operation* funcOp) {
  auto* terminator = funcOp->getRegion(0).front().getTerminator();
  Value terminatorValue = terminator->getOperand(0);

  processChain(terminatorValue, 0);

  assert(powsNeeded.empty());
  llvm::append_range(powsNeeded, llvm::make_second_range(mixPows));
  llvm::sort(powsNeeded);
  powsNeeded.erase(llvm::unique(powsNeeded), powsNeeded.end());

  auto indexed = llvm::map_range(llvm::enumerate(powsNeeded), [](auto elem) {
    auto [idx, pow] = elem;
    return std::make_pair(pow, idx);
  });
  mixPowIndex.insert(indexed.begin(), indexed.end());
}

size_t MixPowAnalysis::processChain(Value val, size_t offset) {
  SmallVector<Operation*> ops;

  while (val) {
    Operation* op = val.getDefiningOp();
    TypeSwitch<Operation*>(op)
        .Case<AndEqzOp, AndCondOp>([&](auto op) {
          ops.push_back(op);
          val = op.getIn();
        })
        .Case<TrueOp>([&](auto) { val = Value(); })
        .Default([&](auto) {
          llvm::errs() << "Unexpected operation in constraint chain: " << *op << "\n";
          assert(false);
        });
  }

  for (Operation* op : llvm::reverse(ops)) {
    TypeSwitch<Operation*>(op)
        .Case<AndEqzOp>([&](auto op) {
          auto [it, didInsert] = mixPows.try_emplace(op, offset);
          if (!didInsert) {
            if (it->second != offset) {
              llvm::errs() << "Offset mismatch: " << offset << " vs " << it->second << " on " << op
                           << "\n";
              assert(false);
            }
          }
          ++offset;
        })
        .Case<AndCondOp>([&](auto op) {
          size_t innerOffset = processChain(op.getInner(), 0);
          auto [it, didInsert] = mixPows.try_emplace(op, offset);
          if (!didInsert) {
            if (it->second != offset) {
              llvm::errs() << "Offset mismatch: " << offset << " vs " << it->second << " on " << op
                           << "\n";
              assert(false);
            }
          }
          offset += innerOffset;
        });
  }
  return offset;
}

} // namespace zirgen
