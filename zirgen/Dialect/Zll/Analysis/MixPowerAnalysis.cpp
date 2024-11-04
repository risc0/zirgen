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

size_t MixPowAnalysis::processChain(Value val, size_t offset, SmallVector<func::CallOp> callStack) {
  SmallVector<std::pair<Operation*, SmallVector<func::CallOp>>> ops;

  //  llvm::errs() << "(" << offset << ") Tracing chain\n";
  while (val) {
    //    llvm::errs() << "(" << offset << ")" << val << "\n";
    TypeSwitch<Value>(val)
        .Case<mlir::BlockArgument>([&](BlockArgument blockVal) {
          // Trace back into caller
          //        llvm::errs() << "Block argument on " << *blockVal.getOwner()->getParentOp() <<
          //        "\n";;
          if (callStack.empty()) {
            llvm::errs() << "Can't trace past " << blockVal << "\n";
            assert(false);
          }
          auto callOp = callStack.back();
          //        llvm::errs() << "Popping " << callOp << "\n";
          callStack.pop_back();
          val = callOp->getOperand(blockVal.getArgNumber());
        })
        .Case<OpResult>([&](OpResult resultVal) {
          Operation* op = resultVal.getDefiningOp();
          TypeSwitch<Operation*>(op)
              .Case<AndEqzOp, AndCondOp>([&](auto op) {
                ops.push_back({op, callStack});
                val = op.getIn();
              })
              .Case<func::CallOp>([&](auto op) {
                //            llvm::errs() << "Calling " << op << "\n";
                callStack.push_back(op);
                auto callee = llvm::cast<SymbolRefAttr>(op.getCallableForCallee());
                auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, callee);
                calledFuncs.push_back(funcOp);
                if (!funcOp) {
                  llvm::errs() << "Couldn't find callee " << callee << "\n";
                  assert(false);
                }
                val = funcOp.getBody().front().getTerminator()->getOperand(0);
              })
              .Case<TrueOp>([&](auto) { val = Value(); })
              .Default([&](auto) {
                llvm::errs() << "Unexpected operation in constraint chain: " << *op << "\n";
                assert(false);
              });
        });
  }

  //  llvm::errs() << "(" << offset << ") Processing chain\n";
  for (auto [op, opCallStackA] : llvm::reverse(ops)) {
    SmallVector<func::CallOp> opCallStack = opCallStackA;
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
          size_t innerOffset = processChain(op.getInner(), 0, opCallStack);
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
  //  llvm::errs() << "(" << offset << ") Done chain\n";
  return offset;
}

} // namespace zirgen
