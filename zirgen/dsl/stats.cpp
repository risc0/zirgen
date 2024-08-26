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

#include "zirgen/dsl/stats.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"
#include "zirgen/Dialect/ZStruct/Analysis/DegreeAnalysis.h"
#include "zirgen/Dialect/Zll/Analysis/TapsAnalysis.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace zirgen::dsl {

namespace {

struct StatsPrinter {
  StatsPrinter(ModuleOp moduleOp) : moduleOp(moduleOp) {}

  // Print the sizes of all buffers.
  void printBufferStats() {
    auto analysis = ZStruct::BufferAnalysis(moduleOp);
    llvm::outs() << "buffers:\n";
    for (auto buffer : analysis.getAllBuffers()) {
      llvm::outs() << "  - " << buffer.name << "\n";
      llvm::outs() << "    - size: " << buffer.regCount << "\n"
                   << "    - kind: " << buffer.kind << "\n"
                   << "    - per cycle: " << (buffer.global ? "false" : "true") << "\n";
    }
    llvm::outs() << "\n";
  }

  // Print information about the tap set for the whole circuit
  void printTapStats() {
    auto tapsGlob = moduleOp.lookupSymbol<ZStruct::GlobalConstOp>(Zhlt::getTapsConstName());
    assert(tapsGlob && "Taps global not found");

    Zll::TapsAnalysis tapsAnalysis(
        moduleOp.getContext(),
        llvm::to_vector(
            llvm::cast<mlir::ArrayAttr>(tapsGlob.getConstant()).getAsRange<Zll::TapAttr>()));
    Zll::TapSet tapSet = tapsAnalysis.getTapSet();

    llvm::outs() << "taps:\n";
    llvm::outs() << "  - total tap count: " << tapSet.tapCount << "\n";
    llvm::outs() << "  - tap combos:\n";
    for (auto combo : tapSet.combos) {
      llvm::outs() << "    - " << combo.combo << ": [";
      llvm::interleaveComma(combo.backs, llvm::outs());
      llvm::outs() << "]\n";
    }
    llvm::outs() << "\n";
  }

  // Print information about constraints in the circuit
  void printConstraintStats() {
    DataFlowSolver solver;
    solver.load<ZStruct::DegreeAnalysis>();
    if (failed(solver.initializeAndRun(moduleOp)))
      return;

    moduleOp.walk([&](Zhlt::CheckFuncOp check) {
      llvm::outs() << "constraints in check function:\n";
      size_t totalConstraints = 0;
      llvm::SmallVector<size_t, 6> constraintsPerDegree(6, 0);

      check.walk([&](Zll::EqualZeroOp constraint) {
        size_t degree = solver.lookupState<Zll::DegreeLattice>(constraint)->getValue().get();
        if (degree >= constraintsPerDegree.size())
          constraintsPerDegree.append(degree - constraintsPerDegree.size() + 1, 0);
        constraintsPerDegree[degree]++;
        totalConstraints++;
      });
      for (size_t i = 0; i < constraintsPerDegree.size(); i++)
        llvm::outs() << "  - degree " << i << " constraints: " << constraintsPerDegree[i] << "\n";
      llvm::outs() << "  - total constraints: " << totalConstraints << "\n";
    });
    llvm::outs() << "\n";
  }

  // Print statistics on evaluating the validity polynomial.  This
  // analyzes ValidityTapsFunc, which as of this writing has all its
  // callees inlined, and CSE performed.  This allows us to count
  // operations as a proxy for evaluation speed.
  void printValidityStats() {
    moduleOp.walk([&](Zhlt::ValidityTapsFuncOp op) { printOpCounts(op); });
  }

private:
  void printOpCounts(FunctionOpInterface funcOp);

  ModuleOp moduleOp;
};

void StatsPrinter::printOpCounts(FunctionOpInterface funcOp) {
  DenseMap<StringAttr, /*count=*/size_t> opCounts;
  StringAttr polyOpName = StringAttr::get(moduleOp.getContext(), "PolyOp (interface)");
  size_t tot_count = 0;
  funcOp.walk([&](Operation* op) {
    if (op == funcOp)
      return;

    ++tot_count;

    if (llvm::isa<Zll::PolyOp>(op)) {
      ++opCounts[polyOpName];
      ++opCounts[StringAttr::get(moduleOp.getContext(),
                                 op->getName().getStringRef() + " (PolyOp)")];
    } else {
      ++opCounts[op->getName().getIdentifier()];
    }
  });

  if (!tot_count)
    return;

  StringAttr totalName = StringAttr::get(moduleOp.getContext(), "(total)");
  opCounts[totalName] = tot_count;

  llvm::SmallVector<StringAttr> ops = llvm::to_vector(llvm::make_first_range(opCounts));
  llvm::sort(ops, [&](auto a, auto b) { return opCounts.at(a) > opCounts.at(b); });

  llvm::outs() << "Operation counts in " << funcOp->getName().getStringRef() << " "
               << SymbolTable::getSymbolName(funcOp) << ":\n";
  for (auto& sortedOp : ops) {
    llvm::outs() << llvm::formatv("{0,-30} {1,+6}   {2,+7:p}",
                                  sortedOp.strref(),
                                  opCounts.at(sortedOp),
                                  opCounts.at(sortedOp) * 1. / tot_count)
                 << "\n";
  }
  llvm::outs() << "\n";
}

} // namespace

void printStats(mlir::ModuleOp moduleOp) {
  StatsPrinter printer(moduleOp);
  printer.printBufferStats();
  printer.printTapStats();
  printer.printConstraintStats();
  printer.printValidityStats();
}

} // namespace zirgen::dsl
