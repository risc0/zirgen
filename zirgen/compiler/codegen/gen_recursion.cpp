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

#include "zirgen/compiler/codegen/Passes.h"
#include "zirgen/compiler/codegen/codegen.h"

#include <fstream>

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/circuit/recursion/encode.h"

using namespace mlir;

namespace zirgen {

#define GEN_PASS_DEF_EMITRECURSION
#include "zirgen/compiler/codegen/Passes.h.inc"

namespace {

class EmitRecursionPass : public impl::EmitRecursionBase<EmitRecursionPass> {
public:
  EmitRecursionPass() = default;
  EmitRecursionPass(StringRef dir) { this->outputDir = dir.str(); }
  void runOnOperation() override {
    recursion::EncodeStats stats;
    emitRecursion(outputDir, getOperation(), &stats);
  }
};

std::unique_ptr<llvm::raw_fd_ostream> openOutputFile(const std::string& path,
                                                     const std::string& name) {
  std::string filename = path + "/" + name;
  std::error_code ec;
  auto ofs = std::make_unique<llvm::raw_fd_ostream>(filename, ec);
  if (ec) {
    throw std::runtime_error("Unable to open file: " + filename);
  }
  return ofs;
}

} // namespace

void emitRecursion(const std::string& path, func::FuncOp func, recursion::EncodeStats* stats) {
  std::string name = func.getName().str();
  llvm::errs() << name << "\n";
  auto ofs = openOutputFile(path, name + ".zkr");

  llvm::DenseMap<Value, uint64_t> toId;
  auto encoded = recursion::encode(recursion::HashType::POSEIDON2, &func.front(), &toId, stats);
  ofs->write(reinterpret_cast<const char*>(encoded.data()), encoded.size() * sizeof(uint32_t));

  std::map<uint64_t, mlir::Location> locs;
  for (const auto& elem : toId) {
    locs.emplace(elem.second, elem.first.getLoc());
  }
  auto debugOfs = openOutputFile(path, name + ".zkr.dbg");
  for (const auto& elem : locs) {
    *debugOfs << elem.first << " <- " << elem.second << "\n";
  }
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createEmitRecursionPass(llvm::StringRef dir) {
  return std::make_unique<EmitRecursionPass>(dir);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createEmitRecursionPass() {
  return std::make_unique<EmitRecursionPass>();
}

} // namespace zirgen
