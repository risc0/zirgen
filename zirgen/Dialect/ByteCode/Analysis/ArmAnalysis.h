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

#pragma once

#include "mlir/IR/Block.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"

namespace zirgen::ByteCode {

struct ArmInfo {
  mlir::LocationAttr loc;

  llvm::SmallVector<llvm::ArrayRef<mlir::Operation*>> allOps;

  // Returns a representative sample of the operations present
  llvm::ArrayRef<mlir::Operation*> getOps() const {
    if (allOps.empty())
      return {};
    else
      return allOps.front();
  }

  // Number of times we've seen this set of operations.
  size_t getCount() const { return allOps.size(); }

  // For each operation in this arm, the number of integer arguments
  // (from getByteCodeIntArgs) that need to be decoded.  This does not
  // include temporary values that are listed as operation operands.
  llvm::SmallVector<size_t, 1> opIntArgs;

  // Arguments needed from outside for this arm, i.e. function
  // arguments that aren't stored in temporary storage.
  llvm::SmallVector<mlir::StringAttr, 1> funcArgNames;

  // Temporary values that need to be loaded in for this arm
  size_t numLoadVals = 0;

  // Temporary values that need to be stored after evaluting this arm
  size_t numYieldVals = 0;

  // All values that are used or produced by/from this arm.
  // In order, these are:
  //   function args (funcArgNames)
  //   temporary values to load (numLoadVals)
  //   results from each operation in `ops`
  llvm::SmallVector<mlir::Value> values;

  llvm::ArrayRef<mlir::Value> getFuncArgValues() const {
    return llvm::ArrayRef(values).slice(0, funcArgNames.size());
  }
  mlir::TypeRange getFuncArgTypes() const {
    return mlir::TypeRange(llvm::ArrayRef(values).slice(0, funcArgNames.size()));
  }
  llvm::ArrayRef<mlir::Value> getLoadVals() const {
    return llvm::ArrayRef(values).slice(funcArgNames.size(), numLoadVals);
  }
  llvm::ArrayRef<size_t> getYieldOffsets() const {
    assert(numYieldVals <= valueOffsets.size());
    return llvm::ArrayRef(valueOffsets).slice(valueOffsets.size() - numYieldVals);
  }

  // Value offsets of operands for each operation in this arm in
  // order, followed by value offsets for all yielded values.
  // Value offsets are offsets into `values`.
  llvm::SmallVector<size_t> valueOffsets;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ArmInfo& armInfo);

class ArmAnalysis {
public:
  // Create a new analysis rooted at topOp.
  ArmAnalysis(mlir::Operation* topOp);
  ArmAnalysis(const ArmAnalysis&) = delete;
  void operator=(const ArmAnalysis&) = delete;

  // Analyzes the given set of operations.
  ArmInfo getArmInfo(llvm::ArrayRef<mlir::Operation*> ops);

  // Returns all distinct operations that necessitate being in separate arms.
  llvm::ArrayRef<ArmInfo> getDistinctOps() const { return distinctOps; };

  // Returns all multi-operation arm candidates found which occur at
  // least kMinArmUseCount times.
  llvm::ArrayRef<ArmInfo> getMultiOpArms() const { return multiOpArms; }

  // Return the input and output types
  mlir::FunctionType getFunctionType() const { return funcType; }

  llvm::ArrayRef<mlir::StringAttr> getArgNames() const { return argNames; }

private:
  void calcDispatchKey(mlir::Operation* op);

  std::vector<ArmInfo> distinctOps;
  std::vector<ArmInfo> multiOpArms;

  std::vector<std::vector<mlir::Operation*>> blockOpStorage;

  mlir::FunctionType funcType;
  llvm::SmallVector<mlir::StringAttr> argNames;

  friend struct ArmAnalysisImpl;
};

} // namespace zirgen::ByteCode
