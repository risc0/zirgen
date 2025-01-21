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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"

using namespace mlir;

namespace zirgen::ByteCode {

DispatchKeyAttr getDispatchKey(Operation* op) {
  SmallVector<size_t> intArgs;
  if (auto bcInterface = llvm::dyn_cast<ByteCodeOpInterface>(op)) {
    bcInterface.getByteCodeIntArgs(intArgs);
  }

  auto operandTypes = llvm::to_vector(op->getOperandTypes());
  auto resultTypes = llvm::to_vector(op->getResultTypes());

  SmallVector<mlir::Attribute> intKinds;
  for (auto idx : llvm::seq(intArgs.size())) {
    intKinds.push_back(StringAttr::get(
        op->getContext(), (op->getName().getStringRef() + "_" + std::to_string(idx)).str()));
  }

  SmallVector<size_t> blockArgNums;
  for (Value operand : op->getOperands()) {
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(operand)) {
      blockArgNums.push_back(blockArg.getArgNumber());
    }
  }

  return DispatchKeyAttr::get(op->getContext(),
                              /*operationName=*/op->getName().getStringRef(),
                              operandTypes,
                              resultTypes,
                              intKinds,
                              /*blockArgs=*/blockArgNums);
}

std::string getNameForIntKind(mlir::Attribute intKind) {
  if (auto strAttr = llvm::dyn_cast<StringAttr>(intKind)) {
    return strAttr.str();
  }
  if (auto unitAttr = llvm::dyn_cast<UnitAttr>(intKind)) {
    return "unit";
  }
  std::string str;
  llvm::raw_string_ostream os(str);
  os << intKind;

  llvm::erase_if(str, [](char c) { return c == '"' || c == ' '; });
  return str;
}

} // namespace zirgen::ByteCode
