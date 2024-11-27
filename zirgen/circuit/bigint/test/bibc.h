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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APInt.h"
#include <gtest/gtest.h>
#include <vector>

namespace zirgen::BigInt::test {

using ZType = std::array<uint32_t, 4>;

struct BibcTest : public testing::Test {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::MLIRContext* ctx;
  mlir::ModuleOp module;

  BibcTest();
  mlir::func::FuncOp makeFunc(std::string name, mlir::OpBuilder& builder);
  mlir::func::FuncOp recycle(mlir::func::FuncOp inFunc);
  void lower();
  void AB(mlir::func::FuncOp func, llvm::ArrayRef<llvm::APInt> inputs, ZType& A, ZType& B);
};

std::vector<llvm::APInt> apints(std::vector<std::string> args);

} // namespace zirgen::BigInt::test
