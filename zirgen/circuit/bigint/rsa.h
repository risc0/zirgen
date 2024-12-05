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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"

namespace zirgen::BigInt {

void genModPow65537(mlir::OpBuilder& builder, mlir::Location loc, size_t bitwidth);
// TODO: Unify our tests so we don't need separate codepaths for the RSA versions with & without
// Loads & Stores
void makeRSAChecker(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
llvm::APInt RSA(llvm::APInt N, llvm::APInt S);

} // namespace zirgen::BigInt
