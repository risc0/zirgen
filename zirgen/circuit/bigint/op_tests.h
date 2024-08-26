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

#include <memory>

#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt {

void makeConstAddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstAddAltTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstMulTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeAddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstOneTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstTwoByteTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeSubTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeMulTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeReduceTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeNondetInvTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);

} // namespace zirgen::BigInt
