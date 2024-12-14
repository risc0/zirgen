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

#include "zirgen/circuit/bigint/rsa.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt {

void genAdd128(mlir::OpBuilder& builder, mlir::Location loc) {
    auto lhs = builder.create<BigInt::LoadOp>(loc, 128, 11, 0);
    auto rhs = builder.create<BigInt::LoadOp>(loc, 128, 12, 0);
    auto sum = builder.create<BigInt::AddOp>(loc, lhs, rhs);
    builder.create<BigInt::StoreOp>(loc, sum, 13, 0);
}

} // namespace zirgen::BigInt
