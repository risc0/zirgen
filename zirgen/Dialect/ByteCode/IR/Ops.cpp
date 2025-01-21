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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"

using namespace mlir;

namespace zirgen::ByteCode {

void TestOp::getByteCodeIntArgs(llvm::SmallVectorImpl<size_t>& args) {
  llvm::append_range(args, getIntArgs());
}

DispatchKeyAttr ExecuteOp::getArmDispatchKey(size_t size) {
  auto& arm = getArms()[size];
  return TypeSwitch<Operation*, DispatchKeyAttr>(arm.front().getTerminator())
      .Case<YieldOp, ExitOp>([&](auto op) { return op.getDispatchKey(); })
      .Default(
          [](auto) -> DispatchKeyAttr { assert(false && "Invalid terminator inside ExecuteOp"); });
}

} // namespace zirgen::ByteCode
