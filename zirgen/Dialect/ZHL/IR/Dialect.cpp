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

#include "zirgen/Dialect/ZHL/IR/ZHL.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHL/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/ZHL/IR/Types.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/ZHL/IR/Attrs.cpp.inc"

using namespace mlir;

namespace zirgen::Zhl {

void ZhlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/ZHL/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zirgen/Dialect/ZHL/IR/Types.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "zirgen/Dialect/ZHL/IR/Attrs.cpp.inc"
      >();
}

} // namespace zirgen::Zhl
