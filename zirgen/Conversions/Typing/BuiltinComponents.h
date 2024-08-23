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

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/SourceMgr.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"

namespace zirgen::Typing {

// Return the preamble ZIR code to be parsed before and user-supplied files.
llvm::StringRef getBuiltinPreamble();

// Add non-preamble builtins using the given builder
void addBuiltins(mlir::OpBuilder& builder);

// Adds a zhlt ComponentOp for an array type which forwards arguments to the
// element constructor.
void addArrayCtor(mlir::OpBuilder& builder,
                  llvm::StringRef mangledName,
                  ZStruct::ArrayType arrayType,
                  ZStruct::LayoutArrayType layoutType,
                  mlir::TypeRange ctorParams);

} // namespace zirgen::Typing
