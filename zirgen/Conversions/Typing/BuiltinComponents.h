// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
