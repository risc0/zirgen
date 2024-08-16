// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/dsl/ast.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace zirgen {
namespace dsl {

std::optional<mlir::ModuleOp> lower(mlir::MLIRContext&, const llvm::SourceMgr&, ast::Module*);

} // namespace dsl
} // namespace zirgen
