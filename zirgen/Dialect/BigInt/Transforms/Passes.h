// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/IOP/IR/IR.h"

namespace zirgen::BigInt {

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerReducePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerZllPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zirgen/Dialect/BigInt/Transforms/Passes.h.inc"

} // namespace zirgen::BigInt
