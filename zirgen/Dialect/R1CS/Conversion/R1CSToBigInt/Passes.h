// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zirgen::R1CSToBigInt {

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createR1CSToBigIntPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zirgen/Dialect/R1CS/Conversion/R1CSToBigInt/Passes.h.inc"

} // namespace zirgen::R1CSToBigInt
