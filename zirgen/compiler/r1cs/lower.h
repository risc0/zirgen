// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <memory>

#include "zirgen/compiler/r1cs/r1csfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace zirgen::R1CS {

std::optional<mlir::ModuleOp> lower(mlir::MLIRContext&, r1csfile::System&);

} // namespace zirgen::R1CS
