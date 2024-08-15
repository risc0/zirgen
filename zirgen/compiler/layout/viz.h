// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/BuiltinOps.h"

#include <iostream>

namespace zirgen {
namespace layout {
namespace viz {

void typeRelation(mlir::Type, std::ostream&);
void storageNest(mlir::Type, std::ostream&);
void layoutSizes(mlir::Type, std::ostream&);
void layoutAttrs(mlir::ModuleOp, std::ostream&);
void columnKeyPaths(mlir::ModuleOp, size_t, std::ostream&);

} // namespace viz
} // namespace layout
} // namespace zirgen
