// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/DenseMap.h"

namespace zirgen {
namespace layout {

using TypeMap = llvm::DenseMap<mlir::Type, mlir::Type>;
mlir::LogicalResult convert(mlir::Operation*, TypeMap&);

} // namespace layout
} // namespace zirgen
