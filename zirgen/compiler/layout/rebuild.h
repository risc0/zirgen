// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/Types.h"
#include "zirgen/compiler/layout/collect.h"

#include "llvm/ADT/DenseMap.h"

namespace zirgen {
namespace layout {

llvm::DenseMap<mlir::Type, mlir::Type> rebuild(Circuit&);

} // namespace layout
} // namespace zirgen
