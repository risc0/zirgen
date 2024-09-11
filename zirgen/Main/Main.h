#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace zirgen {

void registerZirgenCommon();

void registerZirgenDialects(mlir::DialectRegistry& registry);

void addAccumAndGlobalPasses(mlir::PassManager& pm);

void addTypingPasses(mlir::PassManager& pm);

mlir::LogicalResult checkDegreeExceeded(mlir::ModuleOp module, size_t maxDegree);

} // namespace zirgen
