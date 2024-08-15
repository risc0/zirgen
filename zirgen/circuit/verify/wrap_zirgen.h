// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/circuit/verify/verify.h"

namespace zirgen::verify {

std::unique_ptr<CircuitInterface> getInterfaceZirgen(mlir::ModuleOp zhltModule,
                                                     ProtocolInfo protocolInfo);
std::unique_ptr<CircuitInterface>
getInterfaceZirgen(mlir::MLIRContext* ctx, mlir::StringRef filename, ProtocolInfo protocolInfo);

} // namespace zirgen::verify
