// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <vector>

#include "zirgen/Dialect/Zll/IR/IR.h"

namespace zirgen::snark {

// THIS CODE IS FOR TESTING ONLY AND RUNS 'system'!!!
std::vector<uint64_t>
run_circom(mlir::func::FuncOp func, const std::vector<uint32_t>& iop, const std::string& tmp_path);

} // namespace zirgen::snark
