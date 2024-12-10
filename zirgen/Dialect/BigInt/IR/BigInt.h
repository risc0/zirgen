// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "risc0/core/util.h"

namespace zirgen::BigInt {

constexpr size_t kBitsPerCoeff = 8;
constexpr size_t kCoeffsPerPoly = 16;

// Get and set the number of bigint claims to be verified in sequence
// by the generated ZKR for a given function
size_t getIterationCount(mlir::func::FuncOp func);
void setIterationCount(mlir::func::FuncOp func, size_t iters);

} // namespace zirgen::BigInt

#include "zirgen/Dialect/Zll/IR/IR.h"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/BigInt/IR/Types.h.inc"

#include "mlir/IR/Dialect.h"

#include "zirgen/Dialect/BigInt/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/BigInt/IR/Ops.h.inc"
