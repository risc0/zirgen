// Copyright 2025 RISC Zero, Inc.
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

#include <numeric>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "zirgen/Dialect/ZStruct/IR/Types.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

#include "mlir/IR/Dialect.h"

#include "zirgen/Dialect/ZHLT/IR/Dialect.h.inc"

namespace zirgen::Zhlt {

namespace detail {

template <typename ContainerT>
ContainerT snippetGetIndex(ContainerT container, llvm::ArrayRef<int32_t> segmentSizes, size_t idx) {
  assert(idx < segmentSizes.size());
  size_t start = std::accumulate(segmentSizes.begin(), segmentSizes.begin() + idx, 0);
  size_t len = segmentSizes[idx];
  return container.slice(start, len);
}

inline bool
isValidSegmentSizes(llvm::ArrayRef<int32_t> segmentSizes, ssize_t numRaw, size_t numSegmentSizes) {
  if (segmentSizes.size() != numSegmentSizes)
    return false;
  if (std::accumulate(segmentSizes.begin(), segmentSizes.end(), 0) != numRaw)
    return false;
  return true;
}

} // namespace detail

// Constant names for generated constants.
std::string getTapsConstName();

/// Populates value names based on the `zirgen.argName` attribute.  Intended
/// to be used with getAsmBlockArgumentNames.
void getZirgenBlockArgumentNames(mlir::FunctionOpInterface funcOp,
                                 mlir::Region& r,
                                 mlir::OpAsmSetValueNameFn setNameFn);

} // namespace zirgen::Zhlt

#include "zirgen/Dialect/ZHLT/IR/Interfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Types.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/ComponentOps.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Ops.h.inc"

namespace zirgen::Zhlt {

/// True iff component represents a "starting point" of execution like Top,
/// Accum, or a test, or is marked with the `entry` attribute.
bool isEntryPoint(ComponentOp component);

/// True iff component is backed by a buffer, including all entry points and
/// @mix and @global
bool isBufferComponent(ComponentOp component);

} // namespace zirgen::Zhlt
