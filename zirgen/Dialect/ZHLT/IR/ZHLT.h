// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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

} // namespace zirgen::Zhlt

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Types.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/ComponentOps.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Ops.h.inc"
