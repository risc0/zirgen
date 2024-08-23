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

#include <numeric>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "zirgen/Dialect/ZStruct/IR/TypeUtils.h"
#include "zirgen/Dialect/ZStruct/IR/Types.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

#include "mlir/IR/Dialect.h"

#include "zirgen/Dialect/ZHLT/IR/Dialect.h.inc"

namespace zirgen::Zhlt {

class ComponentTypeAttr;

// ComponentManager manages the component collection when generating
// typed components.  Callers can request a component and need not
// know whether the component is built in or generated from a
// zhl.component, or has some other special handling.
class ComponentManager {
public:
  // Require that the given component exists and is a candidate for construction.
  virtual mlir::LogicalResult requireComponent(mlir::Location loc, ComponentTypeAttr name) = 0;

  // Require that the given component exists and is a candidate for
  // construction.  Provides construction arguments in case they can
  // be used for type inferrence; if type inferrence is performed, the
  // provided type name will be updated.
  virtual mlir::LogicalResult requireComponentInferringType(mlir::Location loc,
                                                            ComponentTypeAttr& name,
                                                            mlir::ValueRange constructArgs) = 0;

  // Require that the given component exists and is a candidate for either construction or
  // specialization.
  virtual mlir::LogicalResult requireAbstractComponent(mlir::Location loc,
                                                       ComponentTypeAttr name) = 0;

  // Specializes a compenent type, if possible.
  virtual ComponentTypeAttr specialize(mlir::Location loc,
                                       ComponentTypeAttr orig,
                                       llvm::ArrayRef<mlir::Attribute> typeArgs) = 0;

  // Returns the layout type, if the component exists and has a layout, or null.  The component must
  // exist.
  virtual mlir::Type getLayoutType(ComponentTypeAttr component) = 0;

  // Returns the value type returned by the given component's constructor.  The component must
  // exist.
  virtual mlir::Type getValueType(ComponentTypeAttr component) = 0;

  // Builds a construction of the given component, and returns its value.  The component must exist.
  // Upon failure, emits an error and returns null.
  virtual mlir::Value buildConstruct(mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     ComponentTypeAttr component,
                                     mlir::ValueRange constructArgs,
                                     mlir::Value layout) = 0;

  // Builds a value to reconstruct a component from a layout value at the given back distance (which
  // may be zero).
  // Upon failure, emits an error and returns null.
  virtual mlir::Value reconstructFromLayout(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            mlir::Value layout,
                                            size_t distance = 0) = 0;

  // Returns the component that returns the given type as a value type, if known.
  virtual ComponentTypeAttr getNameForType(mlir::Type type) = 0;

  // Returns the construct params needed by the given component, if
  // the given component declares a fixed number of arguments with
  // fixed types. Prefer to use buildConstruct to verify the number of
  // arguments and their types.  The component must exist.
  virtual std::optional<llvm::SmallVector<mlir::Type>>
  getConstructParams(Zhlt::ComponentTypeAttr component) = 0;

protected:
  ComponentManager() = default;
  virtual ~ComponentManager() = default;
};

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

#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Attrs.h.inc"

#include "zirgen/Dialect/ZHLT/IR/Interfaces.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/ComponentOps.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Ops.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/BuiltinOps.h.inc"
