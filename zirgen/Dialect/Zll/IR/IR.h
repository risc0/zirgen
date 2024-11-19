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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"

#include "zirgen/Dialect/Zll/IR/BigInt.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"
#include "zirgen/Dialect/Zll/IR/Enums.h.inc"
#include "zirgen/Dialect/Zll/IR/Field.h"
#include "zirgen/Dialect/Zll/IR/Types.h"

namespace zirgen::Zll {
class Interpreter;
class InterpVal;
class OpEvaluator;
class FieldAttr;
class BufferType;

inline mlir::IntegerAttr getUI64Attr(mlir::MLIRContext* ctx, uint64_t val) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned), val);
}

std::string getLocString(mlir::Location loc);

FieldAttr getDefaultField(mlir::MLIRContext* cxt);

FieldAttr getField(mlir::MLIRContext* cxt, llvm::StringRef fieldName);

template <typename ConcreteType>
struct CodegenInfixOpTrait : public mlir::OpTrait::TraitBase<ConcreteType, CodegenInfixOpTrait> {};

template <typename ConcreteType>
struct CodegenOpWithPropertiesTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CodegenOpWithPropertiesTrait> {};

template <typename ConcreteType>
struct CodegenSkipTrait : public mlir::OpTrait::TraitBase<ConcreteType, CodegenSkipTrait> {};

template <typename ConcreteType>
struct CodegenAlwaysInlineOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CodegenAlwaysInlineOpTrait> {};

template <typename ConcreteType>
struct CodegenNeverInlineOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CodegenNeverInlineOpTrait> {};

template <typename ConcreteType>
struct CodegenNeedsCloneTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, CodegenNeedsCloneTypeTrait> {};

template <typename ConcreteType>
struct CodegenOnlyPassByReferenceTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, CodegenOnlyPassByReferenceTypeTrait> {};

template <typename ConcreteType>
struct CodegenPassByMutRefTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, CodegenPassByMutRefTypeTrait> {};

template <typename ConcreteType>
struct CodegenLayoutTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, CodegenLayoutTypeTrait> {};

template <typename ConcreteType>
class EvalOpAdaptor : public mlir::OpTrait::TraitBase<ConcreteType, EvalOpAdaptor> {};

template <typename ConcreteType>
class EvalOpFieldAdaptor : public mlir::OpTrait::TraitBase<ConcreteType, EvalOpFieldAdaptor> {};

// lookupNearestImplicitArg looks up a block argument of the given
// type in the nearest enclosing region of the given operation.  It
// can be used to find e.g. a context argument that we don't want to
// keep track of everywhere by building.
template <typename... T>::mlir::Value lookupNearestImplicitArg(mlir::Operation* op) {
  while (op) {
    for (auto& region : op->getRegions()) {
      for (auto arg : region.getArguments()) {
        if (llvm::isa<T...>(arg.getType()))
          return arg;
      }
    }
    if (op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      break;
    op = op->getParentOp();
  }
  return {};
}

// This version of lookupNearestImplicitArg finds a type with the given type instead of searching
// for an exact type.
template <template <typename T> class Trait>
::mlir::Value lookupNearestImplicitArg(mlir::Operation* op) {
  while (op) {
    for (auto& region : op->getRegions()) {
      for (auto arg : region.getArguments()) {
        if (arg.getType().hasTrait<Trait>())
          return arg;
      }
    }
    if (op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      break;
    op = op->getParentOp();
  }
  return {};
}

// Looks up an attribute in the module that is or encloses the given
// operation.  The attribute must define the `lookupModuleAttrName`
// method to provide the name of the attribute.
template <typename AttrT> AttrT lookupModuleAttr(mlir::Operation* op) {
  while (!llvm::isa<mlir::ModuleOp>(op))
    op = op->getParentOp();
  AttrT result = op->getAttrOfType<AttrT>(AttrT::lookupModuleAttrName());
  assert(result && "Missing expected module attribute");
  return result;
}

template <typename AttrT> void setModuleAttr(mlir::Operation* op, AttrT newValue) {
  while (!llvm::isa<mlir::ModuleOp>(op))
    op = op->getParentOp();
  op->setAttr(AttrT::lookupModuleAttrName(), newValue);
}

// Re-infer the return type of the given operation, in case its input types have changed.
void reinferReturnType(mlir::InferTypeOpInterface op);

} // namespace zirgen::Zll

#include "zirgen/Dialect/Zll/IR/TypeInterfaces.h.inc"
#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/Zll/IR/Attrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/Zll/IR/Types.h.inc"

#include "zirgen/Dialect/Zll/IR/Attrs.h"
#include "zirgen/Dialect/Zll/IR/Interfaces.h.inc"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"

#include "zirgen/Dialect/Zll/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/Zll/IR/Ops.h.inc"
