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

/// The Zll (Zero-knowledge Low-Level) dialect is the core MLIR dialect for
/// representing ZK circuits in zirgen. It provides types for field elements,
/// extension-field elements, buffers, digests, and IOPs, along with ops for
/// arithmetic, buffer access, hashing, and proof-system interaction.
namespace zirgen::Zll {
class Interpreter;
class InterpVal;
class OpEvaluator;
class FieldAttr;
class BufferType;

/// Convenience factory for an unsigned 64-bit IntegerAttr.
inline mlir::IntegerAttr getUI64Attr(mlir::MLIRContext* ctx, uint64_t val) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned), val);
}

/// Return a human-readable string representation of an MLIR location.
std::string getLocString(mlir::Location loc);

/// Return the FieldAttr for the default prime field used by the circuit.
FieldAttr getDefaultField(mlir::MLIRContext* cxt);

/// Return the FieldAttr for the prime field identified by `fieldName`.
FieldAttr getField(mlir::MLIRContext* cxt, llvm::StringRef fieldName);

/// @name Codegen op traits
/// Op traits that influence how the codegen backend emits an operation.
/// Attach these to op definitions in TableGen to control the emitted syntax.
/// @{

/// Op should be emitted as an infix binary expression (e.g. `a + b`).
template <typename ConcreteType>
struct CodegenInfixOpTrait : public mlir::OpTrait::TraitBase<ConcreteType, CodegenInfixOpTrait> {};

/// Op carries MLIR properties that must be forwarded to the emitted call.
template <typename ConcreteType>
struct CodegenOpWithPropertiesTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CodegenOpWithPropertiesTrait> {};

/// Op should be silently dropped during code generation (e.g. debug-only ops).
template <typename ConcreteType>
struct CodegenSkipTrait : public mlir::OpTrait::TraitBase<ConcreteType, CodegenSkipTrait> {};

/// Op's result should always be inlined at every use site.
template <typename ConcreteType>
struct CodegenAlwaysInlineOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CodegenAlwaysInlineOpTrait> {};

/// Op's result should never be inlined; always assigned to a local variable.
template <typename ConcreteType>
struct CodegenNeverInlineOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CodegenNeverInlineOpTrait> {};

/// @}

/// @name Codegen type traits
/// Type traits that influence how codegen handles values of that type.
/// @{

/// Values of this type must be cloned when passed by value (Rust: `.clone()`).
template <typename ConcreteType>
struct CodegenNeedsCloneTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, CodegenNeedsCloneTypeTrait> {};

/// Values of this type must always be passed by (immutable) reference.
template <typename ConcreteType>
struct CodegenOnlyPassByReferenceTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, CodegenOnlyPassByReferenceTypeTrait> {};

/// Values of this type must be passed by mutable reference.
template <typename ConcreteType>
struct CodegenPassByMutRefTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, CodegenPassByMutRefTypeTrait> {};

/// This type is a layout type (holds buffer offsets, not runtime values).
template <typename ConcreteType>
struct CodegenLayoutTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, CodegenLayoutTypeTrait> {};

/// @}

/// Op trait for ops that can be evaluated by the Zll interpreter using a
/// simple value adaptor (all operands are scalar field elements).
template <typename ConcreteType>
class EvalOpAdaptor : public mlir::OpTrait::TraitBase<ConcreteType, EvalOpAdaptor> {};

/// Op trait for ops that can be evaluated by the Zll interpreter and whose
/// result type depends on the field in use.
template <typename ConcreteType>
class EvalOpFieldAdaptor : public mlir::OpTrait::TraitBase<ConcreteType, EvalOpFieldAdaptor> {};

/// Search enclosing regions for a block argument whose type matches one of the
/// given types T... . Stops at isolated-from-above boundaries. Useful for
/// locating implicit context arguments (e.g. IOP handles) without threading
/// them through every op.
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

/// Trait-based overload: search enclosing regions for a block argument whose
/// type carries the given type trait (e.g. `CodegenPassByMutRefTypeTrait`).
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

/// Walk up the op's parent chain to find the enclosing ModuleOp and return the
/// attribute of type `AttrT` stored on it. The attribute type must implement a
/// static `lookupModuleAttrName()` method returning the attribute's name.
template <typename AttrT> AttrT lookupModuleAttr(mlir::Operation* op) {
  while (!llvm::isa<mlir::ModuleOp>(op))
    op = op->getParentOp();
  AttrT result = op->getAttrOfType<AttrT>(AttrT::lookupModuleAttrName());
  assert(result && "Missing expected module attribute");
  return result;
}

/// Set the module-level attribute of type `AttrT` on the enclosing ModuleOp.
template <typename AttrT> void setModuleAttr(mlir::Operation* op, AttrT newValue) {
  while (!llvm::isa<mlir::ModuleOp>(op))
    op = op->getParentOp();
  op->setAttr(AttrT::lookupModuleAttrName(), newValue);
}

/// Re-infer the return type of `op` from its current operand types. Needed
/// after mutating operands in-place during canonicalization passes.
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
