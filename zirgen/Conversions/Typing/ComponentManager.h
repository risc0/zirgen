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

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"

namespace zirgen::Typing {

// Converts the given module from the ZHL dialect to the ZHLT dialect.
// Returns std::nullopt if unsuccessful.
std::optional<mlir::ModuleOp> typeCheck(mlir::MLIRContext&, mlir::ModuleOp);

// ComponentManager manages the component collection when generating
// typed components.  Callers can request a component and need not
// know whether the component is built in or generated from a
// zhl.component, or has some other special handling.
class ComponentManager {
public:
  Zhlt::ComponentOp
  getComponent(mlir::Location loc, llvm::StringRef name, llvm::ArrayRef<mlir::Attribute> typeArgs);

  // Looks up the given ZHLT component by mangled name and returns it
  // if it's already been generated.  Otherwise, returns null.
  Zhlt::ComponentOp lookupComponent(llvm::StringRef mangledName) {
    return zhltModule.lookupSymbol<Zhlt::ComponentOp>(mangledName);
  }

  // True if the given ZHL component is generic
  bool isGeneric(llvm::StringRef name);

  // Builds a value to reconstruct a component from a layout value at the given back distance (which
  // may be zero);
  mlir::Value reconstructFromLayout(mlir::OpBuilder& builder,
                                    mlir::Location loc,
                                    mlir::Value layout,
                                    size_t distance = 0);

private:
  struct TypeInfo;
  struct DebugListener;

  ComponentManager(mlir::ModuleOp zhlModule);
  ~ComponentManager();

  void gen();

  /// Resolves an unmangled component name to the component's definition in the
  /// (unlowered) ZHL module.
  Zhl::ComponentOp getUnloweredComponent(llvm::StringRef name);

  mlir::FailureOr<Zhlt::ComponentOp> genGenericBuiltin(mlir::Location loc,
                                                       llvm::StringRef name,
                                                       llvm::StringRef mangledName,
                                                       llvm::ArrayRef<mlir::Attribute> typeArgs);

  /// True iff the given component type has already been partially instantiated,
  /// so that attempting to instantiate it again might loop infinitely.
  std::vector<TypeInfo>::reverse_iterator findIllegalRecursion(Zhl::ComponentOp,
                                                               llvm::ArrayRef<mlir::Attribute>);

  bool isGeneric(Zhl::ComponentOp component);

  mlir::MLIRContext* ctx;
  /// This stack tracks the set of partially instantiated components during
  /// lowering, analogously to how a call stack tracks the set of partially
  /// executed functions. It is used to detect recursion during lowering.
  std::vector<TypeInfo> componentStack;

  /// The module in the ZHL dialect which is being lowered
  mlir::ModuleOp zhlModule;

  /// The module in the ZHLT dialect which is the output of this lowering
  mlir::ModuleOp zhltModule;

  std::unique_ptr<DebugListener> debugListener;
  std::optional<mlir::AsmState> asmState;

  friend std::optional<mlir::ModuleOp> typeCheck(mlir::MLIRContext&, mlir::ModuleOp);
};

} // namespace zirgen::Typing
