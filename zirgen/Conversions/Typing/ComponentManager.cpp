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

#include "zirgen/Conversions/Typing/ComponentManager.h"
#include "zirgen/Conversions/Typing/BuiltinComponents.h"
#include "zirgen/Conversions/Typing/ZhlComponent.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <set>
#include <variant>

#define DEBUG_TYPE "zhlt"

namespace zirgen::Typing {

using namespace mlir;
using namespace zirgen::Zhl;

/// True iff the component contains any TypeParamOps
bool ComponentManager::isGeneric(Zhl::ComponentOp component) {
  return !component.getBody().front().getOps<TypeParamOp>().empty();
}

bool ComponentManager::isGeneric(StringRef name) {
  if (name == "Array")
    return true;
  ComponentOp c = getUnloweredComponent(name);
  if (!c)
    return false;
  return isGeneric(c);
}

struct ComponentManager::DebugListener : public OpBuilder::Listener {
  void notifyOperationInserted(Operation* op, IRRewriter::InsertPoint previous) override {
    LLVM_DEBUG({
      llvm::dbgs() << "Generated: ";
      op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
      llvm::dbgs() << "\n";
    });
  }
};

struct ComponentManager::TypeInfo {
  TypeInfo(ComponentOp component, ArrayRef<Attribute> typeArgs, Location requestedLoc)
      : component(component), typeArgs(typeArgs), requestedLoc(requestedLoc) {}

  ComponentOp component;
  llvm::SmallVector<Attribute> typeArgs;

  // The source location where the instantiation of this type was requested
  Location requestedLoc;
};

ComponentManager::ComponentManager(ModuleOp zhlModule)
    : ctx(zhlModule.getContext())
    , zhlModule(zhlModule)
    , zhltModule(ModuleOp::create(zhlModule.getLoc())) {
  OpBuilder builder = OpBuilder::atBlockEnd(zhltModule.getBody());
  LLVM_DEBUG({
    llvm::errs() << "Adding listener\n";
    debugListener = std::make_unique<DebugListener>();
    builder.setListener(&*debugListener);
  });
  addBuiltins(builder);
}
ComponentManager::~ComponentManager() {}

ComponentOp ComponentManager::getUnloweredComponent(StringRef name) {
  for (Operation& op : zhlModule.getBodyRegion().front()) {
    ComponentOp component = llvm::dyn_cast<ComponentOp>(&op);
    if (component && component.getName() == name) {
      return component;
    }
  }
  return {};
}

std::vector<ComponentManager::TypeInfo>::reverse_iterator
ComponentManager::findIllegalRecursion(Zhl::ComponentOp component, ArrayRef<Attribute> typeArgs) {
  return std::find_if(componentStack.rbegin(), componentStack.rend(), [&](TypeInfo info) {
    if (info.component.getName() != component.getName()) {
      return false;
    }
    for (size_t i = 0; i < info.typeArgs.size(); i++) {
      if (typeArgs[i].isa<PolynomialAttr>()) {
        // Treat all numbers as equal for the sake of detecting recursion
      } else if (auto type = typeArgs[i].dyn_cast<StringAttr>()) {
        if (type != info.typeArgs[i]) {
          return false;
        }
      } else {
        assert(false && "not implemented");
      }
    }
    return true;
  });
}

Zhlt::ComponentOp ComponentManager::getComponent(Location requestedLoc,
                                                 StringRef name,
                                                 ArrayRef<Attribute> typeArgs) {
  std::string mangledName = Zhlt::mangledTypeName(name, typeArgs);
  if (auto op = zhltModule.lookupSymbol<Zhlt::ComponentOp>(mangledName)) {
    // Already lowered to ZHLT.
    return op;
  }

  auto genericBuiltin = genGenericBuiltin(requestedLoc, name, mangledName, typeArgs);
  if (failed(genericBuiltin)) {
    return {};
  } else {
    Zhlt::ComponentOp c = *genericBuiltin;
    if (c)
      return c;
  }

  auto component = getUnloweredComponent(name);
  if (!component) {
    if (typeArgs.empty()) {
      emitError(requestedLoc) << "unknown component `" << name << "`";
    } else {
      emitError(requestedLoc) << "Cannot find generic component " << name;
    }
    return {};
  }

  auto recursion = findIllegalRecursion(component, typeArgs);
  if (recursion != componentStack.rend()) {
    std::string currentName = mangledName;
    auto diag = emitError(component.getLoc())
                << "detected recursion in component `" << currentName << "`";
    while (recursion != componentStack.rbegin()) {
      recursion--;
      Location loc = recursion->requestedLoc;
      std::string nextName =
          Zhlt::mangledTypeName(recursion->component.getName(), recursion->typeArgs);
      diag.attachNote(loc) << "`" << currentName << "` depends on `" << nextName << "`";
      currentName = nextName;
    }
    std::string nextName =
        Zhlt::mangledTypeName(recursion->component.getName(), recursion->typeArgs);
    diag.attachNote(requestedLoc) << "`" << currentName << "` depends on `" << mangledName << "`";
    throw MalformedIRException();
  }

  componentStack.emplace_back(component, typeArgs, requestedLoc);
  auto whenDone = llvm::make_scope_exit([&]() { componentStack.pop_back(); });

  OpBuilder builder = OpBuilder::atBlockEnd(zhltModule.getBody());
  LLVM_DEBUG({ builder.setListener(&*debugListener); });

  return generateTypedComponent(
      builder, this, component, builder.getStringAttr(mangledName), typeArgs);
}

void ComponentManager::gen() {
  bool containsErrors = false;
  for (ComponentOp c : zhlModule.getBodyRegion().front().getOps<ComponentOp>()) {
    if (isGeneric(c))
      continue;

    try {
      getComponent(c.getLoc(), c.getName(), /*typeArgs=*/{});
    } catch (const MalformedIRException& err) {
      // Failed components can be ignored
      containsErrors = true;
    }
  }
  if (containsErrors) {
    zhlModule.emitError("Module contains errors");
  }
}

mlir::FailureOr<Zhlt::ComponentOp> ComponentManager::genGenericBuiltin(
    Location loc, StringRef name, StringRef mangledName, llvm::ArrayRef<Attribute> typeArgs) {
  if (name == "Array") {
    if (typeArgs.size() != 2) {
      return emitError(loc, "array type specialization must have two parameters");
    }

    std::string elementTypeName;
    if (typeArgs[0].isa<StringAttr>()) {
      elementTypeName = typeArgs[0].cast<StringAttr>().getValue();
    } else {
      return emitError(loc, "array element parameter must be a type name");
    }
    auto elemCtor = lookupComponent(elementTypeName);
    if (!elemCtor) {
      return emitError(loc, "array element type must be defined");
    }

    unsigned size = 0;
    if (typeArgs[1].isa<PolynomialAttr>()) {
      size = typeArgs[1].cast<PolynomialAttr>()[0];
    } else {
      return emitError(loc, "array size parameter must be an integer");
    }

    if (size < 1) {
      return emitError(loc, "array must have at least one element");
    }

    auto arrayType = ZStruct::ArrayType::get(ctx, elemCtor.getOutType(), size);
    auto layoutType = elemCtor.getLayoutType()
                          ? ZStruct::LayoutArrayType::get(ctx, elemCtor.getLayoutType(), size)
                          : nullptr;

    OpBuilder builder = OpBuilder::atBlockEnd(zhltModule.getBody());
    addArrayCtor(builder, mangledName, arrayType, layoutType, elemCtor.getConstructParamTypes());
    Zhlt::ComponentOp arrayComponent = lookupComponent(mangledName);
    assert(arrayComponent);
    return arrayComponent;
  } else {
    // No generic builtins matched, but no error encountered; return success.
    return Zhlt::ComponentOp{};
  }
}

mlir::Value ComponentManager::reconstructFromLayout(mlir::OpBuilder& builder,
                                                    mlir::Location loc,
                                                    mlir::Value layout,
                                                    size_t distance) {

  auto layoutType = llvm::dyn_cast<ZStruct::LayoutType>(layout.getType());
  if (!layoutType)
    return {};
  auto ctor = lookupComponent(layoutType.getId());
  if (!ctor)
    return {};
  auto backOp =
      builder.create<Zhlt::BackOp>(loc, ctor.getOutType(), layoutType.getId(), distance, layout);
  return backOp;
}

std::optional<ModuleOp> typeCheck(MLIRContext& ctx, ModuleOp mod) {
  // Inject a diagnostic handler to detect if any errors occurred. We can't do
  // this with an early return because we want to report as many diagnostics as
  // possible, and doing it this way is simpler than detecting the errors where
  // they are emitted.
  bool containsErrors = false;
  ScopedDiagnosticHandler scopedHandler(&ctx, [&](Diagnostic& diagnostic) {
    if (diagnostic.getSeverity() == DiagnosticSeverity::Error) {
      containsErrors = true;
    }
    return failure();
  });

  ComponentManager componentManager(mod);
  componentManager.gen();

  ModuleOp out = componentManager.zhltModule;

  if (!containsErrors && failed(verify(out))) {
    out->emitError("zhl module verification error");
  }

  if (containsErrors) {
    return std::nullopt;
  }
  return out;
}

} // namespace zirgen::Typing
