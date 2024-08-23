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

class ComponentManagerImpl : public Zhlt::ComponentManager {
public:
  Zhlt::ComponentTypeAttr specialize(mlir::Location loc,
                                     Zhlt::ComponentTypeAttr orig,
                                     llvm::ArrayRef<mlir::Attribute> typeArgs) override;
  mlir::LogicalResult requireComponent(mlir::Location loc, Zhlt::ComponentTypeAttr name) override;
  mlir::LogicalResult requireComponentInferringType(mlir::Location loc,
                                                    Zhlt::ComponentTypeAttr& name,
                                                    mlir::ValueRange constructArgs) override;
  mlir::LogicalResult requireAbstractComponent(mlir::Location loc,
                                               Zhlt::ComponentTypeAttr name) override;
  mlir::Type getLayoutType(Zhlt::ComponentTypeAttr component) override;
  mlir::Type getValueType(Zhlt::ComponentTypeAttr component) override;
  mlir::Value buildConstruct(mlir::OpBuilder& builder,
                             mlir::Location loc,
                             Zhlt::ComponentTypeAttr component,
                             mlir::ValueRange constructArgs,
                             mlir::Value layout) override;
  mlir::Value reconstructFromLayout(mlir::OpBuilder& builder,
                                    mlir::Location loc,
                                    mlir::Value layout,
                                    size_t distance = 0) override;
  std::optional<llvm::SmallVector<mlir::Type>>
  getConstructParams(Zhlt::ComponentTypeAttr component) override;
  Zhlt::ComponentTypeAttr getNameForType(mlir::Type type) override;

private:
  struct TypeInfo;
  struct DebugListener;

private:
  ComponentManagerImpl(mlir::ModuleOp zhlModule);
  ~ComponentManagerImpl();

  Zhlt::ComponentOpInterface getComponentInterface(Zhlt::ComponentTypeAttr attr);

  Zhlt::ComponentOp
  genComponent(mlir::Location loc, llvm::StringRef name, llvm::ArrayRef<mlir::Attribute> typeArgs);

  void gen();

  /// Resolves an unmangled component name to the component's definition in the
  /// (unlowered) ZHL module.
  Zhl::ComponentOp getUnloweredComponent(llvm::StringRef name);

  // True if the given ZHL component needs to be specialized before use
  bool isGeneric(Zhl::ComponentOp component);

  /// True iff the given component type has already been partially instantiated,
  /// so that attempting to instantiate it again might loop infinitely.
  std::vector<TypeInfo>::reverse_iterator findIllegalRecursion(Zhl::ComponentOp,
                                                               llvm::ArrayRef<mlir::Attribute>);

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

  // Components that have been required
  llvm::DenseMap<Zhlt::ComponentTypeAttr, Zhlt::ComponentOpInterface> requiredComponents;

  // Layout types of components that have been required, and the
  // unmangled component name to use to reconstruct them.
  llvm::DenseMap<mlir::Type, Zhlt::ComponentTypeAttr> reconstructTypes;

  // Value types of components that have been required, and the
  // unmangled component name.
  llvm::DenseMap<mlir::Type, Zhlt::ComponentTypeAttr> valueTypes;

  friend std::optional<mlir::ModuleOp> typeCheck(mlir::MLIRContext&, mlir::ModuleOp);
};

Zhlt::ComponentOpInterface
ComponentManagerImpl::getComponentInterface(Zhlt::ComponentTypeAttr name) {
  return requiredComponents.at(name);
}

mlir::LogicalResult ComponentManagerImpl::requireComponent(Location loc,
                                                           Zhlt::ComponentTypeAttr name) {
  if (requiredComponents.contains(name))
    return success();

  // Attempt to find a base generic to specialize
  Zhlt::ComponentOpInterface intf =
      zhltModule.lookupSymbol<Zhlt::ComponentOpInterface>(name.getName());
  if (!intf)
    intf = zhlModule.lookupSymbol<Zhlt::ComponentOpInterface>(name.getName());

  // Otherwise, try to instantiate a generic zhl.component
  if (!intf) {
    if (auto zhlOp = getUnloweredComponent(name.getName()))
      intf = genComponent(loc, name.getName(), name.getTypeArgs());
  }

  if (!intf) {
    emitError(loc) << "Unable to instantiate component " << name;
    return failure();
  }

  if (failed(intf.requireComponent(this, loc, name)))
    return failure();

  requiredComponents[name] = intf;
  if (auto layout = intf.getLayoutType(this, name)) {
    reconstructTypes[layout] = name;
  }
  if (auto valType = intf.getValueType(this, name)) {
    valueTypes[valType] = name;
  }
  return success();
}

mlir::LogicalResult ComponentManagerImpl::requireComponentInferringType(
    Location loc, Zhlt::ComponentTypeAttr& name, ValueRange constructArgs) {
  if (requiredComponents.contains(name))
    return success();

  // Attempt to find an interface for specialization
  Zhlt::ComponentOpInterface intf =
      zhltModule.lookupSymbol<Zhlt::ComponentOpInterface>(name.getName());
  if (!intf)
    intf = zhlModule.lookupSymbol<Zhlt::ComponentOpInterface>(name.getName());

  if (intf) {
    intf.inferType(this, name, constructArgs);
  } else if (auto zhlOp = zhlModule.lookupSymbol<ComponentOp>(name.getName())) {
    // TODO: attempt to infer types for user-defined components
  }

  return requireComponent(loc, name);
}

mlir::LogicalResult ComponentManagerImpl::requireAbstractComponent(Location loc,
                                                                   Zhlt::ComponentTypeAttr name) {
  if (requiredComponents.contains(name))
    return success();
  std::string mangled = name.getMangledName();
  if (zhltModule.lookupSymbol(mangled))
    return success();
  if (getUnloweredComponent(name.getName()))
    return success();
  emitError(loc) << "Unable to find anything named " << name << " mangled: " << mangled;
  llvm::errs() << zhltModule;
  return failure();
}

Zhlt::ComponentTypeAttr ComponentManagerImpl::specialize(mlir::Location loc,
                                                         Zhlt::ComponentTypeAttr orig,
                                                         llvm::ArrayRef<mlir::Attribute> typeArgs) {
  if (!orig.getTypeArgs().empty()) {
    emitError(loc) << orig << " is already specialized";
    return {};
  }

  Zhlt::ComponentTypeAttr name = Zhlt::ComponentTypeAttr::get(ctx, orig.getName(), typeArgs);
  if (succeeded(requireComponent(loc, name)))
    return name;

  emitError(loc) << "Unable to specialize " << name;
  return {};
}

mlir::Type ComponentManagerImpl::getLayoutType(Zhlt::ComponentTypeAttr name) {
  return getComponentInterface(name).getLayoutType(this, name);
}

mlir::Type ComponentManagerImpl::getValueType(Zhlt::ComponentTypeAttr name) {
  return getComponentInterface(name).getValueType(this, name);
}

mlir::Value ComponentManagerImpl::buildConstruct(mlir::OpBuilder& builder,
                                                 mlir::Location loc,
                                                 Zhlt::ComponentTypeAttr name,
                                                 mlir::ValueRange constructArgs,
                                                 mlir::Value layout) {
  return getComponentInterface(name).buildConstruct(
      this, builder, loc, name, constructArgs, layout);
}

mlir::Value ComponentManagerImpl::reconstructFromLayout(mlir::OpBuilder& builder,
                                                        mlir::Location loc,
                                                        mlir::Value layout,
                                                        size_t distance) {
  Zhlt::ComponentTypeAttr name = reconstructTypes.lookup(layout.getType());
  if (!name) {
    llvm::errs() << "Don't know how to reconstruct " << layout.getType();
    return {};
  }

  return getComponentInterface(name).reconstructFromLayout(
      this, builder, loc, name, layout, distance);
}

std::optional<llvm::SmallVector<mlir::Type>>
ComponentManagerImpl::getConstructParams(Zhlt::ComponentTypeAttr name) {
  return getComponentInterface(name).getConstructParams(this, name);
}

Zhlt::ComponentTypeAttr ComponentManagerImpl::getNameForType(mlir::Type type) {
  return valueTypes.lookup(type);
}

struct ComponentManagerImpl::DebugListener : public OpBuilder::Listener {
  void notifyOperationInserted(Operation* op, IRRewriter::InsertPoint previous) override {
    LLVM_DEBUG({
      llvm::dbgs() << "Generated: ";
      op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
      llvm::dbgs() << "\n";
    });
  }
};

struct ComponentManagerImpl::TypeInfo {
  TypeInfo(ComponentOp component, ArrayRef<Attribute> typeArgs, Location requestedLoc)
      : component(component), typeArgs(typeArgs), requestedLoc(requestedLoc) {}

  ComponentOp component;
  llvm::SmallVector<Attribute> typeArgs;

  // The source location where the instantiation of this type was requested
  Location requestedLoc;
};

ComponentManagerImpl::ComponentManagerImpl(ModuleOp zhlModule)
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
ComponentManagerImpl::~ComponentManagerImpl() {}

std::vector<ComponentManagerImpl::TypeInfo>::reverse_iterator
ComponentManagerImpl::findIllegalRecursion(Zhl::ComponentOp component,
                                           ArrayRef<Attribute> typeArgs) {
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

ComponentOp ComponentManagerImpl::getUnloweredComponent(StringRef name) {
  for (Operation& op : zhlModule.getBodyRegion().front()) {
    ComponentOp component = llvm::dyn_cast<ComponentOp>(&op);
    if (component && component.getName() == name) {
      return component;
    }
  }
  return {};
}

Zhlt::ComponentOp ComponentManagerImpl::genComponent(Location requestedLoc,
                                                     StringRef name,
                                                     ArrayRef<Attribute> typeArgs) {
  Zhlt::ComponentTypeAttr newType = Zhlt::ComponentTypeAttr::get(ctx, name, typeArgs);
  std::string mangledName = newType.getMangledName();
  assert(!zhltModule.lookupSymbol(mangledName) && "Attempt to re-generate typed component");
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
          Zhlt::ComponentTypeAttr::get(ctx, recursion->component.getName(), recursion->typeArgs)
              .getMangledName();
      diag.attachNote(loc) << "`" << currentName << "` depends on `" << nextName << "`";
      currentName = nextName;
    }
    std::string nextName =
        Zhlt::ComponentTypeAttr::get(ctx, recursion->component.getName(), recursion->typeArgs)
            .getMangledName();
    diag.attachNote(requestedLoc) << "`" << currentName << "` depends on `" << mangledName << "`";
    throw MalformedIRException();
  }
  componentStack.emplace_back(component, typeArgs, requestedLoc);
  auto whenDone = llvm::make_scope_exit([&]() { componentStack.pop_back(); });

  OpBuilder builder = OpBuilder::atBlockEnd(zhltModule.getBody());
  LLVM_DEBUG({ builder.setListener(&*debugListener); });

  auto typed = generateTypedComponent(
      builder, this, component, builder.getStringAttr(mangledName), typeArgs);
  return typed;
}

bool ComponentManagerImpl::isGeneric(Zhl::ComponentOp component) {
  return !component.getBody().front().getOps<TypeParamOp>().empty();
}

void ComponentManagerImpl::gen() {
  bool containsErrors = false;
  for (ComponentOp c : zhlModule.getBodyRegion().front().getOps<ComponentOp>()) {
    if (isGeneric(c))
      continue;

    try {
      genComponent(c.getLoc(), c.getName(), /*typeArgs=*/{});
    } catch (const MalformedIRException& err) {
      // Failed components can be ignored
      containsErrors = true;
    }
  }
  if (containsErrors) {
    zhlModule.emitError("Module contains errors");
  }
}

std::optional<ModuleOp> typeCheck(MLIRContext& ctx, ModuleOp mod) {
  // Inject a diagnostic handler to detect if any errors occurred. We can't do
  // this with an early return because we want to report as many diagnostics as
  // possible, and doing it this way is simpler than detecting the errors where
  // they are emitted.
  bool containsErrors = false;
  ScopedDiagnosticHandler scopedHandler(&ctx, [&](Diagnostic&) {
    containsErrors = true;
    return failure();
  });

  ComponentManagerImpl componentManager(mod);
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
