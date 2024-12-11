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

#include "zirgen/compiler/picus/picus.h"
#include "mlir/Dialect/Arith//IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
#include "zirgen/dsl/passes/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

#include <queue>
#include <set>

using namespace mlir;
using namespace zirgen::Zhlt;
using namespace zirgen::ZStruct;
using namespace zirgen::Zll;

namespace {

using Signal = StringAttr;
using SignalArray = ArrayAttr;
using SignalStruct = DictionaryAttr;
using AnySignal = Attribute;

template <typename F> void visit(AnySignal signal, F f) {
  if (!signal) {
    // no-op
  } else if (auto s = dyn_cast<Signal>(signal)) {
    f(s);
  } else if (auto arr = dyn_cast<SignalArray>(signal)) {
    for (auto elem : arr)
      visit(elem, f);
  } else if (auto str = dyn_cast<SignalStruct>(signal)) {
    for (auto field : str) {
      visit(field.getValue(), f);
    }
  }
}

std::string canonicalizeIdentifier(std::string ident) {
  for (char& ch : ident) {
    if (ch == '$' || ch == '@' || ch == ' ' || ch == ':' || ch == '<' || ch == '>' || ch == ',') {
      ch = '_';
    }
  }
  return ident;
}

class PicusPrinter {
public:
  PicusPrinter(llvm::raw_ostream& os) : os(os) {}

  void print(ModuleOp mod) {
    ctx = mod->getContext();
    this->mod = mod;
    os << "(prime-number 2013265921)\n";
    for (auto component : mod.getOps<ComponentOp>()) {
      if (component->hasAttr("picus")) {
        workQueue.push(component);
      }
    }

    while (!workQueue.empty()) {
      valuesToSignals.clear();
      auto component = workQueue.front();
      printComponent(component);
      workQueue.pop();
    }
  }

private:
  void printComponent(ComponentOp component) {
    if (done.count(component))
      return;

    os << "(begin-module " << canonicalizeIdentifier(component.getName().str()) << ")\n";

    // Non-layout parameters are inputs
    for (BlockArgument param : component.getConstructParam()) {
      if (isa<StringType>(param.getType()) || isa<VariadicType>(param.getType()))
        continue;
      AnySignal signal = signalize(freshName(), param.getType());
      declareSignals(signal, /*isInput=*/true);
      valuesToSignals.insert({param, signal});
      workQueue.push(lookupConstructor(param.getType()));
    }

    // The layout is an output
    if (auto layout = component.getLayout()) {
      AnySignal layoutSignal = signalize("layout", layout.getType());
      declareSignals(layoutSignal, /*isInput=*/false);
      valuesToSignals.insert({layout, layoutSignal});
    }

    // The result is an output
    AnySignal result = signalize("result", component.getOutType());
    declareSignals(result, /*isInput=*/false);
    valuesToSignals.insert({Value(), result});

    for (Operation& op : component.getBody().front()) {
      visitOp(&op);
    }

    os << "(end-module)\n\n";
    done.insert(component);
  }

  void visitOp(Operation* op) {
    // Switch over operation types, and emit corresponding Picus code and track
    // the mapping between MLIR values and Picus signals. Because Picus doesn't
    // have control flow, MapOp/ReduceOp are unrolled.
    llvm::TypeSwitch<Operation*>(op)
        .Case<ConstOp,
              StringOp,
              SubOp,
              VariadicPackOp,
              ExternOp,
              LoadOp,
              LookupOp,
              SubscriptOp,
              ConstructOp,
              EqualZeroOp,
              ArrayOp,
              PackOp,
              ReturnOp,
              GetGlobalLayoutOp,
              AliasLayoutOp>([&](auto op) { visitOp(op); })
        .Case<StoreOp, arith::ConstantOp>([](auto) { /* no-op */ })
        .Default([](Operation* op) { llvm::errs() << "unhandled op: " << *op << "\n"; });
  }

  void visitOp(ConstOp constant) {
    assert(constant.getCoefficients().size() == 1 && "not implemented");
    auto signal = Signal::get(ctx, freshName());
    os << "(assert (= " << signal.str() << " " << constant.getCoefficients()[0] << "))\n";
    valuesToSignals.insert({constant.getOut(), signal});
  }

  void visitOp(StringOp str) {
    auto signal = SignalStruct::get(ctx, {});
    valuesToSignals.insert({str.getOut(), signal});
  }

  void visitOp(VariadicPackOp pack) {
    // Picus doesn't support variadics at call interfaces, so we associate them
    // with null signals. To see that this is sound (but not complete), first
    // note that variadics can only be used as call parameters in Zirgen. Then
    // note that additional variadic inputs only increase the number of
    // deterministic signals and don't increase the number of output signals.
    // Furthermore, adding new constraints to a constraint set (i.e. those
    // associated with the variadic input signals) can only shrink the set of
    // satisfying assignments.
    valuesToSignals.insert({pack.getOut(), nullptr});
  }

  void visitOp(ExternOp ext) {
    for (Value result : ext.getOut()) {
      Signal signal = Signal::get(ctx, freshName());
      valuesToSignals.insert({result, signal});
    }
  }

  void visitOp(LoadOp load) {
    auto signal = cast<Signal>(valuesToSignals.at(load.getRef()));
    valuesToSignals.insert({load.getOut(), signal});
  }

  void visitOp(LookupOp lookup) {
    auto signal = cast<SignalStruct>(valuesToSignals.at(lookup.getBase()));
    auto subSignal = signal.get(lookup.getMember());
    valuesToSignals.insert({lookup.getOut(), subSignal});
  }

  void visitOp(SubscriptOp subscript) {
    auto signal = cast<SignalArray>(valuesToSignals.at(subscript.getBase()));

    SmallVector<OpFoldResult> results;
    if (failed(subscript.getIndex().getDefiningOp()->fold(results))) {
      llvm::errs() << "failed to resolve subscript index\n";
    }
    uint64_t index = cast<PolynomialAttr>(results[0].get<Attribute>())[0];
    auto subSignal = signal[index];
    valuesToSignals.insert({subscript.getOut(), subSignal});
  }

  void visitOp(ConstructOp construct) {
    workQueue.push(mod.lookupSymbol<ComponentOp>(construct.getCallee()));
    AnySignal result = signalize(freshName(), construct.getOutType());
    valuesToSignals.insert({construct.getOut(), result});

    os << "(call [";
    if (auto layout = construct.getLayout()) {
      AnySignal layoutSignal = valuesToSignals.at(layout);
      llvm::interleave(
          flatten(layoutSignal), os, [&](Signal s) { os << s.str(); }, " ");
      os << " ";
    }
    llvm::interleave(
        flatten(result), os, [&](Signal s) { os << s.str(); }, " ");
    os << "] " << construct.getCallee() << " [";
    llvm::interleave(
        construct.getConstructParam(),
        os,
        [&](Value arg) {
          llvm::interleave(
              flatten(valuesToSignals.at(arg)), os, [&](Signal s) { os << s.str(); }, " ");
        },
        " ");
    os << "])\n";
  }

  void visitOp(SubOp sub) {
    auto signal = Signal::get(ctx, freshName());
    valuesToSignals.insert({sub.getOut(), signal});

    os << "(assert (= " << signal.str() << " (- ";
    os << cast<Signal>(valuesToSignals.at(sub.getLhs())).str() << " ";
    os << cast<Signal>(valuesToSignals.at(sub.getRhs())).str();
    os << ")))\n";
  }

  void visitOp(EqualZeroOp eqz) {
    os << "(assert (= ";
    os << cast<Signal>(valuesToSignals.at(eqz.getIn())).str();
    os << " 0))\n";
  }

  void visitOp(ArrayOp arr) {
    SmallVector<AnySignal> elements;
    for (auto arg : arr.getElements()) {
      AnySignal element = valuesToSignals.at(arg);
      elements.push_back(element);
    }
    auto signal = SignalArray::get(ctx, elements);
    valuesToSignals.insert({arr.getOut(), signal});
  }

  void visitOp(PackOp pack) {
    SmallVector<NamedAttribute> fields;
    for (auto [field, arg] : llvm::zip(pack.getOut().getType().getFields(), pack.getMembers())) {
      if (field.name.strref() == "@layout")
        continue;
      AnySignal member = valuesToSignals.at(arg);
      fields.emplace_back(field.name, member);
    }

    auto signal = SignalStruct::get(ctx, fields);
    valuesToSignals.insert({pack.getOut(), signal});
  }

  void visitOp(ReturnOp ret) {
    // The null value in valuesToSignals corresponds to the pre-declared output
    // of the component. Unify those signals with those of the return value.
    AnySignal outputSignal = valuesToSignals.at(Value());
    AnySignal returnSignal = valuesToSignals.at(ret.getValue());
    for (auto [outs, rets] : llvm::zip(flatten(outputSignal), flatten(returnSignal))) {
      os << "(assert (= " << outs.str() << " " << rets.str() << "))\n";
    }
  }

  void visitOp(GetGlobalLayoutOp get) {
    // This is sound but presumably not complete?
    AnySignal signal = signalize(freshName(), get.getType());
    declareSignals(signal, /*isInput=*/false);
    valuesToSignals.insert({get.getOut(), signal});
  }

  void visitOp(AliasLayoutOp alias) {
    auto lhs = valuesToSignals.at(alias.getLhs());
    auto rhs = valuesToSignals.at(alias.getRhs());
    for (auto [sl, sr] : llvm::zip(flatten(lhs), flatten(rhs))) {
      os << "(assert (= " << sl.str() << " " << sr.str() << "))\n";
    }
  }

  // Constructs a fresh signal structure corresponding to the given type
  AnySignal signalize(std::string prefix, Type type) {
    if (isa<ValType>(type) || isa<RefType>(type)) {
      return Signal::get(ctx, prefix);
    } else if (auto array = dyn_cast<ArrayLikeTypeInterface>(type)) {
      SmallVector<AnySignal> elements;
      for (size_t i = 0; i < array.getSize(); i++) {
        std::string name = prefix + "_" + std::to_string(i);
        elements.push_back(signalize(name, array.getElement()));
      }
      return SignalArray::get(ctx, elements);
    } else if (auto str = dyn_cast<StructType>(type)) {
      SmallVector<NamedAttribute> fields;
      for (auto field : str.getFields()) {
        if (field.name.strref() == "@layout")
          continue;
        std::string name = prefix + "_" + canonicalizeIdentifier(field.name.str());
        fields.emplace_back(field.name, signalize(name, field.type));
      }
      return SignalStruct::get(ctx, fields);
    } else if (auto str = dyn_cast<LayoutType>(type)) {
      SmallVector<NamedAttribute> fields;
      for (auto field : str.getFields()) {
        std::string name = prefix + "_" + canonicalizeIdentifier(field.name.str());
        fields.emplace_back(field.name, signalize(name, field.type));
      }
      return SignalStruct::get(ctx, fields);
    } else if (isa<StringType>(type) || isa<VariadicType>(type)) {
      return nullptr;
    } else {
      llvm::errs() << type << "\n";
      throw std::runtime_error("signalizing unhandled type");
    }
  }

  // Returns a flattened list of all the signal names in a signal structure.
  SmallVector<Signal> flatten(AnySignal signal) {
    SmallVector<Signal> flattened;
    visit(signal, [&](Signal s) { flattened.push_back(s); });
    return flattened;
  }

  void declareSignals(AnySignal signal, bool isInput) {
    visit(signal, [&](Signal s) { declareSignal(s, isInput); });
  }

  void declareSignal(Signal signal, bool isInput) {
    os << "(" << (isInput ? "input " : "output ") << signal.str() << ")\n";
  }

  ComponentOp lookupConstructor(Type type) {
    auto mangledName = mangledTypeName(type);
    return mod.lookupSymbol<ComponentOp>(mangledName);
  }

  std::string freshName() { return "x" + std::to_string(nameCounter++); }

  MLIRContext* ctx;
  ModuleOp mod;
  llvm::raw_ostream& os;
  std::queue<ComponentOp> workQueue;
  std::set<ComponentOp> done;
  llvm::DenseMap<Value, AnySignal> valuesToSignals;
  unsigned nameCounter = 0;
};

} // namespace

void printPicus(ModuleOp mod, llvm::raw_ostream& os) {
  PassManager pm(mod->getContext());
  pm.addPass(zirgen::dsl::createInlineForPicusPass());
  pm.addPass(createUnrollPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(mod))) {
    llvm::errs() << "Preprocessing for Picus failed";
    return;
  }

  PicusPrinter(os).print(mod);
}
