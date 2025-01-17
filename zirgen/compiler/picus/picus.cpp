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
using namespace zirgen;
using namespace Zhlt;
using namespace ZStruct;
using namespace Zll;

namespace {

using Signal = StringAttr;
using SignalArray = ArrayAttr;
using SignalStruct = DictionaryAttr;
using AnySignal = Attribute;

enum class SignalType {
  Input,
  Output,
  AssumeDeterministic,
};

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
      assert(field.getName() != "@layout");
      visit(field.getValue(), f);
    }
  }
}

AnySignal getSuperSignal(AnySignal signal) {
  if (auto arr = dyn_cast<SignalArray>(signal)) {
    SmallVector<AnySignal> supers;
    for (auto elem : arr)
      supers.push_back(getSuperSignal(elem));
    return SignalArray::get(signal.getContext(), supers);
  } else if (auto str = dyn_cast<SignalStruct>(signal)) {
    return str.getNamed("@super")->getValue();
  } else {
    // Signal, nullptr, etc
    return nullptr;
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
      if (component->hasAttr("picus_analyze")) {
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
      declareSignals(signal, SignalType::Input);
      valuesToSignals.insert({param, signal});
      workQueue.push(lookupConstructor(param.getType()));
    }

    // The layout is an output
    if (auto layout = component.getLayout()) {
      AnySignal layoutSignal = signalize("layout", layout.getType());
      declareSignals(layoutSignal, SignalType::Output);
      valuesToSignals.insert({layout, layoutSignal});
    }

    // The result is an output
    AnySignal result = signalize("result", component.getOutType());
    declareSignals(result, SignalType::Output);
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
        .Case<InvOp, BitAndOp, ModOp, InRangeOp, ExternOp>([&](auto op) { visitNondetOp(op); })
        .Case<AddOp, SubOp, MulOp>([&](auto op) { visitBinaryPolyOp(op); })
        .Case<ConstOp,
              StringOp,
              VariadicPackOp,
              ExternOp,
              LoadOp,
              LookupOp,
              SubscriptOp,
              ConstructOp,
              EqualZeroOp,
              Zhlt::BackOp,
              SwitchOp,
              ArrayOp,
              PackOp,
              ReturnOp,
              GetGlobalLayoutOp,
              AliasLayoutOp,
              zirgen::Zhlt::BackOp>([&](auto op) { visitOp(op); })
        .Case<StoreOp, YieldOp, arith::ConstantOp>([](auto) { /* no-op */ })
        .Default([](Operation* op) { llvm::errs() << "unhandled op: " << *op << "\n"; });
  }

  // For nondeterministic operations, mark all results as fresh signals.
  void visitNondetOp(Operation* op) {
    for (Value result : op->getResults()) {
      Signal signal = Signal::get(ctx, freshName());
      valuesToSignals.insert({result, signal});
    }
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
      auto diag = subscript->emitError("failed to resolve subscript index\n");
      llvm::errs() << "index: " << *subscript.getIndex().getDefiningOp() << "\n";
      return;
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

  void visitBinaryPolyOp(Operation* op) {
    auto symbol = llvm::TypeSwitch<Operation*, const char*>(op)
                      .Case<AddOp>([](auto) { return "+"; })
                      .Case<SubOp>([](auto) { return "-"; })
                      .Case<MulOp>([](auto) { return "*"; })
                      .Default([&](auto) {
                        op->emitError("unknown binary poly op");
                        return nullptr;
                      });

    auto signal = Signal::get(ctx, freshName());
    valuesToSignals.insert({op->getResult(0), signal});

    os << "(assert (= " << signal.str() << " (" << symbol << " ";
    os << cast<Signal>(valuesToSignals.at(op->getOperand(0))).str() << " ";
    os << cast<Signal>(valuesToSignals.at(op->getOperand(1))).str();
    os << ")))\n";
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

  void visitOp(SwitchOp mux) {
    os << "; begin mux\n";

    SmallVector<Signal> selectorSignals;
    for (Value selector : mux.getSelector()) {
      selectorSignals.push_back(cast<Signal>(valuesToSignals.at(selector)));
    }

    SmallVector<SmallVector<Signal>> armSignals;

    for (Region& arm : mux.getArms()) {
      // Probably need to "turn off" AliasLayoutOps, since different arms may
      // write different values to the common super
      assert(arm.hasOneBlock());
      for (Operation& op : arm.front()) {
        visitOp(&op);
      }
      // Collect values yielded by each arm
      Value yielded = cast<YieldOp>(arm.front().getTerminator()).getValue();
      AnySignal signal = valuesToSignals.at(yielded);
      Type type = yielded.getType();
      while (type != mux.getType()) {
        signal = getSuperSignal(signal);
        type = Zhlt::getSuperType(type);
      }
      armSignals.push_back(flatten(signal));
      os << "; mark mux arm\n";
    }

    AnySignal outSignal = signalize("mux_" + freshName(), mux.getType());
    valuesToSignals.insert({mux.getOut(), outSignal});

    SmallVector<Signal> outSignals = flatten(outSignal);
    for (size_t i = 0; i < outSignals.size(); i++) {
      os << "(assert (= " << outSignals[i].str();
      for (size_t j = 0; j < armSignals.size(); j++) {
        if (j != armSignals.size() - 1)
          os << " (+";
        os << " (* " << selectorSignals[j].str() << " " << armSignals[j][i].str() << ")";
      }
      for (size_t j = 0; j < armSignals.size(); j++) {
        os << ")";
      }
      os << ")\n";
    }
    os << "; end mux\n";
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
      if (field.isPrivate || field.name.strref() == "@layout")
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
    SmallVector<Signal> outs = flatten(outputSignal);
    SmallVector<Signal> rets = flatten(returnSignal);
    assert(outs.size() == rets.size());
    for (auto [outs, rets] : llvm::zip(outs, rets)) {
      os << "(assert (= " << outs.str() << " " << rets.str() << "))\n";
    }
  }

  void visitOp(GetGlobalLayoutOp get) {
    // This is sound but presumably not complete?
    AnySignal signal = signalize(freshName(), get.getType());
    declareSignals(signal, SignalType::Output);
    valuesToSignals.insert({get.getOut(), signal});
  }

  void visitOp(AliasLayoutOp alias) {
    // If lhs and rhs have the same lifetime, then aliasing them is
    // straightforward: simply constrain lhs = rhs. If they have different
    // lifetimes, it's more complicated. Luckily, the only way for things to
    // have different lifetimes is because of muxing: values in a mux arm have
    // a lifetime strictly contained by the enclosing scope, and values in
    // different arms of the same mux have strictly non-overlapping lifetimes.
    // We assert that we never see aliases of the second type, as there
    // currently is no way to produce such aliases in Zirgen and it's not clear
    // how this ought to be handled. For the first case, we require the
    // corresponding signals to be equal only when the shorter-lived value is
    // live, so we produce Picus assertions of the form s * lhs = s * rhs: if
    // s = 0, this is trivially satisified, and if s = 1 it implies lhs = rhs.
    // If there are multiple intervening muxes, we take the product of all the
    // intervening selectors: product_i(s_i) * lhs = product_i(s_i) * rhs.

    Value lhs = alias.getLhs();
    Value rhs = alias.getRhs();
    Operation* lhsOp = lhs.getDefiningOp();
    Operation* rhsOp = rhs.getDefiningOp();
    Block* lhsBlock = lhs.getParentBlock();
    Block* rhsBlock = rhs.getParentBlock();

    // Find the longer-lived value
    Value shortLived;
    Value longLived;
    if (lhsBlock->findAncestorOpInBlock(*rhsOp) != nullptr) {
      shortLived = rhs;
      longLived = lhs;
    } else if (rhsBlock->findAncestorOpInBlock(*lhsOp) != nullptr) {
      shortLived = lhs;
      longLived = rhs;
    } else {
      assert(false && "cannot resolve relative lifetimes of aliased values");
    }

    SmallVector<Value> interveningSelectors;
    Region* x = shortLived.getParentRegion();
    while (x != longLived.getParentRegion()) {
      // All loops are unrolled and no other ops contain regions, so the
      // ancestor must be a SwitchOp.
      auto mux = cast<SwitchOp>(x->getParentOp());
      interveningSelectors.push_back(mux.getSelector()[x->getRegionNumber()]);
      x = x->getParentRegion();
    }

    auto conditionalize = [&](Signal signal) {
      if (interveningSelectors.empty()) {
        os << signal.str();
      } else {
        for (Value s : interveningSelectors)
          os << "(* " << cast<Signal>(valuesToSignals.at(s)).str() << " ";
        os << signal.str();
        for (size_t i = 0; i < interveningSelectors.size(); i++)
          os << ")";
      }
    };

    auto lhsSignal = valuesToSignals.at(lhs);
    auto rhsSignal = valuesToSignals.at(rhs);
    for (auto [sl, sr] : llvm::zip(flatten(lhsSignal), flatten(rhsSignal))) {
      os << "(assert (= ";
      conditionalize(sl);
      os << " ";
      conditionalize(sr);
      os << "))\n";
    }
  }

  void visitOp(Zhlt::BackOp back) {
    size_t distance = back.getDistance().getZExtValue();
    AnySignal signal = signalize(freshName(), back.getType());
    // We cannot handle the zero-distance case this way, so we expect that
    // all zero-distance backs will have been converted & inlined already.
    assert(distance > 0);
    declareSignals(signal, SignalType::AssumeDeterministic);
    valuesToSignals.insert({back.getOut(), signal});
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
        if (!field.isPrivate) {
          std::string name = prefix + "_" + canonicalizeIdentifier(field.name.str());
          fields.emplace_back(field.name, signalize(name, field.type));
        }
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

  void declareSignals(AnySignal signal, SignalType type) {
    visit(signal, [&](Signal s) { declareSignal(s, type); });
  }

  void declareSignal(Signal signal, SignalType type) {
    std::string op;
    switch (type) {
    case SignalType::Input:
      op = "input";
      break;
    case SignalType::Output:
      op = "output";
      break;
    case SignalType::AssumeDeterministic:
      op = "assume-deterministic";
      break;
    }
    os << "(" << op << " " << signal.str() << ")\n";
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
  pm.addPass(zirgen::dsl::createGenerateBackPass());
  pm.addPass(zirgen::dsl::createInlineForPicusPass());
  pm.addPass(createUnrollPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(mod))) {
    llvm::errs() << "Preprocessing for Picus failed";
    return;
  }

  PicusPrinter(os).print(mod);
}
