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
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h"
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

template <typename F> void visit(AnySignal signal, F f, bool visitedLayout = false) {
  if (!signal) {
    // no-op
  } else if (auto s = dyn_cast<Signal>(signal)) {
    f(s);
  } else if (auto arr = dyn_cast<SignalArray>(signal)) {
    for (auto elem : arr)
      visit(elem, f, visitedLayout);
  } else if (auto str = dyn_cast<SignalStruct>(signal)) {
    for (auto field : str) {
      if (!visitedLayout || field.getName() != "@layout") {
        visitedLayout = true;
        visit(field.getValue(), f, visitedLayout);
      }
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

// Determine if a value is used approximately once in the emitted *Picus code*.
// Values are typically emitted where they are used, but may also be emitted
// more times if they occur in a mux selector.
bool isUsedOnce(Value val) {
  return val.hasOneUse() && !isa<SwitchOp>(val.getUses().begin()->getOwner());
}

class MuxDefCollector {
public:
  struct SelectedValue {
    Value selector;
    Value value;
  };

  // A list of definitions/aliases for a value "outside the mux" in different
  // arms of the mux
  using DefCollection = SmallVector<SelectedValue>;

  // A mapping of all the captured values defined/aliased in a mux to their
  // various definitions
  using MuxMapping = DenseMap<Value, DefCollection>;

  bool empty() const { return stack.empty(); }

  void enterMux() { stack.emplace_back(); }

  MuxMapping exitMux() { return stack.pop_back_val(); }

  void nextArm(Value selector) { currentSelector = selector; }

  void addDef(Value outer, Value inner) { stack.back()[outer].push_back({currentSelector, inner}); }

  void removeValue(Value value) { stack.back().erase(value); }

private:
  // Maintain a MuxMapping for each level of mux nesting during traversal
  SmallVector<MuxMapping> stack;

  Value currentSelector;
};

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
    nameCounter = 0;

    outputSignalCounter = 0;
    assert(muxDefCollector.empty());
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

    // The result is an output
    AnySignal result = signalize("result", component.getOutType());
    declareSignals(result, SignalType::Output);
    valuesToSignals.insert({Value(), result});

    // The layout is an output
    if (auto layout = component.getLayout()) {
      AnySignal layoutSignal = cast<SignalStruct>(result).get("@layout");
      if (!layoutSignal) {
        // And it's still an output even if we prune the @layout member
        layoutSignal = signalize("layout", layout.getType());
        declareSignals(layoutSignal, SignalType::Output);
      }
      valuesToSignals.insert({layout, layoutSignal});
    }

    // Picus doesn't currently handle components without any output signals,
    // which means we need to inline these in the short term for successful
    // analysis. But it also means the component is only adding constraints,
    // so it should probably marked with picus_inline so this information is
    // available to Picus for its analysis.
    if (outputSignalCounter == 0) {
      component->emitWarning(
          "This component has a trivial output. Did you mean to add the picus_inline attribute?");
    }

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
        .Case<BitAndOp, ExternOp, InRangeOp, InvOp, IsZeroOp, ModOp>(
            [&](auto op) { visitNondetOp(op); })
        .Case<AddOp, SubOp, MulOp>([&](auto op) { visitBinaryPolyOp(op); })
        .Case<ConstOp,
              StringOp,
              VariadicPackOp,
              NegOp,
              ExternOp,
              LoadOp,
              LookupOp,
              SubscriptOp,
              ConstructOp,
              EqualZeroOp,
              Zhlt::BackOp,
              SwitchOp,
              YieldOp,
              ArrayOp,
              PackOp,
              ReturnOp,
              GetGlobalLayoutOp,
              AliasLayoutOp,
              zirgen::Zhlt::BackOp,
              DirectiveOp>([&](auto op) { visitOp(op); })
        .Case<StoreOp, YieldOp, arith::ConstantOp>([](auto) { /* no-op */ })
        .Default([](Operation* op) { llvm::errs() << "unhandled op: " << *op << "\n"; });
  }

  // For nondeterministic operations, mark all results as fresh signals.
  void visitNondetOp(Operation* op) {
    for (Value result : op->getResults()) {
      assert(result.getType() == Zhlt::getValType(ctx));
      Signal signal = Signal::get(ctx, freshName());
      valuesToSignals.insert({result, signal});
    }
  }

  void visitOp(ConstOp constant) {
    // Optimization: instead of creating a fresh signal for a constant op, make
    // a pseudosignal whose "name" is the constant literal. That way, every time
    // the signal is printed, we end up inlining the literal instead.
    assert(constant.getCoefficients().size() == 1 && "not implemented");
    auto signal = Signal::get(ctx, std::to_string(constant.getCoefficients()[0]));
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
    uint64_t index = UINT64_MAX;
    auto attr = results[0].get<Attribute>();
    if (auto polyAttr = dyn_cast<PolynomialAttr>(attr)) {
      index = polyAttr[0];
    } else if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      index = getIndexVal(intAttr);
    }
    auto subSignal = signal[index];
    valuesToSignals.insert({subscript.getOut(), subSignal});
  }

  void visitOp(ConstructOp construct) {
    workQueue.push(mod.lookupSymbol<ComponentOp>(construct.getCallee()));
    AnySignal layoutSignal;
    if (auto layout = construct.getLayout()) {
      layoutSignal = valuesToSignals.at(layout);
    }
    AnySignal result = signalize(freshName(), construct.getOutType(), /*layout=*/layoutSignal);
    valuesToSignals.insert({construct.getOut(), result});

    os << "(call [";
    if (layoutSignal) {
      llvm::interleave(
          flatten(layoutSignal), os, [&](Signal s) { os << s.str(); }, " ");
      os << " ";
    }
    llvm::interleave(
        flatten(result, /*skipLayout=*/true), os, [&](Signal s) { os << s.str(); }, " ");
    os << "] " << canonicalizeIdentifier(construct.getCallee().str()) << " [";
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

  void visitOp(NegOp neg) {
    // Optimization: if the result is only used once, use a pseudosignal to
    // inline the expression at the point of use instead of creating a dedicated
    // signal.
    std::string expr = "(- 0 " + cast<Signal>(valuesToSignals.at(neg.getIn())).str() + ")";

    if (isUsedOnce(neg->getResult(0))) {
      valuesToSignals.insert({neg.getOut(), Signal::get(ctx, expr)});
    } else {
      auto signal = Signal::get(ctx, freshName());
      valuesToSignals.insert({neg.getOut(), signal});
      os << "(assert (= " << signal.str() << " " << expr << "))\n";
    }
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

    // Optimization: if the result is only used once, use a pseudosignal to
    // inline the expression at the point of use instead of creating a dedicated
    // signal.
    std::string expr = "(" + std::string(symbol) + " " +
                       cast<Signal>(valuesToSignals.at(op->getOperand(0))).str() + " " +
                       cast<Signal>(valuesToSignals.at(op->getOperand(1))).str() + ")";

    if (isUsedOnce(op->getResult(0))) {
      valuesToSignals.insert({op->getResult(0), Signal::get(ctx, expr)});
    } else {
      auto signal = Signal::get(ctx, freshName());
      valuesToSignals.insert({op->getResult(0), signal});
      os << "(assert (= " << signal.str() << " " << expr << "))\n";
    }
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

    muxDefCollector.enterMux();
    for (Region& arm : mux.getArms()) {
      muxDefCollector.nextArm(mux.getSelector()[arm.getRegionNumber()]);
      assert(arm.hasOneBlock());
      for (Operation& op : arm.front()) {
        visitOp(&op);
      }
      os << "; mark mux arm\n";
    }

    // Optimization: if the mux result is never used, don't signalize it
    if (mux.getOut().use_empty()) {
      muxDefCollector.removeValue(mux.getOut());
    } else {
      AnySignal outSignal = signalize("mux_" + freshName(), mux.getType());
      valuesToSignals.insert({mux.getOut(), outSignal});
    }

    for (auto [value, selectedValues] : muxDefCollector.exitMux()) {
      // If a value isn't defined on all mux arms, it's value is undefined. This
      // fact isn't reflected by emitting a linear combination, but we'll cross
      // that bridge if and when we need to.
      assert(selectedValues.size() == mux.getSelector().size());
      SmallVector<SmallVector<Signal>> armSignals;
      armSignals.reserve(selectedValues.size());
      for (auto selVal : selectedValues) {
        armSignals.push_back(flatten(valuesToSignals.at(selVal.value)));
      }

      SmallVector<Signal> valueSignals = flatten(valuesToSignals.at(value));
      for (size_t i = 0; i < valueSignals.size(); i++) {
        os << "(assert (= " << valueSignals[i].str();
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
    }
    os << "; end mux\n";
  }

  void visitOp(YieldOp yield) {
    auto mux = cast<SwitchOp>(yield->getParentOp());
    muxDefCollector.addDef(mux.getOut(), yield.getValue());
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
      if (field.isPrivate)
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
      // Skip emitting vacuous constraints (a = a). These can come from the same
      // layout occuring at different levels of nesting within an @super member.
      if (outs != rets) {
        os << "(assert (= " << outs.str() << " " << rets.str() << "))\n";
      }
    }
  }

  void visitOp(GetGlobalLayoutOp get) {
    // The globals have a single unique value that is shared with the verifier,
    // so we can count on these always being deterministic.
    AnySignal signal = signalize(freshName(), get.getType());
    declareSignals(signal, SignalType::AssumeDeterministic);
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

    assert(interveningSelectors.size() <= 1);
    if (interveningSelectors.empty()) {
      auto lhsSignal = valuesToSignals.at(lhs);
      auto rhsSignal = valuesToSignals.at(rhs);
      for (auto [sl, sr] : llvm::zip(flatten(lhsSignal), flatten(rhsSignal))) {
        os << "(assert (= " << sl.str() << " " << sr.str() << "))\n";
      }
    } else {
      muxDefCollector.addDef(lhs, rhs);
    }

    // auto conditionalize = [&](Signal signal) {
    //   if (interveningSelectors.empty()) {
    //     os << signal.str();
    //   } else {
    //     for (Value s : interveningSelectors)
    //       os << "(* " << cast<Signal>(valuesToSignals.at(s)).str() << " ";
    //     os << signal.str();
    //     for (size_t i = 0; i < interveningSelectors.size(); i++)
    //       os << ")";
    //   }
    // };

    // auto lhsSignal = valuesToSignals.at(lhs);
    // auto rhsSignal = valuesToSignals.at(rhs);
    // for (auto [sl, sr] : llvm::zip(flatten(lhsSignal), flatten(rhsSignal))) {
    //   os << "(assert (= ";
    //   conditionalize(sl);
    //   os << " ";
    //   conditionalize(sr);
    //   os << "))\n";
    // }
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

  void visitOp(DirectiveOp directive) {
    if (directive.getName() == "AssumeRange") {
      auto args = directive.getArgs();
      auto low = cast<Signal>(valuesToSignals.at(args[0]));
      auto x = cast<Signal>(valuesToSignals.at(args[1]));
      auto high = cast<Signal>(valuesToSignals.at(args[2]));
      os << "(assume (<= " << low.str() << " " << x.str() << "))\n";
      os << "(assume (< " << x.str() << " " << high.str() << "))\n";
    } else if (directive.getName() == "AssertRange") {
      auto args = directive.getArgs();
      auto low = cast<Signal>(valuesToSignals.at(args[0]));
      auto x = cast<Signal>(valuesToSignals.at(args[1]));
      auto high = cast<Signal>(valuesToSignals.at(args[2]));
      os << "(assert (<= " << low.str() << " " << x.str() << "))\n";
      os << "(assert (< " << x.str() << " " << high.str() << "))\n";
    } else if (directive.getName() == "PicusHintEq") {
      auto leftSignal = cast<Signal>(valuesToSignals.at(directive.getArgs()[0]));
      auto rightSignal = cast<Signal>(valuesToSignals.at(directive.getArgs()[1]));
      os << "(assert (= " << leftSignal.str() << " " << rightSignal.str() << "))\n";
    } else if (directive.getName() == "PicusInput") {
      auto signal = valuesToSignals.at(directive.getArgs()[0]);
      declareSignals(signal, SignalType::AssumeDeterministic, /*skipLayout=*/true);
    } else {
      directive->emitError("Cannot lower this directive to Picus");
    }
  }

  // Constructs a fresh signal structure corresponding to the given type
  AnySignal signalize(std::string prefix, Type type, AnySignal layout = nullptr) {
    if (isa<ValType>(type) || isa<RefType>(type)) {
      return Signal::get(ctx, prefix);
    } else if (auto array = dyn_cast<ArrayLikeTypeInterface>(type)) {
      assert(isa<ArrayType>(type) || !layout);
      SmallVector<AnySignal> elements;
      for (size_t i = 0; i < array.getSize(); i++) {
        std::string name = prefix + "_" + std::to_string(i);
        AnySignal sublayout;
        if (auto arrLayout = cast_if_present<SignalArray>(layout)) {
          sublayout = arrLayout[i];
        }
        elements.push_back(signalize(name, array.getElement(), sublayout));
      }
      return SignalArray::get(ctx, elements);
    } else if (auto str = dyn_cast<StructType>(type)) {
      SmallVector<NamedAttribute> fields;
      // If we haven't generated a layout yet, generate it first. Then,
      // recursively pass along sublayouts for reuse, so that we don't generate
      // extra signals for registers at every level of nesting. For example,
      // foo.@layout.bar and foo.bar.@layout should refer to the same layout.
      if (!layout) {
        for (auto field : str.getFields()) {
          if (field.name == "@layout") {
            std::string name = prefix + "_" + canonicalizeIdentifier(field.name.str());
            layout = signalize(name, field.type);
            break;
          }
        }
      }
      for (auto field : str.getFields()) {
        if (field.name == "@layout") {
          fields.emplace_back(field.name, layout);
          continue;
        }
        if (!field.isPrivate) {
          std::string name = prefix + "_" + canonicalizeIdentifier(field.name.str());
          AnySignal sublayout;
          if (auto strLayout = cast_if_present<SignalStruct>(layout)) {
            sublayout = strLayout.get(field.name);
            // Mux members have their layouts wrapped in a structure with fields
            // for each arm and the common super, whereas their values are only
            // the common super. Navigate this extra indirection if necessary.
            auto sublayoutCasted = llvm::dyn_cast_or_null<SignalStruct>(sublayout);
            if (sublayoutCasted) {
              bool isMux = false;
              for (NamedAttribute field : sublayoutCasted) {
                if (field.getName().strref().starts_with("arm")) {
                  isMux = true;
                }
              }
              if (isMux) {
                sublayout = sublayoutCasted.get("@super");
              }
            }
          }
          fields.emplace_back(field.name, signalize(name, field.type, sublayout));
        }
      }
      return SignalStruct::get(ctx, fields);
    } else if (auto str = dyn_cast<LayoutType>(type)) {
      assert(!layout);
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
  SmallVector<Signal> flatten(AnySignal signal, bool skipLayout = false) {
    SmallVector<Signal> flattened;
    visit(
        signal, [&](Signal s) { flattened.push_back(s); }, /*visitedLayout=*/skipLayout);
    return flattened;
  }

  void declareSignals(AnySignal signal, SignalType type, bool skipLayout = false) {
    visit(
        signal, [&](Signal s) { declareSignal(s, type); }, /*visitedLayout=*/skipLayout);
  }

  void declareSignal(Signal signal, SignalType type) {
    std::string op;
    switch (type) {
    case SignalType::Input:
      op = "input";
      break;
    case SignalType::Output:
      outputSignalCounter++;
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
  MuxDefCollector muxDefCollector;
  unsigned nameCounter = 0;
  unsigned outputSignalCounter = 0;
};

} // namespace

void printPicus(ModuleOp mod, llvm::raw_ostream& os) {
  PassManager pm(mod->getContext());
  pm.addPass(zirgen::dsl::createGenerateBackPass());
  pm.addPass(zirgen::dsl::createInlineForPicusPass());
  pm.addPass(zirgen::Zhlt::createHoistCommonMuxCodePass(/*eager=*/true));
  pm.addPass(createUnrollPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(mod))) {
    llvm::errs() << "Preprocessing for Picus failed";
    return;
  }

  PicusPrinter(os).print(mod);
}
