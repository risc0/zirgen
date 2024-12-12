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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "zirgen/Dialect/ZStruct/IR/Types.h"

#include <set>
#include <vector>

using namespace mlir;
using namespace zirgen::ZStruct;

namespace zirgen::Zhlt {

namespace {

bool isAlloc(Zhlt::ConstructOp op) {
  return op.getCallee() == "NondetReg";
}

struct Layout {
  std::vector<FieldInfo> fields;
  std::set<std::string> names;
  Layout() {}
  explicit Layout(LayoutType t) : fields(t.getFields()) {
    for (auto& fi : fields) {
      names.insert(fi.name.str());
    }
  }
  void erase(std::string name) {
    names.erase(name);
    for (size_t i = 0; i < fields.size(); ++i) {
      if (fields[i].name == name) {
        fields.erase(i + fields.begin());
        return;
      }
    }
    throw std::runtime_error("Failed to erase nonexistent field " + name);
  }
  void insert(std::string name, Type type) {
    if (contains(name)) {
      throw std::runtime_error("Failed to insert duplicate field " + name);
    }
    names.insert(name);
    FieldInfo field;
    field.name = StringAttr::get(type.getContext(), name);
    field.type = type;
    fields.push_back(field);
  }
  bool contains(std::string name) const { return names.find(name) != names.end(); }
};

using LayoutMap = llvm::DenseMap<LayoutType, Layout>;
using TypeMap = llvm::DenseMap<mlir::Type, mlir::Type>;

class rebuild {
  LayoutMap& layouts;
  TypeMap oldToNew;
  Type build(Type original) {
    // If this type has been rebuilt, use the new version instead.
    auto rebuilt = oldToNew.find(original);
    if (rebuilt != oldToNew.end()) {
      return rebuilt->second;
    }
    MLIRContext* ctx = original.getContext();
    if (LayoutType st = dyn_cast<LayoutType>(original)) {
      // If the type is a layout struct, rebuild it.
      auto found = layouts.find(st);
      if (found != layouts.end()) {
        Layout& layout = found->second;
        for (size_t i = 0; i < layout.fields.size(); ++i) {
          layout.fields[i].type = build(layout.fields[i].type);
        }
        auto repl = LayoutType::get(ctx, st.getId(), layout.fields);
        oldToNew[original] = repl;
        return repl;
      }
    } else if (auto arr = dyn_cast<LayoutArrayType>(original)) {
      // Similarly, rebuild the element types of array layouts
      Type builtElementType = build(arr.getElement());
      auto repl = LayoutArrayType::get(ctx, builtElementType, arr.getSize());
      oldToNew[original] = repl;
      return repl;
    }
    return original;
  }

public:
  rebuild(LayoutMap& layouts) : layouts(layouts) {
    for (auto& pair : layouts) {
      build(pair.first);
    }
  }
  operator TypeMap() { return oldToNew; }
};

class LayoutConverter : public TypeConverter {
public:
  LayoutConverter(TypeMap& tm) : oldToNew(tm) {
    addConversion([&](Type type) {
      auto found = tm.find(type);
      return found != tm.end() ? found->second : type;
    });
  }
  TypeMap& oldToNew;
};

struct LayoutTarget : public ConversionTarget {
  LayoutTarget(MLIRContext& ctx, TypeConverter& tc) : ConversionTarget(ctx) {
    addDynamicallyLegalOp<Zhlt::ComponentOp>([&](Zhlt::ComponentOp comp) -> bool {
      for (Type t : comp.getArgumentTypes()) {
        if (!tc.isLegal(t))
          return false;
      }
      for (Type t : comp.getResultTypes()) {
        if (!tc.isLegal(t))
          return false;
      }
      for (Block& block : comp.getBody()) {
        for (Type t : block.getArgumentTypes()) {
          if (!tc.isLegal(t))
            return false;
        }
      }
      return true;
    });
    markUnknownOpDynamicallyLegal([&](Operation* op) -> bool {
      for (Type t : op->getResultTypes()) {
        if (!tc.isLegal(t)) {
          return false;
        }
      }
      for (Type t : op->getOperandTypes()) {
        if (!tc.isLegal(t)) {
          return false;
        }
      }
      return true;
    });
  }
};

class TypeReplacementPattern : public ConversionPattern {
public:
  TypeReplacementPattern(TypeConverter& converter, MLIRContext* context, PatternBenefit benefit = 0)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), benefit, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<NamedAttribute, 4> newAttr;
    llvm::append_range(newAttr, op->getAttrs());
    llvm::SmallVector<Type, 4> newResults;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), newResults))) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    OperationState state(op->getLoc(),
                         op->getName().getStringRef(),
                         operands,
                         newResults,
                         newAttr,
                         op->getSuccessors());
    for (Region& r : op->getRegions()) {
      Region* newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      if (failed(getTypeConverter()->convertSignatureArgs(newRegion->getArgumentTypes(), result))) {
        return rewriter.notifyMatchFailure(op, "argument type conversion failed");
      }
      rewriter.applySignatureConversion(&newRegion->front(), result);
    }
    Operation* newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertComponent : public OpConversionPattern<Zhlt::ComponentOp> {
  using OpConversionPattern<Zhlt::ComponentOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(Zhlt::ComponentOp comp,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const TypeConverter* converter = getTypeConverter();

    // Convert argument types.
    size_t numArgs = comp.getNumArguments();
    TypeConverter::SignatureConversion signature(numArgs);
    for (size_t i = 0; i < numArgs; ++i) {
      auto t = comp.getArgument(i).getType();
      if (failed(converter->convertSignatureArg(i, t, signature))) {
        return rewriter.notifyMatchFailure(comp, "failed to convert argument type");
      }
    }
    auto outArgs = signature.getConvertedTypes();

    // Convert result types.
    auto inResults = comp.getCallableResults();
    SmallVector<Type> outResults;
    if (failed(converter->convertTypes(inResults, outResults))) {
      return rewriter.notifyMatchFailure(comp, "failed to convert result types");
    }

    // Create complete converted function type from arg & result types.
    // Update the function op to use the new type signature.
    MLIRContext* ctx = comp.getContext();
    auto outFuncType = mlir::FunctionType::get(ctx, outArgs, outResults);
    rewriter.startOpModification(comp);
    comp.setType(outFuncType);
    auto body = &comp.getBody();
    if (failed(rewriter.convertRegionTypes(body, *converter, &signature))) {
      return failure();
    }
    rewriter.finalizeOpModification(comp);
    return success();
  }
};

LogicalResult convert(Operation* op, TypeMap& oldToNew) {
  // Create a type converter and a conversion target.
  // Apply a conversion which legalizes to the target.
  auto ctx = op->getContext();
  LayoutConverter converter(oldToNew);
  LayoutTarget target(*ctx, converter);
  RewritePatternSet patterns(ctx);
  patterns.insert<TypeReplacementPattern>(converter, ctx);
  patterns.insert<ConvertComponent>(converter, ctx);
  return applyFullConversion(op, target, std::move(patterns));
}

void eraseStorage(Zhlt::ConstructOp op, LayoutMap& layouts) {
  Operation* layoutOp = op.getLayout().getDefiningOp();
  if (auto layoutLook = dyn_cast<LookupOp>(layoutOp)) {
    Type layoutType = layoutLook.getBase().getType();
    if (LayoutType st = dyn_cast<LayoutType>(layoutType)) {
      layouts[st].erase(layoutLook.getMember().str());
    }
  }
}

struct Hoist : public OpRewritePattern<Zhlt::ComponentOp> {
  LayoutMap& layouts;
  Hoist(MLIRContext* ctx, LayoutMap& layouts) : OpRewritePattern(ctx), layouts(layouts) {}

  LogicalResult matchAndRewrite(Zhlt::ComponentOp op, PatternRewriter& rewriter) const final {
    // If a parameter of this function is only ever used as an argument to
    // NondetReg, it should be hoisted out to all the calling functions.
    auto ret = failure();
    Region& body = op.getBody();
    MLIRContext* ctx = op.getContext();
    for (size_t argIndex = 0; argIndex != op.getConstructParam().size(); ++argIndex) {
      // Examine each parameter.
      Value argValue = body.getArgument(argIndex);
      if (!usedOnlyInAlloc(argValue)) {
        continue;
      }
      Type regType = replaceArgUses(body, argValue, argIndex, rewriter);
      // Change the component's function signature; the parameter types are
      // duplicated.
      auto compArgTypes = body.getArgumentTypes();
      auto compRetTypes = op.getResultTypes();
      op.setFunctionType(FunctionType::get(ctx, compArgTypes, compRetTypes));

      // Iterate through the sites which construct this component. Wrap each
      // corresponding argument value in a call to NondetReg before passing
      // it in, thereby matching the new type signature.
      op->getParentOfType<ModuleOp>().walk([&](Zhlt::ConstructOp cons) {
        if (cons.getCallee() != op.getSymName()) {
          return;
        }
        auto loc = cons.getLoc();
        rewriter.setInsertionPoint(cons.getOperation());
        Value allocArg = cons.getOperands()[argIndex];
        Zll::ValType allocType = dyn_cast<Zll::ValType>(allocArg.getType());
        // Reserve storage for the allocation we are moving.
        Value layout = cast<LookupOp>(cons.getLayout().getDefiningOp()).getBase();
        Layout& l = layouts[dyn_cast<LayoutType>(layout.getType())];
        // Efficiently find a unique name for this new storage field.
        size_t constructIndex = l.fields.size();
        std::string storageName;
        do {
          storageName = "@construct" + std::to_string(constructIndex++);
        } while (l.contains(storageName));
        RefType storageType = RefType::get(allocType.getContext(), allocType);
        l.insert(storageName, storageType);
        // Look up the storage we just generated.
        auto lookupOp = rewriter.create<LookupOp>(loc, storageType, layout, storageName);
        // Wrap argument #i in a call to NondetReg.
        auto ndr = rewriter.create<Zhlt::ConstructOp>(loc,
                                                      "NondetReg",
                                                      regType,
                                                      /*constructParams=*/allocArg,
                                                      /*layout=*/lookupOp.getResult());
        // Replace the original argument value with the result of this call.
        llvm::SmallVector<Value> args = cons.getConstructParam();
        args[argIndex] = ndr.getResult();
        rewriter.replaceOpWithNewOp<Zhlt::ConstructOp>(
            cons, cons.getCallee(), cons.getOutType(), args, cons.getLayout());
      });
      ret = success();
    }
    return ret;
  }

private:
  Type
  replaceArgUses(Region& body, Value argValue, size_t argIndex, PatternRewriter& rewriter) const {
    // Iterate through the uses of the old argument and replace each of
    // these calls with the new ref argument. We cannot use ranged for,
    // because we would retain the range after all uses were erased.
    Type regType;
    Value::use_iterator iter = argValue.use_begin();
    while (iter != argValue.use_end()) {
      Operation* user = iter.getUser();
      iter++;
      auto owner = dyn_cast<Zhlt::ConstructOp>(user);
      if (!regType) {
        regType = owner->getResultTypes()[0];
      }
      eraseStorage(owner, layouts);
      body.getArgument(argIndex).setType(regType);
      rewriter.replaceOp(owner, argValue);
    }
    return regType;
  }

  bool usedOnlyInAlloc(Value v) const {
    bool usedInAlloc = false;
    for (OpOperand& user : v.getUses()) {
      if (auto call = dyn_cast<Zhlt::ConstructOp>(user.getOwner())) {
        if (isAlloc(call)) {
          usedInAlloc = true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    }
    return usedInAlloc;
  }
};

struct Merge : public OpRewritePattern<Zhlt::ConstructOp> {
  LayoutMap& layouts;
  Merge(MLIRContext* ctx, LayoutMap& layouts) : OpRewritePattern(ctx), layouts(layouts) {}

  LogicalResult matchAndRewrite(Zhlt::ConstructOp op, PatternRewriter& rewriter) const final {
    // If the target operation is a call to NondetReg, it may be a candidate
    // for merging; we will examine its argument value.
    if (!isAlloc(op) || 2 != op.getOperands().size()) {
      return failure();
    }
    Operation* argOp = op.getOperands()[0].getDefiningOp();
    if (!argOp) {
      return failure();
    }
    // If the argument came from zstruct.lookup of @super, it may have been
    // the result of a previous allocation.
    LookupOp argLook = dyn_cast<LookupOp>(argOp);
    if (!argLook || argLook.getMember() != "@super") {
      return failure();
    }
    // Examine the struct whose super value we are allocating; did it come
    // from a previous call to NondetReg?
    Value source = argLook.getBase();
    if (!source.getDefiningOp()) {
      return failure();
    }
    auto lookBase = dyn_cast<Zhlt::ConstructOp>(source.getDefiningOp());
    if (!lookBase || !isAlloc(lookBase)) {
      return failure();
    }
    // We have passed the gauntlet: this operation is redundant and can be
    // replaced with the value provided to the super lookup operation. We
    // can also delete the storage provided in the layout structure.
    eraseStorage(op, layouts);
    rewriter.replaceOp(op, source);
    return success();
  }
};

struct HoistAllocsPass : public HoistAllocsBase<HoistAllocsPass> {
  void runOnOperation() override {
    auto* ctx = &getContext();
    ModuleOp mod = getOperation();
    LayoutMap layouts = collect(mod);
    RewritePatternSet patterns(ctx);
    patterns.insert<Hoist>(ctx, layouts);
    patterns.insert<Merge>(ctx, layouts);
    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    TypeMap replacements = rebuild(layouts);
    if (convert(mod, replacements).failed()) {
      return signalPassFailure();
    }
  }

  LayoutMap collect(ModuleOp mod) {
    LayoutMap out;
    mod.walk([&](Zhlt::ComponentOp comp) {
      if (auto lt = dyn_cast_or_null<LayoutType>(comp.getLayoutType())) {
        out.try_emplace(lt, lt);
        lt.walk([&](LayoutType type) { out.try_emplace(type, type); });
      }
    });
    return out;
  }
};

} // End namespace

std::unique_ptr<OperationPass<ModuleOp>> createHoistAllocsPass() {
  return std::make_unique<HoistAllocsPass>();
}

} // namespace zirgen::Zhlt
