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

#include "zirgen/compiler/layout/convert.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "llvm/ADT/SmallVector.h"

namespace zirgen {
namespace layout {

using namespace mlir;

namespace {

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
    addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) -> bool {
      for (Type t : func.getArgumentTypes()) {
        if (!tc.isLegal(t))
          return false;
      }
      for (Type t : func.getResultTypes()) {
        if (!tc.isLegal(t))
          return false;
      }
      for (Block& block : func.getBody()) {
        for (Type t : block.getArgumentTypes()) {
          if (!tc.isLegal(t))
            return false;
        }
      }
      return true;
    });
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
      for (NamedAttribute attr : op->getAttrs()) {
        if (auto tyAttr = dyn_cast<TypeAttr>(attr.getValue())) {
          if (!tc.isLegal(tyAttr.getValue())) {
            return false;
          }
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
    const TypeConverter* converter = getTypeConverter();
    llvm::SmallVector<NamedAttribute, 4> newAttr;
    llvm::append_range(newAttr, op->getAttrs());
    for (size_t i = 0; i < newAttr.size(); ++i) {
      NamedAttribute& natt = newAttr[i];
      if (auto tyAttr = dyn_cast<TypeAttr>(natt.getValue())) {
        mlir::Type oldTy = tyAttr.getValue();
        mlir::Type newTy;
        if (auto ft = dyn_cast<FunctionType>(oldTy)) {
          if (converter->isSignatureLegal(ft)) {
            continue;
          }
          // Make a new function type with converted operand & result types.
          MLIRContext* ctx = op->getContext();
          llvm::SmallVector<mlir::Type, 4> attparams;
          if (failed(converter->convertTypes(ft.getInputs(), attparams))) {
            return rewriter.notifyMatchFailure(op,
                                               "failed to convert attribute function param types");
          }
          llvm::SmallVector<mlir::Type, 1> attresults;
          if (failed(converter->convertTypes(ft.getResults(), attresults))) {
            return rewriter.notifyMatchFailure(op,
                                               "failed to convert attribute function result types");
          }
          newTy = mlir::FunctionType::get(ctx, attparams, attresults);
        } else {
          if (converter->isLegal(oldTy)) {
            continue;
          }
          newTy = converter->convertType(oldTy);
        }
        natt.setValue(mlir::TypeAttr::get(newTy));
      }
    }
    llvm::SmallVector<Type, 4> newResults;
    if (failed(converter->convertTypes(op->getResultTypes(), newResults))) {
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
      if (failed(converter->convertSignatureArgs(newRegion->getArgumentTypes(), result))) {
        return rewriter.notifyMatchFailure(op, "argument type conversion failed");
      }
      rewriter.applySignatureConversion(&newRegion->front(), result);
    }
    Operation* newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertFunc : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(func::FuncOp func,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const TypeConverter* converter = getTypeConverter();

    // Convert argument types.
    size_t numArgs = func.getNumArguments();
    TypeConverter::SignatureConversion signature(numArgs);
    for (size_t i = 0; i < numArgs; ++i) {
      auto t = func.getArgument(i).getType();
      if (failed(converter->convertSignatureArg(i, t, signature))) {
        return rewriter.notifyMatchFailure(func, "failed to convert argument type");
      }
    }
    auto outArgs = signature.getConvertedTypes();

    // Convert result types.
    auto inResults = func.getResultTypes();
    SmallVector<Type> outResults;
    if (failed(converter->convertTypes(inResults, outResults))) {
      return rewriter.notifyMatchFailure(func, "failed to convert result types");
    }

    // Create complete converted function type from arg & result types.
    // Update the function op to use the new type signature.
    MLIRContext* ctx = func.getContext();
    auto outFuncType = mlir::FunctionType::get(ctx, outArgs, outResults);
    rewriter.startOpModification(func);
    func.setType(outFuncType);
    auto body = &func.getBody();
    if (failed(rewriter.convertRegionTypes(body, *converter, &signature))) {
      return failure();
    }
    rewriter.finalizeOpModification(func);
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

} // namespace

LogicalResult convert(Operation* op, TypeMap& oldToNew) {
  // Create a type converter and a conversion target.
  // Apply a conversion which legalizes to the target.
  auto ctx = op->getContext();
  LayoutConverter converter(oldToNew);
  LayoutTarget target(*ctx, converter);
  RewritePatternSet patterns(ctx);
  patterns.insert<TypeReplacementPattern>(converter, ctx);
  patterns.insert<ConvertFunc>(converter, ctx);
  patterns.insert<ConvertComponent>(converter, ctx);
  return applyFullConversion(op, target, std::move(patterns));
}

} // namespace layout
} // namespace zirgen
