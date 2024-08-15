// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/R1CS/Conversion/R1CSToBigInt/PassDetail.h"
#include "zirgen/Dialect/R1CS/IR/R1CS.h"

using namespace mlir;
using namespace zirgen::R1CS;

namespace zirgen::R1CSToBigInt {

namespace {

struct NoR1CSTarget : public ConversionTarget {
  NoR1CSTarget(MLIRContext& ctx) : ConversionTarget(ctx) {
    addLegalOp<mlir::ModuleOp>();
    addLegalDialect<BigInt::BigIntDialect>();
    addIllegalDialect<R1CS::R1CSDialect>();
  }
};

struct ConvertDef : public OpConversionPattern<R1CS::DefOp> {
  using OpConversionPattern<R1CS::DefOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(R1CS::DefOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    uint32_t bits = 64;
    uint64_t label = op.getLabel();
    bool isPublic = op.getIsPublic();
    rewriter.replaceOpWithNewOp<BigInt::DefOp>(op, bits, label, isPublic);
    return success();
  }
};

struct ConvertMul : public OpConversionPattern<R1CS::MulOp> {
  using OpConversionPattern<R1CS::MulOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(R1CS::MulOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto wire = adaptor.getWire();
    // Multiply the wire value by the constant
    mlir::APInt value = op.getValue();
    mlir::Type type = rewriter.getIntegerType(value.getBitWidth());
    auto valueAttr = rewriter.getIntegerAttr(type, value);
    auto valueExp = rewriter.create<BigInt::ConstOp>(loc, valueAttr);
    auto mulExp = rewriter.create<BigInt::MulOp>(loc, wire, valueExp);
    // Reduce by the field prime
    mlir::APInt prime = op.getPrime();
    type = rewriter.getIntegerType(prime.getBitWidth());
    auto primeAttr = rewriter.getIntegerAttr(type, prime);
    auto primeExp = rewriter.create<BigInt::ConstOp>(loc, primeAttr);
    rewriter.replaceOpWithNewOp<BigInt::ReduceOp>(op, mulExp, primeExp);
    return success();
  }
};

struct ConvertSum : public OpConversionPattern<R1CS::SumOp> {
  using OpConversionPattern<R1CS::SumOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(R1CS::SumOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<BigInt::AddOp>(op, lhs, rhs);
    return success();
  }
};

struct ConvertConstrain : public OpConversionPattern<R1CS::ConstrainOp> {
  using OpConversionPattern<R1CS::ConstrainOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(R1CS::ConstrainOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto a = adaptor.getA();
    auto b = adaptor.getB();
    auto prod = rewriter.create<BigInt::MulOp>(loc, a, b);
    auto c = adaptor.getC();
    if (c) {
      auto diff = rewriter.create<BigInt::SubOp>(loc, prod, c);
      rewriter.replaceOpWithNewOp<BigInt::EqualZeroOp>(op, diff);
    } else {
      rewriter.replaceOpWithNewOp<BigInt::EqualZeroOp>(op, prod);
    }
    return success();
  }
};

struct R1CSToBigIntPass : public R1CSToBigIntBase<R1CSToBigIntPass> {

  void runOnOperation() override {
    auto* ctx = &getContext();
    auto module = getOperation();

    NoR1CSTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertDef>(ctx);
    patterns.insert<ConvertMul>(ctx);
    patterns.insert<ConvertSum>(ctx);
    patterns.insert<ConvertConstrain>(ctx);
    if (applyFullConversion(module, target, std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // end namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createR1CSToBigIntPass() {
  return std::make_unique<R1CSToBigIntPass>();
}

} // namespace zirgen::R1CSToBigInt
