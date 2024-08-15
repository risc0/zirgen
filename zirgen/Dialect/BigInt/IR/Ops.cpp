// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/Builders.h"
#include "llvm/ADT/APSInt.h"

#include "risc0/core/util.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"

#define GET_OP_CLASSES
#include "zirgen/Dialect/BigInt/IR/Ops.cpp.inc"

using namespace mlir;
using risc0::ceilDiv;

namespace zirgen::BigInt {

// Type inference

LogicalResult DefOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  size_t coeffsWidth = ceilDiv(adaptor.getBitWidth(), kBitsPerCoeff);
  out.push_back(BigIntType::get(ctx,
                                /*coeffs=*/coeffsWidth,
                                /*maxPos=*/(1 << kBitsPerCoeff) - 1,
                                /*maxNeg=*/0,
                                /*minBits=*/adaptor.getMinBits()));
  return success();
}

LogicalResult ConstOp::inferReturnTypes(MLIRContext* ctx,
                                        std::optional<Location> loc,
                                        Adaptor adaptor,
                                        SmallVectorImpl<Type>& out) {
  size_t coeffsWidth = ceilDiv(adaptor.getValue().getBitWidth(), kBitsPerCoeff);
  out.push_back(BigIntType::get(ctx,
                                /*coeffs=*/coeffsWidth,
                                /*maxPos=*/(1 << kBitsPerCoeff) - 1,
                                /*maxNeg=*/0,
                                /*minBits=*/adaptor.getValue().getActiveBits()));
  return success();
}

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValueAttr();
}

LogicalResult AddOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  auto lhsType = adaptor.getLhs().getType().cast<BigIntType>();
  auto rhsType = adaptor.getRhs().getType().cast<BigIntType>();
  size_t maxCoeffs = std::max(lhsType.getCoeffs(), rhsType.getCoeffs());
  size_t maxPos = std::max(lhsType.getMaxPos(), rhsType.getMaxPos());
  size_t maxNeg = std::max(lhsType.getMaxNeg(), rhsType.getMaxNeg());
  size_t minBits = std::max(lhsType.getMinBits(), rhsType.getMinBits());
  out.push_back(BigIntType::get(ctx, maxCoeffs, maxPos, maxNeg, minBits));
  return success();
}

LogicalResult SubOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  auto lhsType = adaptor.getLhs().getType().cast<BigIntType>();
  auto rhsType = adaptor.getRhs().getType().cast<BigIntType>();
  size_t maxCoeffs = std::max(lhsType.getCoeffs(), rhsType.getCoeffs());
  size_t maxPos = std::max(lhsType.getMaxPos(), rhsType.getMaxNeg());
  size_t maxNeg = std::max(lhsType.getMaxNeg(), rhsType.getMaxPos());
  // TODO: We could be more clever on minBits, but probably doesn't matter
  out.push_back(BigIntType::get(ctx, maxCoeffs, maxPos, maxNeg, 0));
  return success();
}

LogicalResult MulOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  auto lhsType = adaptor.getLhs().getType().cast<BigIntType>();
  auto rhsType = adaptor.getRhs().getType().cast<BigIntType>();
  size_t maxCoeffs = std::max(lhsType.getCoeffs(), rhsType.getCoeffs());
  size_t totCoeffs = lhsType.getCoeffs() + rhsType.getCoeffs();
  size_t maxPos = std::max(lhsType.getMaxPos() * rhsType.getMaxPos(),
                           lhsType.getMaxNeg() * rhsType.getMaxNeg()) *
                  maxCoeffs;
  size_t maxNeg = std::max(lhsType.getMaxPos() * rhsType.getMaxNeg(),
                           lhsType.getMaxNeg() * rhsType.getMaxPos()) *
                  maxCoeffs;
  size_t minBits;
  if (lhsType.getMinBits() == 0 || rhsType.getMinBits() == 0) {
    minBits = 0;
  } else {
    minBits = lhsType.getMinBits() + rhsType.getMinBits() - 1;
  }
  out.push_back(BigIntType::get(ctx, totCoeffs, maxPos, maxNeg, minBits));
  return success();
}

LogicalResult NondetRemOp::inferReturnTypes(MLIRContext* ctx,
                                            std::optional<Location> loc,
                                            Adaptor adaptor,
                                            SmallVectorImpl<Type>& out) {
  auto rhsType = adaptor.getRhs().getType().cast<BigIntType>();
  size_t coeffsWidth = ceilDiv(rhsType.getMaxBits(), kBitsPerCoeff);
  out.push_back(BigIntType::get(ctx,
                                /*coeffs=*/coeffsWidth,
                                /*maxPos=*/(1 << kBitsPerCoeff) - 1,
                                /*maxNeg=*/0,
                                /*minBits=*/0));
  return success();
}

LogicalResult NondetQuotOp::inferReturnTypes(MLIRContext* ctx,
                                             std::optional<Location> loc,
                                             Adaptor adaptor,
                                             SmallVectorImpl<Type>& out) {
  auto lhsType = adaptor.getLhs().getType().cast<BigIntType>();
  auto rhsType = adaptor.getRhs().getType().cast<BigIntType>();
  size_t outBits = lhsType.getMaxBits();
  if (rhsType.getMinBits() > 0) {
    outBits -= rhsType.getMinBits() - 1;
  }
  size_t coeffsWidth = ceilDiv(outBits, kBitsPerCoeff);
  out.push_back(BigIntType::get(ctx,
                                /*coeffs=*/coeffsWidth,
                                /*maxPos=*/(1 << kBitsPerCoeff) - 1,
                                /*maxNeg=*/0,
                                /*minBits=*/0 /*TODO: maybe better bound? */
                                ));
  return success();
}

LogicalResult ReduceOp::inferReturnTypes(MLIRContext* ctx,
                                         std::optional<Location> loc,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type>& out) {
  auto rhsType = adaptor.getRhs().getType().cast<BigIntType>();
  size_t coeffsWidth = ceilDiv(rhsType.getMaxBits(), kBitsPerCoeff);
  out.push_back(BigIntType::get(ctx,
                                /*coeffs=*/coeffsWidth,
                                /*maxPos=*/(1 << kBitsPerCoeff) - 1,
                                /*maxNeg=*/0,
                                /*minBits=*/0));
  return success();
}

namespace {

codegen::CodegenValue toConstantValue(codegen::CodegenEmitter& cg, MLIRContext* ctx, size_t val) {
  return cg.guessAttributeType(IntegerAttr::get(ctx, APSInt(APInt(64, val))));
}

} // namespace

void DefOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitFuncCall(cg.getStringAttr("def"),
                  /*contextArgs=*/{"ctx"},
                  {
                      toConstantValue(cg, getContext(), getType().getCoeffs()),
                      cg.guessAttributeType(getLabelAttr()),
                      cg.guessAttributeType(getIsPublicAttr()),
                  });
}

void EqualZeroOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitFuncCall(cg.getStringAttr("eqz"),
                  /*contextArgs=*/{"ctx"},
                  {getIn(),
                   toConstantValue(cg, getContext(), getIn().getType().getCarryOffset()),
                   toConstantValue(cg, getContext(), getIn().getType().getCarryBytes())});
}

void NondetRemOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitFuncCall(cg.getStringAttr("nondet_rem"),
                  /*contextArgs=*/{"ctx"},
                  {getLhs(), getRhs(), toConstantValue(cg, getContext(), getType().getCoeffs())});
}

void NondetQuotOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitFuncCall(cg.getStringAttr("nondet_quot"),
                  /*contextArgs=*/{"ctx"},
                  {getLhs(), getRhs(), toConstantValue(cg, getContext(), getType().getCoeffs())});
}

void ConstOp::emitExpr(codegen::CodegenEmitter& cg) {
  auto bytePoly = fromAPInt(getValue(), getType().getCoeffs());
  SmallVector<codegen::EmitPart> macroArgs;
  for (int32_t val : bytePoly) {
    macroArgs.push_back([val](codegen::CodegenEmitter& cg) { cg << val; });
  }
  cg.emitInvokeMacro(cg.getStringAttr("bigint_const"), {"ctx"}, macroArgs);
}

} // namespace zirgen::BigInt
