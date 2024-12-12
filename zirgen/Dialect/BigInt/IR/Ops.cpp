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

#include "mlir/IR/Builders.h"
#include "llvm/ADT/APSInt.h"

#include "risc0/core/util.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"

#define GET_OP_CLASSES
#include "zirgen/Dialect/BigInt/IR/Ops.cpp.inc"

using namespace mlir;
using risc0::ceilDiv;

// Additional comments on how type inference works for the BigInt dialect can be found in
// `test/type_infer.mlir`, including descriptions at the beginning of each op's suite of tests.

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

LogicalResult LoadOp::inferReturnTypes(MLIRContext* ctx,
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
  auto lhsType = cast<BigIntType>(adaptor.getLhs().getType());
  auto rhsType = cast<BigIntType>(adaptor.getRhs().getType());
  size_t maxCoeffs = std::max(lhsType.getCoeffs(), rhsType.getCoeffs());
  size_t maxPos = lhsType.getMaxPos() + rhsType.getMaxPos();
  size_t maxNeg = lhsType.getMaxNeg() + rhsType.getMaxNeg();
  // TODO: We could be more clever on minBits, but probably doesn't matter
  size_t minBits = maxNeg > 0 ? 0 : std::max(lhsType.getMinBits(), rhsType.getMinBits());
  out.push_back(BigIntType::get(ctx, maxCoeffs, maxPos, maxNeg, minBits));
  return success();
}

LogicalResult SubOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  auto lhsType = cast<BigIntType>(adaptor.getLhs().getType());
  auto rhsType = cast<BigIntType>(adaptor.getRhs().getType());
  size_t maxCoeffs = std::max(lhsType.getCoeffs(), rhsType.getCoeffs());
  size_t maxPos = lhsType.getMaxPos() + rhsType.getMaxNeg();
  size_t maxNeg = lhsType.getMaxNeg() + rhsType.getMaxPos();
  // TODO: We could be more clever on minBits, but probably doesn't matter
  out.push_back(BigIntType::get(ctx, maxCoeffs, maxPos, maxNeg, 0));
  return success();
}

LogicalResult MulOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  auto lhsType = cast<BigIntType>(adaptor.getLhs().getType());
  auto rhsType = cast<BigIntType>(adaptor.getRhs().getType());
  size_t coeffs = lhsType.getCoeffs() + rhsType.getCoeffs() - 1;
  // The maximum number of coefficient pairs from the inputs used to calculate an output coefficient
  size_t maxCoeffs = std::min(lhsType.getCoeffs(), rhsType.getCoeffs());
  // This calculation could overflow if size_t is 32 bits, so cast to 64 bits
  uint64_t maxPos = std::max((uint64_t)lhsType.getMaxPos() * rhsType.getMaxPos(),
                             (uint64_t)lhsType.getMaxNeg() * rhsType.getMaxNeg());
  // The next step can potentially overflow even 64 bits; but if we're already above 32 bits we'll
  // fail validation anyway. Therefore, skip this if we're above 32 bits
  if (maxPos < (uint64_t)1 << 32) {
    maxPos *= maxCoeffs;
  }
  // Clamp to size_t
  if (maxPos > std::numeric_limits<size_t>::max()) {
    maxPos = std::numeric_limits<size_t>::max();
  }
  // As with maxPos, this could overflow if size_t is 32 bits, so cast to 64 bits
  uint64_t maxNeg = std::max((uint64_t)lhsType.getMaxPos() * rhsType.getMaxNeg(),
                             (uint64_t)lhsType.getMaxNeg() * rhsType.getMaxPos());
  // The next step can potentially overflow even 64 bits; but if we're already above 32 bits we'll
  // fail validation anyway. Therefore, skip this if we're above 32 bits
  if (maxNeg < (uint64_t)1 << 32) {
    maxNeg *= maxCoeffs;
  }
  // Clamp to size_t
  if (maxNeg > std::numeric_limits<size_t>::max()) {
    maxNeg = std::numeric_limits<size_t>::max();
  }
  size_t minBits;
  if (lhsType.getMinBits() == 0 || rhsType.getMinBits() == 0) {
    // Note that this catches _both_ cases where the input might be zero _and_ cases where the input
    // might be negative, as type verification enforces that when minBits is zero, so is maxNeg.
    minBits = 0;
  } else {
    minBits = lhsType.getMinBits() + rhsType.getMinBits() - 1;
  }
  out.push_back(BigIntType::get(ctx, coeffs, maxPos, maxNeg, minBits));
  return success();
}

LogicalResult NondetRemOp::inferReturnTypes(MLIRContext* ctx,
                                            std::optional<Location> loc,
                                            Adaptor adaptor,
                                            SmallVectorImpl<Type>& out) {
  auto lhsType = cast<BigIntType>(adaptor.getLhs().getType());
  auto rhsType = cast<BigIntType>(adaptor.getRhs().getType());
  auto outBits = lhsType.getMaxPosBits();
  if (rhsType.getMaxPosBits() < outBits) {
    outBits = rhsType.getMaxPosBits();
  }
  size_t coeffsWidth = ceilDiv(outBits, kBitsPerCoeff);
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
  auto lhsType = cast<BigIntType>(adaptor.getLhs().getType());
  auto rhsType = cast<BigIntType>(adaptor.getRhs().getType());
  size_t outBits = lhsType.getMaxPosBits();
  if (rhsType.getMinBits() > 0) {
    outBits -= rhsType.getMinBits() - 1;
  }
  size_t coeffsWidth = ceilDiv(outBits, kBitsPerCoeff);
  // TODO: We could be more clever on minBits, but probably doesn't matter
  out.push_back(BigIntType::get(ctx,
                                /*coeffs=*/coeffsWidth,
                                /*maxPos=*/(1 << kBitsPerCoeff) - 1,
                                /*maxNeg=*/0,
                                /*minBits=*/0));
  return success();
}

LogicalResult NondetInvOp::inferReturnTypes(MLIRContext* ctx,
                                            std::optional<Location> loc,
                                            Adaptor adaptor,
                                            SmallVectorImpl<Type>& out) {
  auto rhsType = cast<BigIntType>(adaptor.getRhs().getType());
  size_t coeffsWidth = ceilDiv(rhsType.getMaxPosBits(), kBitsPerCoeff);
  out.push_back(BigIntType::get(ctx,
                                /*coeffs=*/coeffsWidth,
                                /*maxPos=*/(1 << kBitsPerCoeff) - 1,
                                /*maxNeg=*/0,
                                /*minBits=*/0));
  return success();
}

LogicalResult InvOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  auto rhsType = cast<BigIntType>(adaptor.getRhs().getType());
  size_t coeffsWidth = ceilDiv(rhsType.getMaxPosBits(), kBitsPerCoeff);
  out.push_back(BigIntType::get(ctx,
                                /*coeffs=*/coeffsWidth,
                                /*maxPos=*/(1 << kBitsPerCoeff) - 1,
                                /*maxNeg=*/0,
                                /*minBits=*/0));
  return success();
}

LogicalResult ReduceOp::inferReturnTypes(MLIRContext* ctx,
                                         std::optional<Location> loc,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type>& out) {
  auto lhsType = cast<BigIntType>(adaptor.getLhs().getType());
  auto rhsType = cast<BigIntType>(adaptor.getRhs().getType());
  auto outBits = lhsType.getMaxPosBits();
  if (rhsType.getMaxPosBits() < outBits) {
    outBits = rhsType.getMaxPosBits();
  }
  size_t coeffsWidth = ceilDiv(outBits, kBitsPerCoeff);
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
  cg.emitInvokeMacro(cg.getStringAttr("bigint_def"),
                     /*contextArgs=*/{"ctx"},
                     {
                         toConstantValue(cg, getContext(), getType().getCoeffs()),
                         cg.guessAttributeType(getLabelAttr()),
                         cg.guessAttributeType(getIsPublicAttr()),
                     });
}

void AddOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitInvokeMacro(cg.getStringAttr("bigint_add"),
                     {
                         getLhs(),
                         getRhs(),
                         toConstantValue(cg, getContext(), getType().getCoeffs()),
                     });
}

void SubOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitInvokeMacro(cg.getStringAttr("bigint_sub"),
                     {
                         getLhs(),
                         getRhs(),
                         toConstantValue(cg, getContext(), getType().getCoeffs()),
                     });
}

void MulOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitInvokeMacro(cg.getStringAttr("bigint_mul"),
                     {
                         getLhs(),
                         getRhs(),
                         toConstantValue(cg, getContext(), getType().getCoeffs()),
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
  cg.emitInvokeMacro(
      cg.getStringAttr("bigint_nondet_rem"),
      /*contextArgs=*/{"ctx"},
      {getLhs(), getRhs(), toConstantValue(cg, getContext(), getType().getCoeffs())});
}

void NondetQuotOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitInvokeMacro(
      cg.getStringAttr("bigint_nondet_quot"),
      /*contextArgs=*/{"ctx"},
      {getLhs(), getRhs(), toConstantValue(cg, getContext(), getType().getCoeffs())});
}

void NondetInvOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitInvokeMacro(
      cg.getStringAttr("bigint_nondet_inv"),
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
