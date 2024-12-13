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

#include "zirgen/Dialect/ZStruct/Transforms/RewritePatterns.h"
#include "mlir/IR/IRMapping.h"

namespace zirgen::ZStruct {

using namespace mlir;

LogicalResult UnrollMaps::matchAndRewrite(MapOp op, PatternRewriter& rewriter) const {
  Value in = op.getArray();
  auto inType = mlir::cast<ZStruct::ArrayLikeTypeInterface>(in.getType());
  auto outType = mlir::cast<ZStruct::ArrayType>(op.getOut().getType());

  llvm::SmallVector<Value, 8> mapped;

  Block& innerBlock = op.getBody().front();
  auto innerValArg = innerBlock.getArgument(0);
  Value innerLayoutArg;
  if (innerBlock.getNumArguments() > 1)
    innerLayoutArg = innerBlock.getArgument(1);

  auto yieldOp = llvm::cast<ZStruct::YieldOp>(innerBlock.getTerminator());
  auto innerReturnVal = yieldOp.getValue();

  for (size_t i = 0; i < inType.getSize(); i++) {
    IRMapping mapping;
    Value idx = rewriter.create<Zll::ConstOp>(op.getLoc(), i);
    Value inVal = rewriter.create<ZStruct::SubscriptOp>(op.getLoc(), in, idx);
    mapping.map(innerValArg, inVal);

    if (op.getLayout()) {
      Value inLayout = rewriter.create<ZStruct::SubscriptOp>(op.getLoc(), op.getLayout(), idx);
      assert(innerLayoutArg);
      mapping.map(innerLayoutArg, inLayout);
    }

    for (auto& innerOp : innerBlock.without_terminator()) {
      rewriter.clone(innerOp, mapping);
    }
    mapped.push_back(mapping.lookupOrDefault(innerReturnVal));
  }
  auto unrolled = outType.materialize(op.getLoc(), mapped, rewriter);
  rewriter.replaceOp(op, unrolled);
  return success();
}

LogicalResult UnrollReduces::matchAndRewrite(ReduceOp op, PatternRewriter& rewriter) const {
  Value inArray = op.getArray();
  auto inType = mlir::cast<ZStruct::ArrayType>(inArray.getType());

  Block& innerBlock = op.getBody().front();
  auto innerLhsArg = innerBlock.getArgument(0);
  auto innerRhsArg = innerBlock.getArgument(1);
  Value innerLayoutArg;
  if (innerBlock.getNumArguments() > 2)
    innerLayoutArg = innerBlock.getArgument(2);

  auto yieldOp = llvm::cast<ZStruct::YieldOp>(innerBlock.getTerminator());
  auto innerReturnVal = yieldOp.getValue();
  assert(innerReturnVal.getType() == innerLhsArg.getType());

  Value reduced = op.getInit();
  for (size_t i = 0; i < inType.getSize(); i++) {
    IRMapping mapping;
    mapping.map(innerLhsArg, reduced);

    Value idx = rewriter.create<Zll::ConstOp>(op.getLoc(), i);
    Value inVal = rewriter.create<ZStruct::SubscriptOp>(op.getLoc(), inArray, idx);
    mapping.map(innerRhsArg, inVal);

    if (op.getLayout()) {
      assert(innerLayoutArg);
      Value inLayout = rewriter.create<ZStruct::SubscriptOp>(op.getLoc(), op.getLayout(), idx);
      mapping.map(innerLayoutArg, inLayout);
    }

    for (auto& innerOp : innerBlock.without_terminator()) {
      rewriter.clone(innerOp, mapping);
    }
    reduced = mapping.lookup(innerReturnVal);
  }
  rewriter.replaceOp(op, reduced);
  return success();
}

LogicalResult SplitSwitchArms::matchAndRewrite(SwitchOp op, PatternRewriter& rewriter) const {
  if (!op->use_empty())
    return failure();

  rewriter.setInsertionPoint(op);
  for (auto [cond, arm] : llvm::zip(op.getSelector(), op.getArms())) {
    auto ifOp = rewriter.create<Zll::IfOp>(op.getLoc(), cond);
    auto termOp = arm.front().getTerminator();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(termOp);
    rewriter.replaceOpWithNewOp<Zll::TerminateOp>(termOp);
    ifOp.getInner().takeBody(arm);
  }
  rewriter.eraseOp(op);
  return success();
}

} // namespace zirgen::ZStruct
