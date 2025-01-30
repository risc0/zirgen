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

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Bufferize.h"
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h"
#include "zirgen/Dialect/ByteCode/Transforms/Schedule.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

using namespace mlir;
using namespace zirgen::Zll;

#define DEBUG_TYPE "schedule-zll"

namespace zirgen::ByteCode {

#define GEN_PASS_DEF_SCHEDULEZLL
#define GEN_PASS_DEF_CLONESIMPLEZLL
#define GEN_PASS_DEF_BUFFERIZEZLL
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h.inc"

struct ZllSchedule : public ScheduleInterface {
  size_t getValueRegs(mlir::Value value) override {
    return TypeSwitch<Type, size_t>(value.getType())
        .Case<ValType>([&](auto valType) { return valType.getFieldK(); })
        .Case<ConstraintType>([&](auto constraintType) { return 4; })
        .Default([&](auto) { return 0; });
  }

  bool isPure(mlir::Operation* op) override {
    if (llvm::isa<GetOp, GetGlobalOp, ConstOp>(op))
      return true;
    return mlir::isPure(op);
  }
};

struct ScheduleZllPass : public impl::ScheduleZllBase<ScheduleZllPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());

    Block* block = &funcOp.getBody().front();
    ZllSchedule schedule;
    scheduleBlock(block, schedule);
  }
};

namespace {

template <typename OpT> struct ClonePerUserPattern : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op, PatternRewriter& rewriter) const final {
    if (llvm::hasNItemsOrLess(op->getUsers(), 1))
      return failure();

    // Clone and pick one off
    Operation* cloned = rewriter.clone(*op);

    for (auto [oldResult, newResult] : llvm::zip_equal(op->getResults(), cloned->getResults())) {
      if (!oldResult.use_empty())
        oldResult.getUses().begin()->set(newResult);
    }
    return success();
  }
};

} // namespace

struct CloneSimpleZllPass : public impl::CloneSimpleZllBase<CloneSimpleZllPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());
    auto* ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ClonePerUserPattern<ConstOp>>(ctx);
    patterns.insert<ClonePerUserPattern<GetOp>>(ctx);
    patterns.insert<ClonePerUserPattern<GetGlobalOp>>(ctx);
    patterns.insert<ClonePerUserPattern<TrueOp>>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

struct ZllBufferize : public BufferizeInterface {
  ZllBufferize(MLIRContext* ctx) : kind(StringAttr::get(ctx, "fpBuffer")) {}

  std::pair</*intKind=*/mlir::StringAttr, /*index=*/size_t>
  getKindAndSize(mlir::Value value) override {
    return std::make_pair(
        kind,
        TypeSwitch<Type, size_t>(value.getType())
            .Case<ValType>([&](auto valType) { return valType.getFieldK(); })
            .Case<ConstraintType>([&](auto) { return 4; })
            .Default([&](auto) -> size_t { assert(0 && "Unknown type in zll bufferize"); }));
  }
  StringAttr kind;
};

struct BufferizeZllPass : public impl::BufferizeZllBase<BufferizeZllPass> {
  void runOnOperation() override {

    ZllBufferize zllBufferize(&getContext());
    if (failed(bufferize(getOperation(), zllBufferize))) {
      getOperation()->emitError("Unable to bufferize");
      signalPassFailure();
    }
  }
};

} // namespace zirgen::ByteCode
